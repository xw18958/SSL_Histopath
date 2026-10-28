#!/usr/bin/env python3
"""Finish a post-selection DINOv3 classification test interrupted by FD exhaustion.

Only loads the already-frozen probe and checkpoint. Never tunes or trains.
Refuses to overwrite any test output and retains the exclusive test marker.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from pannuke_ssl.ssl_framework.downstream import _extract, _split_rows
from pannuke_ssl.ssl_framework.external_datasets import load_external_manifest
from pannuke_ssl.ssl_framework.trainer import load_checkpoint
from pannuke_ssl.ssl_methods.registry import build_method
from pannuke_ssl.probe import _evaluate
from pannuke_ssl.utils import atomic_json_dump, write_csv

def recover(checkpoint: Path, output: Path, manifest_root: Path, model_root: Path):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    dataset="gashissdb_binary"
    marker_path=output/"test_started.json"
    if not marker_path.is_file():
        raise RuntimeError("No previous test marker: never run a new evaluation through recovery")
    if (output/"test_metrics.json").exists() or (output/"test_predictions.npz").exists():
        raise RuntimeError("Test outputs already exist, refuse overwrite")
    if (output/"test_recovery_started.json").exists():
        raise RuntimeError("Recovery already started; never repeat held-out test")
    if not (output/"best_linear_probe.pt").is_file() or not (output/"probe_selection.json").is_file():
        raise RuntimeError("Frozen probe and selection were not saved")
    marker=json.loads(marker_path.read_text())
    if marker != dict(encoder_epoch=200, probe_selected=True, dataset=dataset, num_classes=2):
        raise RuntimeError("Unexpected original immutable marker")
    sel=json.loads((output/"probe_selection.json").read_text())
    state=torch.load(output/"best_linear_probe.pt",map_location="cpu",weights_only=False)
    if state["selection"]!=sel or sel["test_used"] is not False or sel["num_classes"]!=2:
        raise RuntimeError("Frozen probe selections do not match")
    raw=torch.load(checkpoint,map_location="cpu",weights_only=False)
    if raw["framework"]!="ssl_standard_v1" or raw["method_name"]!="dinov3" or raw["epoch"]!=200:
        raise RuntimeError("Wrong checkpoint")
    c=raw["config"]
    c["backbone"]["config_dir"]=str(model_root/"plip_model")
    c["manifests"]["root"]=str(manifest_root)
    # Only the number of IO worker subprocesses changes; image preprocessing,
    # patches, classifier, splits, metrics and weights remain unchanged.
    c["downstream"]["num_workers"]=0
    ds=load_external_manifest(dataset,manifest_root)
    if (output/"dataset_provenance.json").exists():
        old=json.loads((output/"dataset_provenance.json").read_text())
        if old["manifest_sha256"]!=ds.manifest_sha256:
            raise RuntimeError("Manifest changed since original selection")
    by=_split_rows(c,ds)
    device=torch.device("cuda:0")
    method=build_method(c,device)
    load_checkpoint(method,checkpoint,device)
    enc=method.encoder.eval().requires_grad_(False)
    classes=list(ds.class_names)
    if len(classes)!=2:
        raise RuntimeError("Unexpected class count")
    # Atomic exclusive marker *before* any held-out test images can be decoded.
    with (output/"test_recovery_started.json").open("x") as file:
        json.dump({"reason":"OS errno 24 before any test batch was decoded",
                   "original_test_marker":marker,
                   "frozen_probe_reused":True,
                   "encoder_epoch":200,
                   "dataset":dataset,
                   "source_checkpoint":str(checkpoint)},file)
    x,y,keys=_extract(enc,by["test"],c,device,ds)
    if x.ndim!=2 or x.shape[1]!=state["feature_mean"].shape[1]:
        raise RuntimeError("Feature shapes incompatible with previously selected probe")
    clf=torch.nn.Linear(x.shape[1],2).to(device).eval()
    clf.load_state_dict(state["classifier"],strict=True)
    loss,metrics,preds=_evaluate(clf,(x-state["feature_mean"])/state["feature_std"],y,device)
    np.savez_compressed(output/"test_predictions.npz",labels=y.numpy(),predictions=preds,keys=keys)
    labels=[0,1]
    report=classification_report(y.numpy(),preds,labels=labels,target_names=classes,
                                output_dict=True,zero_division=0)
    write_csv([{"class_id":i,"class_name":classes[i],**report[classes[i]]}
               for i in labels],output/"test_per_class_metrics.csv")
    np.savetxt(output/"test_confusion_matrix.csv",
               confusion_matrix(y.numpy(),preds,labels=labels),delimiter=",",fmt="%d")
    lookup={int(row["record_index"]):row for row in by["test"]}
    def sample_ref(row):
        if row.get("relative_path"):return str(row["relative_path"])
        if row.get("storage")=="parquet":
            return f"{row.get('parquet_file','')}#row_group={row.get('row_group','')}#row={row.get('row_in_group','')}"
        if row.get("storage")=="hdf5":
            return f"{row.get('hdf5_file','')}#index={row.get('hdf5_index','')}"
        return str(row.get("source_name",row.get("record_index","")))
    write_csv([{"record_index":int(k),"sample_ref":sample_ref(lookup[int(k)]),
                "relative_path":lookup[int(k)].get("relative_path",""),
                "class_id":int(label),"class_name":classes[int(label)],
                "prediction_id":int(pred),"prediction_name":classes[int(pred)]}
               for k,label,pred in zip(keys,y.numpy(),preds)],
              output/"test_prediction_records.csv")
    result={"method":"dinov3","dataset":dataset,"encoder_epoch":200,
            "feature_dim":int(x.shape[1]),"num_classes":2,"class_names":classes,
            "probe_validation_macro_f1":float(sel["validation_macro_f1"]),
            "test_loss":float(loss),"test":metrics,"test_images":int(y.numel()),
            "test_evaluated_once":True,"dataset_metadata":ds.metadata()}
    atomic_json_dump(result,output/"test_metrics.json")
    print(json.dumps({"status":"RECOVERY_COMPLETE","dataset":dataset,"epoch":200,
                      "test_images":int(y.numel()),"frozen_probe_reused":True,
                      "test_accuracy":metrics.get("accuracy")}),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--manifest-root",type=Path,required=True)
    p.add_argument("--model-root",type=Path,required=True)
    x=p.parse_args()
    recover(x.checkpoint,x.output,x.manifest_root,x.model_root)

if __name__=="__main__":
    main()
