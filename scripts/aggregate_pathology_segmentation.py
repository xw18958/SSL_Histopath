#!/usr/bin/env python3
"""Independently verify and aggregate 12 frozen linear pathology segmentation jobs."""
from __future__ import annotations
import argparse,csv,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np

DATASETS=("glas","ebhi","bcss")
METHODS=("lejepa","simplex_k8")
EPOCHS=(100,200)

def recompute(c):
    x=np.asarray(c,dtype=np.int64)
    if x.shape!=(2,2) or (x<0).any():raise ValueError("Invalid confusion matrix")
    t=np.diag(x).astype(float)
    u=x.sum(0)+x.sum(1)-t
    if (u<=0).any():raise ValueError("Absent foreground/background")
    iou=t/u
    d=2*t/(x.sum(0)+x.sum(1))
    return dict(miou=float(iou.mean()),foreground_iou=float(iou[1]),
                foreground_dice=float(d[1]),valid_pixels=int(x.sum()))

def write_csv(path,records):
    if not records:raise ValueError("Empty output")
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]))
        w.writeheader();w.writerows(records)

def verify(raw_dir,out_dir):
    docs={}
    for f in sorted(raw_dir.rglob("result.json")):
        d=json.loads(f.read_text())
        key=(d["dataset"],d["method"],int(d["epoch"]))
        if key in docs:raise ValueError(f"Duplicate job result {key}")
        if d.get("smoke") is not False or d.get("test_used_for_tuning") is not False:
            raise ValueError("Smoke/test leakage in result")
        for keyname in ("manifest_sha256","ckpt_sha256"):
            value=d.get(keyname)
            if not isinstance(value,str) or len(value)!=64:raise ValueError(f"Missing {keyname}")
        if d.get("representation")!="frozen 64 final patch tokens post-layernorm, pre-projector":
            raise ValueError("Wrong representation")
        s=d["test"]
        for name,v in recompute(s["confusion_matrix"]).items():
            if isinstance(v,int):
                if s[name]!=v:raise ValueError(f"Wrong {name}")
            elif abs(s[name]-v)>1e-12:raise ValueError(f"Wrong score {key} {name}")
        if d["dataset"]=="glas":
            cm=np.asarray(s["testa"]["confusion_matrix"])+np.asarray(s["testb"]["confusion_matrix"])
            if not np.array_equal(cm,np.asarray(s["confusion_matrix"])):
                raise ValueError("GlaS test A and B do not reconcile")
            if s["testa"]["valid_pixels"]!=60*256*256 or s["testb"]["valid_pixels"]!=20*256*256:
                raise ValueError("Wrong GlaS official split")
        docs[key]=d
    expected={(dataset,method,epoch) for dataset in DATASETS for method in METHODS for epoch in EPOCHS}
    if set(docs)!=expected:
        raise ValueError(f"Missing {sorted(expected-set(docs))}; unexpected {sorted(set(docs)-expected)}")
    by_manifest=defaultdict(set);by_samples=defaultdict(set);by_ckpt=defaultdict(set)
    for (dataset,method,epoch),d in docs.items():
        by_manifest[dataset].add(d["manifest_sha256"])
        by_samples[dataset].add((d["test_images"],d["test"]["valid_pixels"]))
        by_ckpt[(method,epoch)].add(d["ckpt_sha256"])
    if any(len(x)!=1 for x in list(by_manifest.values())+list(by_samples.values())+list(by_ckpt.values())):
        raise ValueError("Dataset/checkpoint provenance mismatch")
    if len({next(iter(x)) for x in by_ckpt.values()})!=4:
        raise ValueError("Same checkpoint SHA across distinct methods/epochs")
    rows=[]
    for dataset in DATASETS:
        for epoch in EPOCHS:
            for method in METHODS:
                d=docs[(dataset,method,epoch)]
                row=dict(dataset=dataset,epoch=epoch,method=method,test_images=d["test_images"],
                         valid_pixels=d["test"]["valid_pixels"],
                         miou=round(100*d["test"]["miou"],4),
                         foreground_iou=round(100*d["test"]["foreground_iou"],4),
                         foreground_dice=round(100*d["test"]["foreground_dice"],4),
                         val_miou=round(100*d["val_miou"],4),
                         lr=d["lr"],weight_decay=d["weight_decay"],best_epoch=d["best_epoch"],
                         checkpoint_sha256=d["ckpt_sha256"],manifest_sha256=d["manifest_sha256"])
                rows.append(row)
    pairs=[]
    for dataset in DATASETS:
        for epoch in EPOCHS:
            b=docs[(dataset,"lejepa",epoch)]["test"];s=docs[(dataset,"simplex_k8",epoch)]["test"]
            pairs.append(dict(dataset=dataset,epoch=epoch,
                              lejepa_miou=round(100*b["miou"],4),
                              simplex_k8_miou=round(100*s["miou"],4),
                              simplex_minus_lejepa_pp=round(100*(s["miou"]-b["miou"]),4),
                              lejepa_foreground_dice=round(100*b["foreground_dice"],4),
                              simplex_k8_foreground_dice=round(100*s["foreground_dice"],4)))
    out_dir.mkdir(exist_ok=True,parents=True)
    write_csv(out_dir/"per_dataset.csv",rows)
    write_csv(out_dir/"paired_comparisons.csv",pairs)
    report=dict(status="VERIFIED",evaluation_count=len(rows),
                datasets=list(DATASETS),methods=list(METHODS),epochs=list(EPOCHS),
                protocol="frozen pre-projector 8x8 spatial-token, trainable 1x1 linear semantic segmentation probe",
                metric="pixel-level micro-aggregated mIoU (2 classes), percentages",
                manifest_sha256={k:next(iter(v)) for k,v in by_manifest.items()},
                checkpoint_sha256={f"{m}@{e}":next(iter(v)) for (m,e),v in by_ckpt.items()},
                test_images_and_valid_pixels={k:list(next(iter(v))) for k,v in by_samples.items()},
                exclusions="EBHI-SEG two unpaired source images, BCSS predominantly ignored tiles",
                limitation="GlaS/EBHI no guaranteed patient-disjoint source split; GlaS semantic not official instance metrics; BCSS binary sampled patches",
                mIoU_delta_dataset_mean_pp_by_epoch={str(epoch):round(np.mean([
                    100*(docs[(data,"simplex_k8",epoch)]["test"]["miou"]-
                         docs[(data,"lejepa",epoch)]["test"]["miou"]) for data in DATASETS]),4) for epoch in EPOCHS})
    (out_dir/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return rows,pairs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--raw-dir",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args();verify(a.raw_dir,a.out_dir)
if __name__=="__main__":main()
