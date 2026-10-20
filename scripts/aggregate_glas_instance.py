#!/usr/bin/env python3
"""Audit per-image GlaS TestA/TestB object metrics and 4 checkpoint identities."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
CLASSES=("testa","testb")
METHODS=("lejepa","simplex_k8")
EPOCHS=(100,200)
METRICS=("object_f1","object_dice","object_hausdorff")

def rows_to_csv(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def validate(raw,out):
    docs={}
    for path in sorted(raw.rglob("result.json")):
        data=json.loads(path.read_text())
        key=(data["method"],data["epoch"])
        if key in docs:raise ValueError("Duplicate checkpoint result "+str(key))
        if data["dataset"]!="glas_instance_v1" or data["smoke"] or data["test_used_for_tuning"]:
            raise ValueError("Wrong experiment or test tuning")
        if data["source_splits"]!={"train":70,"val":15,"test":80}:
            raise ValueError("Nonstandard held-out dataset")
        if len(data["source_manifest_sha256"])!=64 or len(data["ssl_checkpoint_sha256"])!=64:
            raise ValueError("Missing provenance SHA")
        if data.get("representation")!="frozen ViT-B/32 8x8x768 last patch tokens, pre-projector":
            raise ValueError("Wrong frozen representation")
        per=data["per_image"]
        if len(per)!=80 or len({x["image"] for x in per})!=80:
            raise ValueError("Missing / duplicate test image")
        for name,count,label in (("testA",60,"testa"),("testB",20,"testb")):
            subset=[x for x in per if x["category"]==label]
            if len(subset)!=count or data[name]["images"]!=count:
                raise ValueError("GlaS test-set mismatch "+name)
            for metric in METRICS:
                values=np.asarray([x[metric] for x in subset],dtype=np.float64)
                if not np.isfinite(values).all():raise ValueError("Nonfinite "+metric)
                if metric!="object_hausdorff" and not ((values>=0)&(values<=1)).all():
                    raise ValueError("Out-of-range "+metric)
                if abs(float(np.mean(values))-data[name][metric])>1e-12:
                    raise ValueError("Mean mismatches "+name+" "+metric)
        for metric in METRICS:
            m=float(np.mean([p[metric] for p in per]))
            if abs(data["combined"][metric]-m)>1e-12:raise ValueError("Incorrect pooled test "+metric)
        docs[key]=data
    if set(docs)!={(m,e) for m in METHODS for e in EPOCHS}:
        raise ValueError("Missing checkpoint runs: "+str(set(docs)))
    if len({d["source_manifest_sha256"] for d in docs.values()})!=1:
        raise ValueError("Dataset split differs")
    if len({d["ssl_checkpoint_sha256"] for d in docs.values()})!=4:
        raise ValueError("Four SSL checkpoints are not distinct")
    by_method={}
    for (m,e),d in docs.items():
        source_identity=tuple(sorted(x["image"] for x in d["per_image"]))
        by_method[(m,e)]=source_identity
    if len(set(by_method.values()))!=1:raise ValueError("Paired compared test samples differ")
    rows=[];per_img=[]
    for name in ("testA","testB"):
        for ep in EPOCHS:
            for method in METHODS:
                d=docs[(method,ep)]
                r=d[name]
                rows.append(dict(test=name,epoch=ep,method=method,
                                 object_F1=round(100*r["object_f1"],4),
                                 object_Dice=round(100*r["object_dice"],4),
                                 object_Hausdorff_px=round(r["object_hausdorff"],4),
                                 n_images=r["images"],checkpoint_sha256=d["ssl_checkpoint_sha256"]))
                for z in d["per_image"]:
                    if z["category"]==name.lower():
                        per_img.append(dict(epoch=ep,method=method,image=z["image"],test=name,
                                            gt_glands=z["gt_glands"],pred_glands=z["predicted_glands"],
                                            object_F1=z["object_f1"],object_Dice=z["object_dice"],
                                            object_Hausdorff_px=z["object_hausdorff"]))
    paired=[]
    for name in ("testA","testB"):
        for ep in EPOCHS:
            a=docs[("lejepa",ep)][name];b=docs[("simplex_k8",ep)][name]
            paired.append(dict(test=name,epoch=ep,
               F1_LeJEPA=round(100*a["object_f1"],3),F1_Simplex=round(100*b["object_f1"],3),
               Simplex_minus_LeJEPA_F1_pp=round(100*(b["object_f1"]-a["object_f1"]),3),
               Dice_LeJEPA=round(100*a["object_dice"],3),Dice_Simplex=round(100*b["object_dice"],3),
               Simplex_minus_LeJEPA_Dice_pp=round(100*(b["object_dice"]-a["object_dice"]),3),
               HD_LeJEPA=a["object_hausdorff"],HD_Simplex=b["object_hausdorff"]))
    out.mkdir(parents=True,exist_ok=True)
    rows_to_csv(out/"summary.csv",rows)
    rows_to_csv(out/"paired.csv",paired)
    rows_to_csv(out/"per_image.csv",per_img)
    verification=dict(status="VERIFIED",runs=4,source_images=165,
                      source_split=dict(train=70,val=15,testA=60,testB=20),
                      source_manifest_sha256=next(iter({d["source_manifest_sha256"] for d in docs.values()})),
                      checkpoint_sha256={f"{m}@{e}":d["ssl_checkpoint_sha256"] for (m,e),d in docs.items()},
                      protocol="full-resolution 256-patch overlap, frozen 8x8x768, 2-channel gland+distance linear head, watershed instance map",
                      published_metric_family="GlaS object detection F1, symmetric area-weighted object Dice, symmetric area-weighted object Hausdorff",
                      qualifier="Python challenge-style metric port with explicit one-to-one matching and empty-prediction penalty; not official Matlab evaluator",
                      test_independence="head and postprocessing selected using train/validation, test read only after selection")
    (out/"verification.json").write_text(json.dumps(verification,indent=2)+"\n")
    print(json.dumps({"status":"VERIFIED","paired":paired,"provenance":verification},indent=2))
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--raw-dir",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    a=ap.parse_args();validate(a.raw_dir,a.out_dir)
