#!/usr/bin/env python3
"""Independently audit 4 official-institution BCSS five-class frozen probe results."""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

METHODS=("lejepa","simplex_k8")
EPOCHS=(100,200)
CLASSES=("tumor","stroma","inflammatory_infiltration","necrosis","other")

def metrics(cm):
    cm=np.asarray(cm,dtype=np.int64)
    if cm.shape!=(5,5) or (cm<0).any():raise AssertionError("Expected nonnegative 5x5 matrix")
    gt=cm.sum(1);tp=np.diag(cm)
    if (gt==0).any():raise ValueError("Official benchmark missing a class in test GT")
    union=gt+cm.sum(0)-tp
    if (union<=0).any():raise ValueError("No union for a class")
    iou=tp/union
    dice=2*tp/(gt+cm.sum(0))
    return dict(miou=float(np.mean(iou)),mean_dice=float(np.mean(dice)),
                per_class_iou=iou.tolist(),per_class_dice=dice.tolist(),
                valid_pixels=int(cm.sum()),class_gt_pixels=gt.tolist(),
                class_pred_pixels=cm.sum(0).tolist())

def writer(path,items):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)

def aggregate(raw_dir,out):
    results={}
    for f in sorted(raw_dir.rglob("result.json")):
        r=json.loads(f.read_text())
        key=(r["method"],r["epoch"])
        if key in results:raise ValueError("Duplicate pair "+str(key))
        if r["dataset"]!="bcss5" or r.get("smoke") is not False:raise ValueError("Wrong dataset/smoke")
        if r.get("test_used_for_tuning") is not False:raise ValueError("Test leakage flag")
        for tag in ("ckpt_sha256","manifest_sha256"):
            if len(r.get(tag,""))!=64:raise ValueError("Missing hash "+tag)
        if r.get("representation")!="frozen 64 final patch tokens post-layernorm, pre-projector":
            raise ValueError("Unexpected representation")
        if r["train_images"]!=1216 or r["val_images"]!=480 or r["test_images"]!=720:
            raise ValueError("Incorrect official-institution 16-grid patch split")
        m=metrics(r["test"]["confusion_matrix"])
        for name,value in m.items():
            actual=r["test"][name]
            if isinstance(value,list):
                if not np.allclose(actual,value,atol=1e-12,rtol=0):
                    raise ValueError("Recomputed "+name+" mismatch")
            elif abs(actual-value)>1e-12:raise ValueError("Recomputed "+name+" mismatch")
        if r["test"]["evaluated_classes"]!=list(range(5)):raise ValueError("Five class evaluation incomplete")
        weight=np.array(r["train_only_class_weights"])
        if len(weight)!=5 or (weight<=0).any() or not np.isclose(weight.mean(),1):
            raise ValueError("Invalid training-only class weights")
        results[key]=r
    if set(results)!={(m,e) for m in METHODS for e in EPOCHS}:
        raise ValueError("Incomplete four-way comparison: "+str(sorted(results)))
    if len({r["manifest_sha256"] for r in results.values()})!=1:
        raise ValueError("Different test manifest across compared encoders")
    if len({tuple(r["test"]["class_gt_pixels"]) for r in results.values()})!=1:
        raise ValueError("GT pixel support varies between methods")
    reference_weights=np.asarray(results[("lejepa",100)]["train_only_class_weights"])
    if any(not np.allclose(r["train_only_class_weights"],reference_weights,rtol=0,atol=1e-12)
           for r in results.values()):
        raise ValueError("Training-weight protocol differs")
    if len({r["ckpt_sha256"] for r in results.values()})!=4:
        raise ValueError("Checkpoints not distinct")
    rows=[];class_rows=[]
    for e in EPOCHS:
        base=results[("lejepa",e)]["test"];simp=results[("simplex_k8",e)]["test"]
        rows.append(dict(epoch=e,LeJEPA_mIoU=round(base["miou"]*100,4),
                         SimplexK8_mIoU=round(simp["miou"]*100,4),
                         delta_mIoU_pp=round((simp["miou"]-base["miou"])*100,4),
                         LeJEPA_mDice=round(base["mean_dice"]*100,4),
                         SimplexK8_mDice=round(simp["mean_dice"]*100,4)))
        for i,name in enumerate(CLASSES):
            class_rows.append(dict(epoch=e,category=name,GT_pixels=base["class_gt_pixels"][i],
                LeJEPA_IoU=round(base["per_class_iou"][i]*100,4),
                SimplexK8_IoU=round(simp["per_class_iou"][i]*100,4),
                delta_IoU_pp=round((simp["per_class_iou"][i]-base["per_class_iou"][i])*100,4)))
    out.mkdir(exist_ok=True,parents=True)
    writer(out/"summary.csv",rows);writer(out/"per_class.csv",class_rows)
    doc=dict(status="VERIFIED",runs=4,classes=CLASSES,
             source_roi_counts={"train":76,"val":30,"test":45},
             sampled_tile_counts={"train":1216,"val":480,"test":720},
             official_test_institutions=["E2","EW","GM","LL","OL","S3"],
             full_source_mpp=0.25,eval_mpp=0.5,tile_selection="fixed uniform 4x4 grid/ROI",
             manifest_sha256=next(iter({r["manifest_sha256"] for r in results.values()})),
             gt_pixel_counts=results[("lejepa",100)]["test"]["class_gt_pixels"],
             checkpoint_sha256={f"{method}@{epoch}":results[(method,epoch)]["ckpt_sha256"]
                                for method in METHODS for epoch in EPOCHS},
             class_weights_train_only=results[("lejepa",100)]["train_only_class_weights"],
             primary="5-class macro mIoU from globally pooled valid test pixels",
             secondary="5-class macro Dice and per-class IoUs",
             limitations="Custom mask-independent 4x4 patch sampling at 0.5mpp; ViT-B/32 8x8 token grid; one SSL pretraining run per setting")
    (out/"verification.json").write_text(json.dumps(doc,indent=2)+"\n")
    print(json.dumps(dict(summary=rows,per_class=class_rows,verification=doc),indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--raw-dir",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args();aggregate(a.raw_dir,a.out_dir)
