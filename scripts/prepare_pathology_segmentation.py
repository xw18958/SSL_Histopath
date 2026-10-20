#!/usr/bin/env python3
"""Build deterministic pathology segmentation manifests, without touching source datasets."""
from __future__ import annotations
import argparse, hashlib, io, json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

SEED=20260903
VERSION="pathology_semantic_seg_v1"

def digest(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def grouped_split(groups):
    order=sorted(set(groups),key=lambda g:digest(f"{SEED}|{g}"))
    if len(order)<3: raise ValueError("At least 3 independent groups needed")
    nt=max(1,round(len(order)*.7)); nv=max(1,round(len(order)*.15))
    if nt+nv>=len(order): raise ValueError("Not enough groups")
    return {g:("train" if i<nt else "val" if i<nt+nv else "test") for i,g in enumerate(order)}

def ebhi(data_root):
    source=data_root/"EBHI-SEG"
    records=[]; absent=[]; dup={}
    for category in sorted(source.iterdir()):
        if not (category/"image").is_dir(): continue
        labels={p.stem.casefold():p for p in (category/"label").iterdir() if p.suffix.lower() in (".png",".jpg",".jpeg")}
        for p in sorted((category/"image").iterdir()):
            if p.suffix.lower() not in (".png",".jpg",".jpeg"): continue
            m=labels.get(p.stem.casefold())
            if m is None:
                absent.append(str(p.relative_to(data_root)));continue
            key=hashlib.sha256(p.read_bytes()).hexdigest()
            if key in dup:
                # Identical images are excluded globally to avoid split contamination.
                absent.append(str(p.relative_to(data_root))+" [duplicate image of "+dup[key]+"]")
                continue
            dup[key]=str(p.relative_to(data_root))
            group=category.name.casefold()+"/"+p.stem.rsplit("-",1)[0].casefold()
            records.append(dict(image=str(p.relative_to(data_root)),mask=str(m.relative_to(data_root)),group=group,category=category.name))
    if len(records)<1000: raise ValueError("Insufficient EBHI pairs")
    spl=grouped_split([r["group"] for r in records])
    for r in records:r["split"]=spl[r["group"]]
    return records,dict(mask_semantics="uint8 grayscale >=128 foreground; nearest neighbour mask resize",
                        unmatched_or_duplicate=absent,
                        group_policy="category and filename stem without final patch index; specimen-proxy only, patient identities unverified")

def bcss(data_root,tiles_per_roi=6):
    import pyarrow.parquet as pq
    source=data_root/"BCSS"
    shards=sorted((source/"data").glob("*.parquet"))
    if len(shards)!=10: raise ValueError("Expected 10 BCSS shards")
    target=data_root/"segmentation_benchmark_v1"/"bcss_binary"
    (target/"images").mkdir(parents=True,exist_ok=True)
    (target/"masks").mkdir(parents=True,exist_ok=True)
    records=[]; patients=[]
    for shard in shards:
        table=pq.read_table(shard,columns=["image","mask","patient_id","image_id"])
        for entry in table.to_pylist():
            patient=str(entry["patient_id"])
            if not patient: raise ValueError("Missing patient ID")
            patients.append(patient)
            with Image.open(io.BytesIO(entry["image"]["bytes"])) as im:
                im=im.convert("RGB"); w,h=im.size
                with Image.open(io.BytesIO(entry["mask"]["bytes"])) as ma:
                    ma=ma.convert("L")
                    if ma.size!=im.size:raise ValueError("ROI image/mask dimension mismatch")
                    # Reproducible spatially spread tiles, no mask-guided selection.
                    grid=[(.15,.20),(.5,.20),(.85,.20),(.15,.75),(.5,.75),(.85,.75)]
                    for ti,(rx,ry) in enumerate(grid[:tiles_per_roi]):
                        x=max(0,min(w-256,round((w-256)*rx)))
                        y=max(0,min(h-256,round((h-256)*ry)))
                        roi=np.asarray(ma.crop((x,y,x+256,y+256)))
                        mask=np.full((256,256),255,dtype=np.uint8)
                        valid=(roi!=0)&(roi!=7)&(roi!=15)
                        mask[valid]=0;mask[roi==1]=1
                        if np.count_nonzero(valid)<256*256*.25:continue
                        stem=f"roi{len(patients):04d}_tile{ti:02d}"
                        destimg=target/"images"/f"{stem}.png";destmask=target/"masks"/f"{stem}.png"
                        if destimg.exists() or destmask.exists():raise FileExistsError(destimg)
                        im.crop((x,y,x+256,y+256)).save(destimg)
                        Image.fromarray(mask).save(destmask)
                        records.append(dict(image=str(destimg.relative_to(data_root)),
                                            mask=str(destmask.relative_to(data_root)),
                                            group=patient,roi=str(entry["image_id"]),tile=[x,y]))
    if len(set(patients))<12 or len(records)<200: raise ValueError("Insufficient BCSS patients/tiles")
    spl=grouped_split(patients)
    for r in records:r["split"]=spl[r["group"]]
    return records,dict(mask_semantics="original BCSS 1=tumor, valid labels 2..21=non-tumor; 0/7/15 ignored; 255 ignore",
                        split_policy="patient-disjoint; 6 deterministic tiles per ROI; no mask-dependent selection",
                        tiles_per_roi=tiles_per_roi,patients=len(set(patients)),source_roi_count=len(patients))

def glas(data_root):
    source=data_root/"GlaS"
    candidates=[p for p in source.rglob("*.bmp") if not p.stem.lower().endswith(("_anno","_anoo"))]
    records=[]
    for p in sorted(candidates):
        prefix=p.stem.lower()
        if not prefix.startswith(("train_","testa_","testb_")):continue
        m=next((q for q in [p.with_name(p.stem+"_anno.bmp"),p.with_name(p.stem+"_anoo.bmp")]
               if q.exists()),None)
        if m is None:raise FileNotFoundError("Missing official GlaS annotation: "+str(p))
        group=prefix
        records.append(dict(image=str(p.relative_to(data_root)),mask=str(m.relative_to(data_root)),
                            group=group,category=prefix.split("_")[0]))
    if len(records)!=165:raise ValueError(f"Expected 165 GlaS pairs, found {len(records)}")
    train=sorted([r for r in records if r["category"]=="train"],key=lambda r:digest(f"{SEED}|{r['group']}"))
    if len(train)!=85: raise AssertionError("GlaS expected 85 training images")
    for i,r in enumerate(train):r["split"]="val" if i<15 else "train"
    for r in records:
        if r["category"].startswith("test"):r["split"]="test"
    return records,dict(mask_semantics="instance ID >0 is gland foreground; label 0 background",
                        split_policy="official 85 train, 60 testA and 20 testB; 15 of 85 train reserved as validation; source patient ids unavailable")

def validate(records, data_root):
    if not records:raise ValueError("empty manifest")
    groups={}; seen=set(); counts=Counter()
    for r in records:
        for key in ("image","mask"):
            p=(data_root/r[key]).resolve()
            if not p.is_relative_to(data_root.resolve()) or not p.is_file():
                raise ValueError("Unsafe/missing path: "+str(p))
        if r["image"] in seen:raise ValueError("duplicate image record")
        seen.add(r["image"]);counts[r["split"]]+=1
        if r["group"] in groups and groups[r["group"]]!=r["split"]:
            raise ValueError("group crosses train/validation/test splits")
        groups[r["group"]]=r["split"]
    if set(counts)!={"train","val","test"}:raise ValueError("missing split")
    return dict(counts=dict(counts),groups=len(groups),images=len(records))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--dataset",choices=["ebhi","bcss","glas"],required=True)
    ap.add_argument("--tiles-per-roi",type=int,default=6,choices=[3,6])
    a=ap.parse_args()
    data_root=a.data_root.resolve()
    file=a.out_dir/f"{a.dataset}.json"
    if file.exists():raise FileExistsError("Immutable manifest exists: "+str(file))
    maker={"ebhi":lambda:ebhi(data_root),"bcss":lambda:bcss(data_root,a.tiles_per_roi),"glas":lambda:glas(data_root)}
    rec,audit=maker[a.dataset]()
    diag=validate(rec,data_root)
    payload=dict(version=VERSION,dataset=a.dataset,seed=SEED,records=rec,audit=audit,summary=diag)
    payload["manifest_sha256"]=digest(json.dumps(payload,sort_keys=True,separators=(",",":")))
    a.out_dir.mkdir(parents=True,exist_ok=True)
    file.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({**diag,"dataset":a.dataset,"sha256":payload["manifest_sha256"],"audit":audit},indent=2)[:5000],flush=True)
if __name__=="__main__":main()
