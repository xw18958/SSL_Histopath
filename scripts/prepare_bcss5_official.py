#!/usr/bin/env python3
"""Prepare BCSS official five-group histology segmentation, institutional 76/30/45 split.

Source 151 full annotated ROIs (0.25um/px) -> 0.5um/px tiles.
Never overwrite existing source masks, the old binary task, or previous results.
"""
from __future__ import annotations
import argparse,hashlib,io,json,math
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from PIL import Image
import pyarrow.parquet as pq

TEST={"OL","LL","E2","EW","GM","S3"}
VAL={"BH","C8","A8","A1","E9"}
IGNORE={0,7,15}
CLASSES=("tumor","stroma","inflammatory_infiltration","necrosis","other")
# Original BCSS authors' 5 region groups; refer to bcsegmentation.grand-challenge.org/Baseline/
GROUPS={
  1:0,19:0,20:0, # invasive/in-situ/vascular carcinoma
  2:1,
  3:2,10:2,11:2,
  4:3,
  5:4,6:4,8:4,9:4,12:4,13:4,14:4,16:4,17:4,18:4,21:4
}
if set(GROUPS)!=set(range(1,22))-IGNORE:raise AssertionError("BCSS ontology incomplete")
LUT=np.full(256,255,dtype=np.uint8)
for old,new in GROUPS.items():LUT[old]=new

def split(institute):
    return "test" if institute in TEST else "val" if institute in VAL else "train"

def sha256_file(path):
    s=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):s.update(block)
    return s.hexdigest()

def build(data_root,out,manifest,tiles_axis=4):
    data_root=data_root.resolve();out=out.resolve();out.mkdir(parents=True,exist_ok=True)
    manifest=manifest.resolve()
    if manifest.exists():raise FileExistsError("Refuse overwrite manifest: "+str(manifest))
    if (out/"COMPLETE").exists():raise FileExistsError("Output exists: "+str(out))
    shards=sorted((data_root/"BCSS/data").glob("*.parquet"))
    if len(shards)!=10:raise ValueError("Expected 10 BCSS source shards")
    pair_counts=Counter(); group_split={}; records=[]; per_split=defaultdict(Counter)
    imgdir=out/"images"; maskdir=out/"masks"
    imgdir.mkdir(exist_ok=True); maskdir.mkdir(exist_ok=True)
    for shard in shards:
        f=pq.ParquetFile(shard)
        for batch in f.iter_batches(batch_size=1,columns=["image","mask","patient_id","image_id"]):
            entry=batch.to_pylist()[0]
            patient=str(entry["patient_id"])
            institution=patient.split("-")[1]
            part=split(institution)
            pair_counts[part]+=1
            if patient in group_split and group_split[patient]!=part:
                raise ValueError("Patient leaked across split")
            group_split[patient]=part
            with Image.open(io.BytesIO(entry["image"]["bytes"])) as rgb, Image.open(io.BytesIO(entry["mask"]["bytes"])) as source_mask:
                rgb=rgb.convert("RGB"); source_mask=source_mask.convert("L")
                if rgb.size!=source_mask.size:raise ValueError("RGB/mask mismatch")
                # 0.5um/px is the evaluation resolution used by eva-probe;
                # both RGB & mask downsampled once with appropriate filters.
                w,h=rgb.size
                width=round(w/2);height=round(h/2)
                rgb=rgb.resize((width,height),Image.Resampling.BICUBIC)
                source_mask=source_mask.resize((width,height),Image.Resampling.NEAREST)
                for yi in range(tiles_axis):
                    for xi in range(tiles_axis):
                        # Uniform coverage; no test-mask-dependent sampling.
                        left=round(max(0,width-256)*xi/max(1,tiles_axis-1))
                        top=round(max(0,height-256)*yi/max(1,tiles_axis-1))
                        if width<256 or height<256:raise ValueError("ROI too small after 0.5mpp downsampling")
                        box=(left,top,left+256,top+256)
                        orig=np.asarray(source_mask.crop(box),dtype=np.uint8)
                        if np.any(orig>21):raise ValueError("Unknown BCSS labels")
                        mask=LUT[orig]
                        stem=f"roi{sum(pair_counts.values()):04d}_{yi:02d}{xi:02d}"
                        imfile=imgdir/(stem+".png");maskfile=maskdir/(stem+".png")
                        if imfile.exists() and maskfile.exists():
                            # Safe preparation resume: never rewrite data; verify every existing mask.
                            with Image.open(maskfile) as old:
                                if not np.array_equal(np.asarray(old),mask):
                                    raise ValueError("Existing generated mask differs: "+str(maskfile))
                        elif imfile.exists() or maskfile.exists():
                            raise FileExistsError("Partial image/mask pair; refuse overwrite: "+str(imfile))
                        else:
                            rgb.crop(box).save(imfile)
                            Image.fromarray(mask).save(maskfile)
                        per_split[part].update(dict(enumerate(np.bincount(mask.ravel(),minlength=256)[:5])))
                        records.append(dict(image=str(imfile.relative_to(data_root)),
                                            mask=str(maskfile.relative_to(data_root)),
                                            group=patient,institution=institution,
                                            roi=str(entry["image_id"]),tile=[left,top],
                                            split=part))
    if dict(pair_counts)!={"train":76,"val":30,"test":45}:
        raise ValueError("Official BCSS institution split expected 76/30/45 ROI: "+str(pair_counts))
    if len(records)!=151*tiles_axis**2:raise AssertionError("Tile count mismatch")
    for part in ("train","val","test"):
        if any(per_split[part][c]<1 for c in range(5)):
            raise ValueError("A BCSS canonical class missing in "+part+": "+str(per_split[part]))
    identity={p:set() for p in ("train","val","test")}
    for r in records:identity[r["split"]].add(r["group"])
    if any(identity[a]&identity[b] for a,b in (("train","val"),("train","test"),("val","test"))):
        raise AssertionError("patient leakage")
    cfg=dict(version="bcss5_official_institutions_v1",dataset="bcss5",source="Beijuka/BCSS (Amgad et al. 2019)",
             source_shards=[dict(file=p.name,sha256=sha256_file(p)) for p in shards],
             mask_labels=list(CLASSES),mapping={str(k):v for k,v in GROUPS.items()},
             ignored_original_codes=sorted(IGNORE),
             source_mpp=0.25,evaluation_mpp=0.5,tiles_axis=tiles_axis,selection="fixed 4x4 uniform grid per ROI, mask independent",
             official_test_institutions=sorted(TEST),validation_institutions=sorted(VAL),
             source_roi_count=dict(pair_counts),tile_count={k:sum(r["split"]==k for r in records) for k in ("train","val","test")},
             class_pixel_counts={k:{str(c):int(v) for c,v in per_split[k].items()} for k in ("train","val","test")},
             records=records)
    raw=json.dumps(cfg,sort_keys=True,separators=(",",":"))
    cfg["manifest_sha256"]=hashlib.sha256(raw.encode()).hexdigest()
    manifest.parent.mkdir(parents=True,exist_ok=True)
    # Write complete marker only after every image, mask & manifest validation passes.
    manifest.write_text(json.dumps(cfg,indent=2,sort_keys=True)+"\n")
    (out/"COMPLETE").write_text(cfg["manifest_sha256"]+"\n")
    print(json.dumps({k:v for k,v in cfg.items() if k not in ("records","source_shards","mapping")},indent=2),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    a=p.parse_args()
    build(a.data_root,a.out_dir,a.manifest)
