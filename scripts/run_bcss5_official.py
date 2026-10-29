#!/usr/bin/env python3
"""Frozen-encoder OFFICIAL GROUPED BCSS FIVE-CLASS semantic segmentation; immutable checkpoints.

Three histopathology datasets; split/feature/hash audits; test isolation.
This is a *semantic* segmentation probe, not an instance segmentation benchmark.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pannuke_ssl.ssl_methods.registry import build_method
from pannuke_ssl.ssl_framework.trainer import load_checkpoint

SEED=20260903
IGNORE=255
EPOCHS_TUNE=5
LR=(0.001,0.01)
WD=(0.,0.0001)
BATCH=32
FINAL_EPOCHS=25
PATIENCE=5

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def manifest_load(path):
    d=json.loads(path.read_text())
    h=d.pop("manifest_sha256")
    raw=json.dumps(d,sort_keys=True,separators=(",",":")).encode()
    if hashlib.sha256(raw).hexdigest()!=h:raise ValueError("Manifest checksum mismatch")
    d["manifest_sha256"]=h
    if d.get("version")!="bcss5_official_institutions_v1":raise ValueError("Unknown manifest version")
    groups={};rows=set();splits=set()
    for r in d["records"]:
        if r["image"] in rows:raise ValueError("duplicate image")
        rows.add(r["image"]);splits.add(r["split"])
        if r["group"] in groups and groups[r["group"]]!=r["split"]:raise ValueError("split leakage")
        groups[r["group"]]=r["split"]
    if splits!={"train","val","test"}:raise ValueError("Missing split")
    return d

def resize_crop(pil,resample):
    w,h=pil.size;factor=256/min(w,h)
    nw=max(256,round(w*factor));nh=max(256,round(h*factor))
    pil=pil.resize((nw,nh),resample)
    left=(nw-256)//2;top=(nh-256)//2
    return pil.crop((left,top,left+256,top+256))

def prep_pair(im,ma,dataset):
    rgb=np.asarray(resize_crop(im.convert("RGB"),Image.Resampling.BICUBIC),dtype=np.uint8).copy()
    if dataset=="ebhi":
        raw=np.asarray(resize_crop(ma.convert("L"),Image.Resampling.NEAREST))
        lab=(raw>=128).astype(np.uint8)
    elif dataset=="glas":
        raise ValueError("GlaS is not part of the five-class BCSS benchmark")
    elif dataset=="bcss5":
        lab=np.asarray(resize_crop(ma.convert("L"),Image.Resampling.NEAREST),dtype=np.uint8).copy()
        if not set(np.unique(lab)).issubset({0,1,2,3,4,IGNORE}):
            raise ValueError("BCSS mask has non-protocol multiclass labels")
    else:
        raise ValueError("Unexpected dataset: "+str(dataset))
    if rgb.shape!=(256,256,3) or lab.shape!=(256,256):raise AssertionError("Bad spatial alignment")
    return torch.from_numpy(rgb).permute(2,0,1),torch.from_numpy(lab.copy())

class PairDataset(Dataset):
    def __init__(self,rows,root,dataset):
        self.rows=rows;self.root=root;self.dataset=dataset
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]
        with Image.open(self.root/row["image"]) as image, Image.open(self.root/row["mask"]) as mask:
            rgb,y=prep_pair(image,mask,self.dataset)
        return rgb,y

def score(pred,truth,ncls=5):
    pred=np.asarray(pred).reshape(-1)
    truth=np.asarray(truth).reshape(-1)
    keep=truth!=IGNORE;pred=pred[keep];truth=truth[keep]
    if not keep.any():return np.zeros((ncls,ncls),dtype=np.int64)
    if not ((truth>=0)&(truth<ncls)&(pred>=0)&(pred<ncls)).all():
        raise ValueError("Unrecognized segmentation label")
    return np.bincount(ncls*truth+pred,minlength=ncls*ncls).reshape(ncls,ncls)

def summary(cm):
    cm=np.asarray(cm,dtype=np.int64)
    assert cm.shape==(5,5)
    tp=np.diag(cm).astype(float)
    support=cm.sum(1).astype(float)
    union=cm.sum(0).astype(float)+support-tp
    observed=support>0
    if not observed.any(): raise ValueError("No labeled pixels")
    # Undefined IoU of a class absent from the ground truth is never recorded as 1.
    iou=np.divide(tp,union,out=np.full(5,np.nan),where=union>0)
    dice=np.divide(2*tp,cm.sum(0)+support,out=np.full(5,np.nan),
                   where=(cm.sum(0)+support)>0)
    # Standard documented mIoU averages only classes with ground-truth support.
    miou=float(np.mean(iou[observed]))
    mdice=float(np.mean(dice[observed]))
    if not np.isfinite(miou) or not np.isfinite(mdice):raise FloatingPointError("Nonfinite metrics")
    return dict(miou=miou,mean_dice=mdice,
                per_class_iou=[None if not np.isfinite(z) else float(z) for z in iou],
                per_class_dice=[None if not np.isfinite(z) else float(z) for z in dice],
                class_gt_pixels=support.astype(int).tolist(),
                class_pred_pixels=cm.sum(0).astype(int).tolist(),
                evaluated_classes=np.flatnonzero(observed).tolist(),
                confusion_matrix=cm.tolist(),valid_pixels=int(cm.sum()))

@torch.inference_mode()
def features(encoder,rows,root,dataset,device,batch=48):
    ds=PairDataset(rows,root,dataset)
    dl=DataLoader(ds,batch_size=batch,shuffle=False,num_workers=2,pin_memory=True)
    xs=[];ys=[]
    for images,mask in dl:
        x=images.to(device=device,dtype=torch.float32,non_blocking=True).div_(255.)
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            tokens=encoder(x)
        if tokens.ndim!=3 or tokens.shape[1:]!=(64,768):
            raise ValueError(f"Expected 8x8x768 spatial patch tokens, got {tuple(tokens.shape)}")
        xs.append(tokens.detach().to("cpu",dtype=torch.float16))
        ys.append(mask)
    return torch.cat(xs),torch.cat(ys)

def logits(head,feat):
    z=head(feat.float().transpose(1,2).reshape(-1,768,8,8))
    return F.interpolate(z,size=(256,256),mode="bilinear",align_corners=False)

@torch.inference_mode()
def evaluate(head,cache,device,limit=None):
    xs,ys=cache;cm=np.zeros((5,5),dtype=np.int64)
    head.eval()
    for start in range(0,len(xs),BATCH):
        if limit is not None and start>=limit:break
        x=xs[start:start+BATCH].to(device,non_blocking=True)
        y=ys[start:start+BATCH]
        pred=logits(head,x).argmax(1).cpu().numpy()
        cm+=score(pred,y.numpy())
    return summary(cm)

def fit_once(train,val,lr,wd,max_epochs,patience,device,class_weights):
    torch.manual_seed(SEED)
    head=nn.Conv2d(768,5,kernel_size=1,bias=True).to(device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=wd)
    best=-1;best_epoch=0;state=None;history=[]
    x,y=train
    for epoch in range(1,max_epochs+1):
        head.train()
        generator=torch.Generator().manual_seed(SEED+epoch)
        indices=torch.randperm(len(x),generator=generator).tolist()
        losses=[]
        for start in range(0,len(indices),BATCH):
            ids=indices[start:start+BATCH]
            xi=x[ids].to(device,non_blocking=True)
            yi=y[ids].to(device,non_blocking=True).long()
            valid=(yi!=IGNORE).sum()
            if valid==0:continue
            optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(logits(head,xi),yi,weight=class_weights,ignore_index=IGNORE)
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite segmentation loss")
            loss.backward();optimizer.step()
            losses.append(float(loss.detach()))
        if not losses:raise RuntimeError("No labeled training pixels")
        metric=evaluate(head,val,device)
        history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),val_miou=metric["miou"]))
        if metric["miou"]>best+1e-9:
            best=metric["miou"];best_epoch=epoch
            state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        elif epoch-best_epoch>=patience:break
    if state is None:raise RuntimeError("No fitted head")
    return dict(val_miou=float(best),best_epoch=best_epoch,state=state,history=history)

def train_probe(train,val,device,class_weights):
    board=[]
    base_wd=WD[0];best_lr=None
    for lr in LR:
        trial=fit_once(train,val,lr,base_wd,EPOCHS_TUNE,EPOCHS_TUNE,device,class_weights)
        board.append(dict(stage="lr",lr=lr,wd=base_wd,miou=trial["val_miou"]))
        if best_lr is None or trial["val_miou"]>best_lr[1]:
            best_lr=(lr,trial["val_miou"])
    best_wd=None
    for wd in WD:
        trial=fit_once(train,val,best_lr[0],wd,EPOCHS_TUNE,EPOCHS_TUNE,device,class_weights)
        board.append(dict(stage="wd",lr=best_lr[0],wd=wd,miou=trial["val_miou"]))
        if best_wd is None or trial["val_miou"]>best_wd[1]:
            best_wd=(wd,trial["val_miou"])
    final=fit_once(train,val,best_lr[0],best_wd[0],FINAL_EPOCHS,PATIENCE,device,class_weights)
    return best_lr[0],best_wd[0],final,board

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--model-root",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--method",choices=["lejepa","simplex_k8","mae","dinov3","ijepa"],required=True)
    parser.add_argument("--epoch",type=int,choices=[100,200],required=True)
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("GPU required")
    torch.cuda.set_device(args.gpu)
    device=torch.device(f"cuda:{args.gpu}")
    torch.manual_seed(SEED);np.random.seed(SEED)
    torch.set_num_threads(4)
    src=manifest_load(args.manifest)
    if src["dataset"]!="bcss5" or src.get("mask_labels")!=[
        "tumor","stroma","inflammatory_infiltration","necrosis","other"]:
        raise ValueError("Only canonical five-group BCSS is supported")
    if src["source_roi_count"]!={"train":76,"val":30,"test":45}:
        raise ValueError("Official BCSS institutional source split not met")
    ckpt=args.checkpoint.resolve()
    chash=sha(ckpt)
    raw=torch.load(ckpt,map_location="cpu",weights_only=False)
    expected={"lejepa":"lejepa", "simplex_k8":"simplex_sigreg_lejepa", "mae":"mae", "dinov3":"dinov3", "ijepa":"ijepa"}[args.method]
    if raw.get("framework")!="ssl_standard_v1" or raw["method_name"]!=expected or raw["epoch"]!=args.epoch:
        raise ValueError("Wrong checkpoint method/framework/epoch")
    config=copy.deepcopy(raw["config"])
    if config["method"]["name"]!=expected:raise ValueError("Wrong checkpoint config")
    if args.method=="simplex_k8" and config["method"]["objective"]["simplex_components"]!=8:
        raise ValueError("Simplex checkpoint is not K=8")
    config["backbone"]["config_dir"]=str(args.model_root/"plip_model")
    method=build_method(config,device)
    _=load_checkpoint(method,ckpt,device)
    encoder=method.encoder.eval().to(device).requires_grad_(False)
    if encoder.training or any(p.requires_grad for p in encoder.parameters()):
        raise RuntimeError("Encoder is not frozen")
    del raw
    split={s:[r for r in src["records"] if r["split"]==s] for s in ("train","val","test")}
    if args.smoke:
        # Minimal independent train/val/test subsets, in smoke-only output tree.
        split={s:rows[:min(8,len(rows))] for s,rows in split.items()}
    out=args.out_dir
    out.mkdir(parents=True,exist_ok=True)
    if (out/"result.json").exists() or (out/"test_started.json").exists():
        raise FileExistsError("Refuse to repeat test/overwrite existing result: "+str(out))
    prefix=dict(dataset=src["dataset"],manifest_sha256=src["manifest_sha256"],
                ckpt_sha256=chash,method=args.method,epoch=args.epoch,smoke=args.smoke,
                representation="frozen 64 final patch tokens post-layernorm, pre-projector",
                preprocessing="RGB bicubic short-side-256 center-crop-256; mask nearest",
                head="1x1 linear conv 768->5 then bilinear upsampling; pixel CE",
                protocol="official 5 BCSS tissue groups, institute-held-out test; train/val sequential greedy LR then WD; test once")
    caches={}
    for s in ("train","val"):
        print("EXTRACT",args.method,args.epoch,src["dataset"],s,len(split[s]),flush=True)
        caches[s]=features(encoder,split[s],args.data_root,src["dataset"],device,
                          batch=8 if args.smoke else 48)
        if caches[s][0].shape[0]!=len(split[s]):raise AssertionError("Feature ordering")
    del method,encoder
    torch.cuda.empty_cache()
    print("TRAIN_HEAD",args.method,args.epoch,src["dataset"],flush=True)
    # Fixed training-only inverse-sqrt-frequency class balancing.
    # No validation/test labels or frequencies affect these weights.
    pixel_counts=np.array([src["class_pixel_counts"]["train"][str(i)]
                           for i in range(5)],dtype=np.float64)
    if (pixel_counts<=0).any():raise ValueError("Training pixels missing BCSS classes")
    weights=(pixel_counts/pixel_counts.sum())**(-.5)
    weights=weights/weights.mean()
    class_weights=torch.tensor(weights,dtype=torch.float32,device=device)
    lr,wd,fit,board=train_probe(caches["train"],caches["val"],device,class_weights)
    selection={**prefix,"train_only_class_weights":weights.tolist(),"lr":lr,"weight_decay":wd,"best_epoch":fit["best_epoch"],
               "val_miou":fit["val_miou"],"tuning":board,"history":fit["history"],
               "train_images":len(split["train"]),"val_images":len(split["val"]),
               "test_images":len(split["test"])}
    torch.save({"head":fit["state"],"selection":selection},out/"linear_head.pt")
    (out/"selection.json").write_text(json.dumps(selection,indent=2)+"\n")
    # Exclusive marker must precede reading any test image or pixel label.
    with (out/"test_started.json").open("x") as f:
        json.dump(prefix,f)
    # Reload frozen encoder ONLY after all hyperparameter and epoch choices are fixed.
    config["backbone"]["config_dir"]=str(args.model_root/"plip_model")
    method=build_method(config,device)
    _=load_checkpoint(method,ckpt,device)
    encoder=method.encoder.eval().to(device).requires_grad_(False)
    print("TEST_EXTRACT",args.method,args.epoch,src["dataset"],flush=True)
    test=features(encoder,split["test"],args.data_root,src["dataset"],device,
                  batch=8 if args.smoke else 48)
    del method,encoder
    torch.cuda.empty_cache()
    head=nn.Conv2d(768,5,1).to(device)
    head.load_state_dict(fit["state"],strict=True)
    metrics=evaluate(head,test,device)
    if not args.smoke and metrics["evaluated_classes"]!=[0,1,2,3,4]:
        raise ValueError("Official BCSS five-class test missing a class; cannot claim full mIoU")
    result={**selection,"test":metrics,"test_used_for_tuning":False}
    (out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print("COMPLETED",json.dumps(dict(dataset=src["dataset"],method=args.method,
          epoch=args.epoch,val_miou=fit["val_miou"],test=result["test"])),flush=True)

if __name__=="__main__":main()
