#!/usr/bin/env python3
"""Frozen ViT-B/32 GlaS gland INSTANCE probe with native-resolution evaluation.

Uses original integer instance labels, distance-transform head, watershed instances,
held-out TestA/TestB, and per-image object F1, object Dice, object Hausdorff.
No SSL encoder or existing segmentation pipeline is modified.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage as ndi
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pannuke_ssl.ssl_methods.registry import build_method
from pannuke_ssl.ssl_framework.trainer import load_checkpoint

SEED=20260903
TILE=256
STRIDE=192
TRAIN_PATCHES_PER_IMAGE=6
BATCH=24
LR=(0.001,0.01)
WD=(0.0,0.0001)
HP_EPOCHS=3
FINAL_EPOCHS=18
PATIENCE=4

def hashfile(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
    return h.hexdigest()

def read_manifest(path):
    doc=json.loads(path.read_text())
    expected=doc.pop("manifest_sha256")
    got=hashlib.sha256(json.dumps(doc,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    if expected!=got or doc.get("dataset")!="glas" or doc.get("version")!="pathology_semantic_seg_v1":
        raise ValueError("Invalid GlaS source manifest")
    doc["manifest_sha256"]=expected
    groups={};ids=set();counts={}
    for row in doc["records"]:
        if row["image"] in ids:raise ValueError("Duplicate source image")
        ids.add(row["image"])
        if row["group"] in groups and groups[row["group"]]!=row["split"]:
            raise ValueError("Group crosses partitions")
        groups[row["group"]]=row["split"]
        counts[row["split"]]=counts.get(row["split"],0)+1
    if counts!={"train":70,"val":15,"test":80}:raise ValueError(f"Wrong GlaS split {counts}")
    test=[x for x in doc["records"] if x["split"]=="test"]
    if sum(x["category"]=="testa" for x in test)!=60 or sum(x["category"]=="testb" for x in test)!=20:
        raise ValueError("Not the official 60/20 GlaS evaluation")
    return doc

def load_image_mask(row,root):
    with Image.open(root/row["image"]) as f:
        img=np.asarray(f.convert("RGB"),dtype=np.uint8).copy()
    with Image.open(root/row["mask"]) as f:
        mask=np.asarray(f)
    if mask.ndim!=2 or not np.issubdtype(mask.dtype,np.integer):
        raise ValueError("GlaS annotation must be integer instance ID plane")
    if mask.shape!=img.shape[:2] or int(mask.min())<0:raise ValueError("Mask/image geometry")
    return img,mask.astype(np.int32)

def object_distance(instance_ids):
    dist=np.zeros(instance_ids.shape,np.float32)
    for idx in np.unique(instance_ids):
        if idx==0:continue
        obj=instance_ids==idx
        y=ndi.distance_transform_edt(obj)
        hi=y.max()
        if hi>0:dist[obj]=y[obj]/hi
    return dist

def axis_positions(n,stride=STRIDE):
    if n<TILE:raise ValueError("Native image smaller than patch")
    xs=list(range(0,n-TILE+1,stride))
    if xs[-1]!=n-TILE:xs.append(n-TILE)
    return xs

def make_windows(row,root,mode):
    img,lab=load_image_mask(row,root)
    h,w=lab.shape
    assert h>=TILE and w>=TILE
    xx=axis_positions(w)
    yy=axis_positions(h)
    positions=[(y,x) for y in yy for x in xx]
    if mode=="train":
        if len(positions)>=TRAIN_PATCHES_PER_IMAGE:
            # Fixed grid representative of whole image. Independent of masks.
            positions=[positions[i] for i in np.linspace(0,len(positions)-1,TRAIN_PATCHES_PER_IMAGE).round().astype(int)]
        else:
            positions=positions*(1+TRAIN_PATCHES_PER_IMAGE//len(positions))
            positions=positions[:TRAIN_PATCHES_PER_IMAGE]
        d=object_distance(lab)
        result=[]
        for y,x in positions:
            result.append((img[y:y+TILE,x:x+TILE].copy(),
                           (lab[y:y+TILE,x:x+TILE]>0).astype(np.float32),
                           d[y:y+TILE,x:x+TILE].copy(),(y,x)))
        return result,lab
    return [(img[y:y+TILE,x:x+TILE].copy(),(y,x)) for y,x in positions],lab

class RGBDataset(Dataset):
    def __init__(self,patches):self.patches=patches
    def __len__(self):return len(self.patches)
    def __getitem__(self,i):
        img=self.patches[i]
        return torch.from_numpy(img).permute(2,0,1).float().div_(255.)

@torch.inference_mode()
def frozen_features(encoder,patches,device):
    dl=DataLoader(RGBDataset(patches),batch_size=48,shuffle=False,num_workers=0)
    out=[]
    for batch in dl:
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            f=encoder(batch.to(device,non_blocking=True))
        if f.ndim!=3 or tuple(f.shape[1:])!=(64,768):raise ValueError(f"Unexpected encoder spatial tokens {tuple(f.shape)}")
        out.append(f.cpu().half())
    return torch.cat(out)

def collect(encoder,rows,root,device,mode):
    features=[];targets=[];images=[]
    for i,row in enumerate(rows):
        windows,gt=make_windows(row,root,mode)
        imgs=[w[0] for w in windows]
        f=frozen_features(encoder,imgs,device)
        features.append(f)
        if mode=="train":
            for w in windows:targets.append(torch.from_numpy(np.stack([w[1],w[2]]).copy()))
        else:
            images.append(dict(image=row["image"],split=row["split"],
                               category=row["category"],gt=gt,
                               coords=[w[1] for w in windows],features=f,
                               size=gt.shape))
    if mode=="train":
        return torch.cat(features),torch.stack(targets)
    return images

def head_logits(head,features):
    z=head(features.float().transpose(1,2).reshape(-1,768,8,8))
    return F.interpolate(z,size=(TILE,TILE),mode="bilinear",align_corners=False)

@torch.no_grad()
def native_prob_maps(head,image,device):
    patches=image["features"]
    h,w=image["size"]
    accum=np.zeros((2,h,w),np.float32)
    weights=np.zeros((h,w),np.float32)
    head.eval()
    for pos in range(0,len(patches),BATCH):
        z=head_logits(head,patches[pos:pos+BATCH].to(device))
        probabilities=z.sigmoid().cpu().numpy()
        for idx,prob in enumerate(probabilities):
            y,x=image["coords"][pos+idx]
            accum[:,y:y+TILE,x:x+TILE]+=prob
            weights[y:y+TILE,x:x+TILE]+=1
    if not np.all(weights):raise AssertionError("Native inference has uncovered pixels")
    return accum/weights[None,:,:]

def relabel_small(inst,min_area=200):
    ids,counts=np.unique(inst,return_counts=True)
    out=np.zeros(inst.shape,np.int32)
    n=0
    for idx,area in zip(ids,counts):
        if idx==0 or area<min_area:continue
        n+=1
        out[inst==idx]=n
    return out

def watershed_instances(prob_fg,prob_dist,peak_threshold=.35,min_area=200):
    if prob_fg.shape!=prob_dist.shape:raise ValueError("Native maps must align")
    fg=prob_fg>=.5
    fg=ndi.binary_fill_holes(fg)
    fg=ndi.binary_opening(fg,iterations=1)
    if not np.any(fg):return np.zeros(prob_fg.shape,np.int32)
    distance=ndi.gaussian_filter(prob_dist.astype(np.float32),sigma=2)
    # Use predicted topology to generate seed markers (model-trained distance head).
    peak=(distance==ndi.maximum_filter(distance,size=29))&(distance>=peak_threshold)&fg
    peak=ndi.binary_dilation(peak,iterations=2)&fg
    seeds,n=ndi.label(peak)
    if n==0:
        seeds,n=ndi.label(fg)
    gradient=np.clip(np.round(255*(1-distance)),0,255).astype(np.uint8)
    seg=ndi.watershed_ift(gradient,seeds.astype(np.int32),structure=ndi.generate_binary_structure(2,1))
    seg[~fg]=0
    return relabel_small(seg,min_area=min_area)

def label_overlap(pred,gt):
    pids,pcts=np.unique(pred,return_counts=True)
    gids,gcts=np.unique(gt,return_counts=True)
    pa={int(a):int(c) for a,c in zip(pids,pcts) if a>0}
    ga={int(a):int(c) for a,c in zip(gids,gcts) if a>0}
    overlap={}
    for p in pa:
        g,c=np.unique(gt[pred==p],return_counts=True)
        for gi,ci in zip(g,c):
            if gi>0:overlap[(p,int(gi))]=int(ci)
    return pa,ga,overlap

def object_f1(pred,gt):
    """GlaS gland detection: dominant intersecting GT covers >50% of GT gland;
    one-to-one matches prevent double counting split predictions."""
    pa,ga,ov=label_overlap(pred,gt)
    if not pa and not ga:return 1.
    if not pa or not ga:return 0.
    candidates=[]
    for p in pa:
        overlaps=[(inter,g) for (pp,g),inter in ov.items() if pp==p]
        if overlaps:
            inter,g=max(overlaps)
            if inter/ga[g]>.5:
                candidates.append((inter/ga[g],inter,p,g))
    candidates.sort(reverse=True)
    used_p=set();used_g=set()
    for _,__,p,g in candidates:
        if p not in used_p and g not in used_g:
            used_p.add(p);used_g.add(g)
    tp=len(used_p)
    return 2.*tp/(len(pa)+len(ga))

def object_dice(pred,gt):
    pa,ga,ov=label_overlap(pred,gt)
    if not pa and not ga:return 1.
    if not pa or not ga:return 0.
    total_p=sum(pa.values());total_g=sum(ga.values())
    ans_p=0.;ans_g=0.
    for p,a in pa.items():
        matches=[(inter,g) for (pp,g),inter in ov.items() if pp==p]
        if matches:
            inter,g=max(matches)
            ans_p+=a/total_p*(2*inter/(a+ga[g]))
    for g,a in ga.items():
        matches=[(inter,p) for (p,gg),inter in ov.items() if gg==g]
        if matches:
            inter,p=max(matches)
            ans_g+=a/total_g*(2*inter/(a+pa[p]))
    return (ans_p+ans_g)/2.

def hausdorff(a,b):
    if not a.any() and not b.any():return 0.
    if not a.any() or not b.any():return float(np.hypot(*a.shape))
    # Exact 2D symmetric Hausdorff (all-object pixels) via Euclidean DT.
    return float(max(ndi.distance_transform_edt(~a)[b].max(),
                     ndi.distance_transform_edt(~b)[a].max()))

def object_hausdorff(pred,gt):
    pa,ga,overlap=label_overlap(pred,gt)
    if not pa and not ga:return 0.
    if not pa or not ga:return float(np.hypot(*pred.shape))
    total_p=sum(pa.values());total_g=sum(ga.values())
    predicted={p:(pred==p) for p in pa};real={g:(gt==g) for g in ga}
    def nearest(mask,other,ids,label,is_p):
        ov=[(v,b if is_p else a) for (a,b),v in overlap.items()
            if (a==label if is_p else b==label)]
        if ov:
            _,best=max(ov)
            return other[best]
        distances=[(hausdorff(mask,other[k]),k) for k in ids]
        return other[min(distances)[1]]
    one=sum((pa[p]/total_p)*hausdorff(predicted[p],nearest(predicted[p],real,ga,p,True)) for p in pa)
    two=sum((ga[g]/total_g)*hausdorff(real[g],nearest(real[g],predicted,pa,g,False)) for g in ga)
    return (one+two)/2.

def instance_metrics(pred,gt,with_hd=False):
    if pred.shape!=gt.shape or pred.dtype.kind not in "iu":raise ValueError("Incorrect instance map")
    result=dict(object_f1=object_f1(pred,gt),object_dice=object_dice(pred,gt),
                predicted_glands=int(np.unique(pred).size-1),
                gt_glands=int(np.unique(gt).size-1))
    if with_hd:result["object_hausdorff"]=object_hausdorff(pred,gt)
    return result

@torch.inference_mode()
def validate(head,images,device,config,with_hd=False):
    result=[]
    for image in images:
        p=native_prob_maps(head,image,device)
        pred=watershed_instances(p[0],p[1],**config)
        met=instance_metrics(pred,image["gt"],with_hd=with_hd)
        met.update(image=image["image"],category=image["category"])
        result.append(met)
    keys=("object_f1","object_dice","object_hausdorff") if with_hd else ("object_f1","object_dice")
    agg={k:float(np.mean([r[k] for r in result])) for k in keys}
    return agg,result

def fit(train,val,device,lr,wd,epochs,patience,post):
    torch.manual_seed(SEED)
    head=nn.Conv2d(768,2,1).to(device)
    opt=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=wd)
    xs,ys=train
    best=-1.;best_state=None;best_ep=0;history=[]
    for ep in range(1,epochs+1):
        head.train();losses=[]
        order=torch.randperm(len(xs),generator=torch.Generator().manual_seed(SEED+ep)).tolist()
        for start in range(0,len(order),BATCH):
            ids=order[start:start+BATCH]
            x=xs[ids].to(device);target=ys[ids].to(device)
            opt.zero_grad(set_to_none=True)
            scores=head_logits(head,x)
            foreground_loss=F.binary_cross_entropy_with_logits(scores[:,0],target[:,0])
            mask=target[:,0]>.5
            distance_loss=F.smooth_l1_loss(scores[:,1].sigmoid()[mask],target[:,1][mask]) if mask.any() else 0*scores[:,1].mean()
            loss=foreground_loss+.5*distance_loss
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite distance/fg loss")
            loss.backward();opt.step()
            losses.append(float(loss.detach()))
        val_met,_=validate(head,val,device,post)
        score=val_met["object_dice"]
        history.append(dict(epoch=ep,loss=float(np.mean(losses)),val_object_dice=score,val_object_f1=val_met["object_f1"]))
        if score>best+1e-8:
            best=score;best_ep=ep;best_state={k:v.cpu().clone() for k,v in head.state_dict().items()}
        elif ep-best_ep>=patience:break
    return dict(state=best_state,best_epoch=best_ep,val_object_dice=best,history=history)

def train_head(train,val,device):
    base=dict(peak_threshold=.35,min_area=200)
    board=[]
    best_lr=None
    for lr in LR:
        trial=fit(train,val,device,lr,WD[0],HP_EPOCHS,HP_EPOCHS,base)
        board.append(dict(stage="lr",lr=lr,wd=WD[0],val_object_dice=trial["val_object_dice"]))
        if best_lr is None or trial["val_object_dice"]>best_lr[1]:best_lr=(lr,trial["val_object_dice"])
    best_wd=None
    for wd in WD:
        trial=fit(train,val,device,best_lr[0],wd,HP_EPOCHS,HP_EPOCHS,base)
        board.append(dict(stage="wd",lr=best_lr[0],wd=wd,val_object_dice=trial["val_object_dice"]))
        if best_wd is None or trial["val_object_dice"]>best_wd[1]:best_wd=(wd,trial["val_object_dice"])
    final=fit(train,val,device,best_lr[0],best_wd[0],FINAL_EPOCHS,PATIENCE,base)
    head=nn.Conv2d(768,2,1).to(device)
    head.load_state_dict(final["state"])
    # Independent sequential postprocessing selection on *validation* only.
    chosen=base.copy()
    post_board=[]
    for field,candidates in [("peak_threshold",(.25,.35,.50)),("min_area",(100,200,500))]:
        winner=None
        for value in candidates:
            arg={**chosen,field:value}
            m,_=validate(head,val,device,arg,False)
            post_board.append(dict(stage=field,value=value,object_dice=m["object_dice"],object_f1=m["object_f1"]))
            if winner is None or m["object_dice"]>winner[1]:winner=(value,m["object_dice"])
        chosen[field]=winner[0]
    final_val,_=validate(head,val,device,chosen,False)
    return final,chosen,dict(lr=best_lr[0],wd=best_wd[0],selection=board,post_selection=post_board,final_val=final_val)

def main():
    p=argparse.ArgumentParser()
    for arg in ("data-root","manifest","checkpoint","model-root","out-dir"):
        p.add_argument("--"+arg,type=Path,required=True)
    p.add_argument("--method",choices=["lejepa","simplex_k8","mae"],required=True)
    p.add_argument("--epoch",type=int,choices=(100,200),required=True)
    p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required")
    torch.cuda.set_device(args.gpu)
    device=torch.device(f"cuda:{args.gpu}")
    torch.set_num_threads(4)
    torch.manual_seed(SEED);np.random.seed(SEED)
    manifest=read_manifest(args.manifest)
    ckpt=args.checkpoint.resolve()
    digest=hashfile(ckpt)
    raw=torch.load(ckpt,map_location="cpu",weights_only=False)
    expected={"lejepa":"lejepa", "simplex_k8":"simplex_sigreg_lejepa", "mae":"mae"}[args.method]
    if raw.get("framework")!="ssl_standard_v1" or raw["method_name"]!=expected or int(raw["epoch"])!=args.epoch:
        raise ValueError("Checkpoint method/epoch/framework do not match request")
    cfg=copy.deepcopy(raw["config"])
    if cfg["method"]["name"]!=expected:raise ValueError("Wrong method config")
    if args.method=="simplex_k8" and cfg["method"]["objective"]["simplex_components"]!=8:
        raise ValueError("Not a K8 simplex checkpoint")
    cfg["backbone"]["config_dir"]=str(args.model_root/"plip_model")
    method=build_method(cfg,device)
    load_checkpoint(method,ckpt,device)
    enc=method.encoder.eval().to(device).requires_grad_(False)
    assert all(not a.requires_grad for a in enc.parameters())
    splits={s:[r for r in manifest["records"] if r["split"]==s] for s in ("train","val","test")}
    if args.smoke:
        splits={s:rows[:min(len(rows),4 if s=="train" else 2)] for s,rows in splits.items()}
    out=args.out_dir
    out.mkdir(parents=True,exist_ok=True)
    if (out/"result.json").exists() or (out/"test_started.json").exists():raise FileExistsError(out)
    print("EXTRACT",args.method,args.epoch,"train",len(splits["train"]),flush=True)
    train=collect(enc,splits["train"],args.data_root,device,"train")
    print("EXTRACT",args.method,args.epoch,"val",len(splits["val"]),flush=True)
    val=collect(enc,splits["val"],args.data_root,device,"eval")
    del enc,method,raw
    torch.cuda.empty_cache()
    print("FIT",args.method,args.epoch,flush=True)
    result,post,trace=train_head(train,val,device)
    meta=dict(dataset="glas_instance_v1",method=args.method,epoch=args.epoch,
              ssl_checkpoint_sha256=digest,source_manifest_sha256=manifest["manifest_sha256"],
              source_splits={s:len(v) for s,v in splits.items()},
              representation="frozen ViT-B/32 8x8x768 last patch tokens, pre-projector",
              training="native 256x256 fixed RGB crops; supervised 2-channel fg/distance linear probe; only head trained",
              postprocess="validation-selected watershed markers from predicted instance distance map; native-resolution TestA/TestB",
              eval="mean per-image object F1, symmetric area-weighted object Dice, symmetric area-weighted object Hausdorff",
              test_used_for_tuning=False,smoke=args.smoke,
              lr=trace["lr"],wd=trace["wd"],best_epoch=result["best_epoch"],
              hp_trace=trace,post_cfg=post)
    torch.save({"head":result["state"],"selection":meta},out/"instance_head.pt")
    (out/"selection.json").write_text(json.dumps(meta,indent=2)+"\n")
    with (out/"test_started.json").open("x") as f:json.dump({"checkpoint":digest,"manifest":manifest["manifest_sha256"]},f)
    # Held-out original native-size test images loaded only after HP + postprocessing selection.
    method=build_method(cfg,device)
    load_checkpoint(method,ckpt,device)
    enc=method.encoder.eval().to(device).requires_grad_(False)
    print("TEST_EXTRACT",args.method,args.epoch,len(splits["test"]),flush=True)
    test=collect(enc,splits["test"],args.data_root,device,"eval")
    del method,enc
    torch.cuda.empty_cache()
    head=nn.Conv2d(768,2,1).to(device)
    head.load_state_dict(result["state"])
    per_image=[]
    for im in test:
        prob=native_prob_maps(head,im,device)
        pred=watershed_instances(prob[0],prob[1],**post)
        m=instance_metrics(pred,im["gt"],with_hd=True)
        m.update(image=im["image"],category=im["category"])
        per_image.append(m)
    eval_splits={}
    for name in ("testa","testb"):
        subset=[x for x in per_image if x["category"]==name]
        if len(subset)!=(2 if args.smoke and name=="testa" else 0 if args.smoke else 60 if name=="testa" else 20):
            raise ValueError(f"Unexpected {name} number of test images: {len(subset)}")
        if not subset:continue
        eval_splits[name]={metric:float(np.mean([r[metric] for r in subset])) for metric in ("object_f1","object_dice","object_hausdorff")}
        eval_splits[name]["images"]=len(subset)
    overall={metric:float(np.mean([r[metric] for r in per_image])) for metric in ("object_f1","object_dice","object_hausdorff")}
    (out/"result.json").write_text(json.dumps({**meta,"testA":eval_splits.get("testa"),
         "testB":eval_splits.get("testb"),"combined":overall,"per_image":per_image},indent=2)+"\n")
    print("FINISHED",args.method,args.epoch,json.dumps({"testA":eval_splits.get("testa"),"testB":eval_splits.get("testb")}),flush=True)

if __name__=="__main__":main()
