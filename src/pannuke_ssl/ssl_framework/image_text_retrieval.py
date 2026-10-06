from __future__ import annotations

import csv, hashlib, io, json, math, random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoTokenizer, CLIPModel

from pannuke_ssl.data import loader_kwargs
from pannuke_ssl.utils import atomic_json_dump, seed_everything, write_csv
from .image_text_datasets import load_image_text_manifest
from .trainer import load_checkpoint
from .retrieval_protocol import caption_aware_metrics, multi_positive_clip_loss


class ImageTextPairDataset(Dataset):
    def __init__(self, rows:list[Mapping[str,Any]], root:Path):
        self.rows=[dict(x) for x in rows]; self.root=Path(root)
    def __len__(self): return len(self.rows)
    @staticmethod
    def _prep(image:Image.Image)->torch.Tensor:
        w,h=image.size; scale=256.0/min(w,h)
        image=image.resize((max(256,int(math.floor(w*scale+.5))),max(256,int(math.floor(h*scale+.5)))),Image.Resampling.BICUBIC)
        l=(image.width-256)//2; t=(image.height-256)//2
        a=np.asarray(image.crop((l,t,l+256,t+256)).convert('RGB'),dtype=np.uint8).copy()
        return torch.from_numpy(a).permute(2,0,1)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(self.root/r['relative_path']) as im: x=self._prep(im)
        return x,str(r['text']),i


def _freeze(module:nn.Module)->nn.Module:
    module.eval()
    for p in module.parameters(): p.requires_grad_(False)
    return module


def _assert_frozen(module:nn.Module,name:str):
    bad=[n for n,p in module.named_parameters() if p.requires_grad]
    if bad: raise AssertionError(f'{name} has trainable parameters: {bad[:5]}')
    if module.training: raise AssertionError(f'{name} must remain in eval mode')


@torch.inference_mode()
def _ssl_features(encoder:nn.Module, rows, root:Path, c:dict[str,Any], device:torch.device):
    _assert_frozen(encoder,'SSL encoder')
    ds=ImageTextPairDataset(rows,root)
    dl=DataLoader(ds,**loader_kwargs(int(c['downstream']['batch_size']),int(c['downstream']['num_workers']),shuffle=False))
    xs=[]
    for images,_,_ in dl:
        images=images.to(device,dtype=torch.float32,non_blocking=True).div_(255.)
        with torch.autocast('cuda',dtype=torch.bfloat16): z=encoder(images)
        if z.ndim==3: z=z.mean(1)
        if z.ndim!=2: raise RuntimeError(f'Unexpected SSL feature shape {tuple(z.shape)}')
        xs.append(z.float().cpu())
    return torch.cat(xs)


@torch.inference_mode()
def _text_features(text_branch:nn.Module,text_projection:nn.Module,tokenizer,texts:list[str],device:torch.device,batch_size:int):
    _assert_frozen(text_branch,'PLIP text encoder'); _assert_frozen(text_projection,'PLIP text projection')
    outs=[]
    for start in range(0,len(texts),batch_size):
        tok=tokenizer(texts[start:start+batch_size],padding=True,truncation=True,max_length=77,return_tensors='pt')
        tok={k:v.to(device) for k,v in tok.items()}
        out=text_branch(input_ids=tok['input_ids'],attention_mask=tok.get('attention_mask'))
        pooled=out.pooler_output
        z=text_projection(pooled)
        outs.append(F.normalize(z.float(),dim=-1).cpu())
    return torch.cat(outs)


def _sym_clip_loss(image_emb:torch.Tensor,text_emb:torch.Tensor,logit_scale:float):
    logits=logit_scale*(F.normalize(image_emb,dim=-1)@F.normalize(text_emb,dim=-1).T)
    target=torch.arange(logits.shape[0],device=logits.device)
    return .5*(F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))


def _retrieval_metrics(image_emb:torch.Tensor,text_emb:torch.Tensor,caption_ids=None):
    if caption_ids is not None:
        return caption_aware_metrics(image_emb,text_emb,caption_ids)
    sim=F.normalize(image_emb.float(),dim=-1)@F.normalize(text_emb.float(),dim=-1).T
    n=sim.shape[0]; truth=torch.arange(n)
    out={}
    for direction,matrix in [('i2t',sim),('t2i',sim.T)]:
        order=torch.argsort(matrix,dim=1,descending=True)
        vals=[]
        for k in (1,5,10):
            kk=min(k,n); r=float((order[:,:kk]==truth[:,None]).any(dim=1).float().mean().item())
            out[f'{direction}_r@{k}']=r; vals.append(r)
        out[f'{direction}_mean_recall']=sum(vals)/3.0
    out['overall_mean_recall']=(out['i2t_mean_recall']+out['t2i_mean_recall'])/2.0
    return out


def _train_trial(train_i,train_t,val_i,val_t,*,lr,wd,epochs,batch_size,seed,device,logit_scale,patience=None,train_caption_ids=None,val_caption_ids=None):
    seed_everything(seed)
    head=nn.Linear(train_i.shape[1],train_t.shape[1],bias=False).to(device)
    trainable=[name for name,param in head.named_parameters() if param.requires_grad]
    if trainable != ['weight']:
        raise AssertionError(f'Only the image projector weight may be trainable, got {trainable}')
    opt=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=wd)
    groups=torch.arange(len(train_i)) if train_caption_ids is None else train_caption_ids
    ds=TensorDataset(train_i,train_t,groups); gen=torch.Generator().manual_seed(seed)
    dl=DataLoader(ds,batch_size=batch_size,shuffle=True,generator=gen,num_workers=0,drop_last=False)
    best=None; history=[]; stale=0
    for epoch in range(1,epochs+1):
        head.train(); losses=[]
        for xi,xt,gi in dl:
            xi=xi.to(device); xt=xt.to(device); opt.zero_grad(set_to_none=True)
            loss=(_sym_clip_loss(head(xi),xt,logit_scale) if train_caption_ids is None else multi_positive_clip_loss(head(xi),xt,gi.to(device),logit_scale))
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite retrieval training loss')
            loss.backward()
            if not torch.isfinite(head.weight.grad).all(): raise RuntimeError('Nonfinite retrieval head gradient')
            opt.step(); losses.append(float(loss.detach()))
        head.eval()
        with torch.inference_mode(): metrics=_retrieval_metrics(head(val_i.to(device)).cpu(),val_t,val_caption_ids)
        row={'epoch':epoch,'train_loss':sum(losses)/max(1,len(losses)),**metrics}; history.append(row)
        score=metrics['overall_mean_recall']
        if best is None or score>best['score']:
            best={'score':score,'epoch':epoch,'state':{k:v.detach().cpu().clone() for k,v in head.state_dict().items()},'metrics':metrics}; stale=0
        else: stale+=1
        if patience is not None and stale>=patience: break
    assert best is not None
    return {'learning_rate':lr,'weight_decay':wd,'best_epoch':best['epoch'],'val_overall_mean_recall':best['score'],'val_metrics':best['metrics'],'state':best['state'],'history':history}


def _tune(train_i,train_t,val_i,val_t,protocol,*,seed,device,logit_scale,train_caption_ids=None,val_caption_ids=None):
    lrs=[float(x) for x in protocol['learning_rates']]; wds=[float(x) for x in protocol['weight_decays']]
    te=int(protocol['tuning_epochs']); bs=int(protocol['batch_size']); board=[]
    best_lr=None
    for j,lr in enumerate(lrs,1):
        tr=_train_trial(train_i,train_t,val_i,val_t,lr=lr,wd=wds[0],epochs=te,batch_size=bs,seed=seed,device=device,logit_scale=logit_scale,train_caption_ids=train_caption_ids,val_caption_ids=val_caption_ids)
        board.append({'stage':'lr','trial':j,'learning_rate':lr,'weight_decay':wds[0],'best_epoch':tr['best_epoch'],'val_overall_mean_recall':tr['val_overall_mean_recall']})
        if best_lr is None or tr['val_overall_mean_recall']>best_lr['val_overall_mean_recall']: best_lr=tr
    selected_lr=float(best_lr['learning_rate']); best_wd=None
    for j,wd in enumerate(wds,1):
        tr=_train_trial(train_i,train_t,val_i,val_t,lr=selected_lr,wd=wd,epochs=te,batch_size=bs,seed=seed,device=device,logit_scale=logit_scale,train_caption_ids=train_caption_ids,val_caption_ids=val_caption_ids)
        board.append({'stage':'weight_decay','trial':j,'learning_rate':selected_lr,'weight_decay':wd,'best_epoch':tr['best_epoch'],'val_overall_mean_recall':tr['val_overall_mean_recall']})
        if best_wd is None or tr['val_overall_mean_recall']>best_wd['val_overall_mean_recall']: best_wd=tr
    return selected_lr,float(best_wd['weight_decay']),board


def run_image_text_retrieval(c, checkpoint, out, *, dataset):
    from .retrieval_repair import run_train_val, run_test
    cache_root = Path(out).parent.parent / 'caption_aware_feature_cache'
    run_train_val(c, checkpoint, out, dataset=dataset, cache_root=cache_root)
    return run_test(c, checkpoint, out, dataset=dataset, cache_root=cache_root)
