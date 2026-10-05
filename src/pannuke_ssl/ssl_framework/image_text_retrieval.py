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


def _retrieval_metrics(image_emb:torch.Tensor,text_emb:torch.Tensor):
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


def _train_trial(train_i,train_t,val_i,val_t,*,lr,wd,epochs,batch_size,seed,device,logit_scale,patience=None):
    seed_everything(seed)
    head=nn.Linear(train_i.shape[1],train_t.shape[1],bias=False).to(device)
    trainable=[name for name,param in head.named_parameters() if param.requires_grad]
    if trainable != ['weight']:
        raise AssertionError(f'Only the image projector weight may be trainable, got {trainable}')
    opt=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=wd)
    ds=TensorDataset(train_i,train_t); gen=torch.Generator().manual_seed(seed)
    dl=DataLoader(ds,batch_size=batch_size,shuffle=True,generator=gen,num_workers=0,drop_last=False)
    best=None; history=[]; stale=0
    for epoch in range(1,epochs+1):
        head.train(); losses=[]
        for xi,xt in dl:
            xi=xi.to(device); xt=xt.to(device); opt.zero_grad(set_to_none=True)
            loss=_sym_clip_loss(head(xi),xt,logit_scale); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        head.eval()
        with torch.inference_mode(): metrics=_retrieval_metrics(head(val_i.to(device)).cpu(),val_t)
        row={'epoch':epoch,'train_loss':sum(losses)/max(1,len(losses)),**metrics}; history.append(row)
        score=metrics['overall_mean_recall']
        if best is None or score>best['score']:
            best={'score':score,'epoch':epoch,'state':{k:v.detach().cpu().clone() for k,v in head.state_dict().items()},'metrics':metrics}; stale=0
        else: stale+=1
        if patience is not None and stale>=patience: break
    assert best is not None
    return {'learning_rate':lr,'weight_decay':wd,'best_epoch':best['epoch'],'val_overall_mean_recall':best['score'],'val_metrics':best['metrics'],'state':best['state'],'history':history}


def _tune(train_i,train_t,val_i,val_t,protocol,*,seed,device,logit_scale):
    lrs=[float(x) for x in protocol['learning_rates']]; wds=[float(x) for x in protocol['weight_decays']]
    te=int(protocol['tuning_epochs']); bs=int(protocol['batch_size']); board=[]
    best_lr=None
    for j,lr in enumerate(lrs,1):
        tr=_train_trial(train_i,train_t,val_i,val_t,lr=lr,wd=wds[0],epochs=te,batch_size=bs,seed=seed,device=device,logit_scale=logit_scale)
        board.append({'stage':'lr','trial':j,'learning_rate':lr,'weight_decay':wds[0],'best_epoch':tr['best_epoch'],'val_overall_mean_recall':tr['val_overall_mean_recall']})
        if best_lr is None or tr['val_overall_mean_recall']>best_lr['val_overall_mean_recall']: best_lr=tr
    selected_lr=float(best_lr['learning_rate']); best_wd=None
    for j,wd in enumerate(wds,1):
        tr=_train_trial(train_i,train_t,val_i,val_t,lr=selected_lr,wd=wd,epochs=te,batch_size=bs,seed=seed,device=device,logit_scale=logit_scale)
        board.append({'stage':'weight_decay','trial':j,'learning_rate':selected_lr,'weight_decay':wd,'best_epoch':tr['best_epoch'],'val_overall_mean_recall':tr['val_overall_mean_recall']})
        if best_wd is None or tr['val_overall_mean_recall']>best_wd['val_overall_mean_recall']: best_wd=tr
    return selected_lr,float(best_wd['weight_decay']),board


def run_image_text_retrieval(c:dict[str,Any],checkpoint:Path,out:Path,*,dataset:str):
    from pannuke_ssl.ssl_methods.registry import build_method
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if (out/'test_started.json').exists(): raise FileExistsError('TEST has already started for this retrieval run')
    seed=int(c['seed']); seed_everything(seed); device=torch.device('cuda')
    manifest=load_image_text_manifest(dataset,Path(c['manifests']['root']))
    by=manifest['split_rows']; root=manifest['root']; protocol=manifest['alignment_protocol']
    if protocol is None: raise RuntimeError('Image-text alignment protocol is not frozen')
    method=build_method(c,device); ck=load_checkpoint(method,Path(checkpoint),device); encoder=_freeze(method.encoder.to(device)); _assert_frozen(encoder,'SSL encoder')
    plip_dir=Path(protocol['plip_model_dir']); plip=CLIPModel.from_pretrained(plip_dir,local_files_only=True).to(device).eval()
    tokenizer=AutoTokenizer.from_pretrained(plip_dir,local_files_only=True)
    _freeze(plip)
    text_branch=plip.text_model; text_projection=plip.text_projection
    _assert_frozen(text_branch,'PLIP text encoder'); _assert_frozen(text_projection,'PLIP text projection')
    logit_scale=float(plip.logit_scale.detach().exp().cpu())
    # The pretrained PLIP vision branch is deliberately not part of downstream alignment.
    # Keep only references to the frozen text components and release the unused vision branch.
    del plip
    torch.cuda.empty_cache()
    # TEST rows are deliberately untouched here. TRAIN/VAL only before selection.
    train_i=_ssl_features(encoder,by['train'],root,c,device); val_i=_ssl_features(encoder,by['val'],root,c,device)
    train_t=_text_features(text_branch,text_projection,tokenizer,[r['text'] for r in by['train']],device,int(protocol['text_batch_size']))
    val_t=_text_features(text_branch,text_projection,tokenizer,[r['text'] for r in by['val']],device,int(protocol['text_batch_size']))
    lr,wd,board=_tune(train_i,train_t,val_i,val_t,protocol,seed=seed,device=device,logit_scale=logit_scale); write_csv(board,out/'projector_tuning.csv')
    final=_train_trial(train_i,train_t,val_i,val_t,lr=lr,wd=wd,epochs=int(protocol['final_maximum_epochs']),batch_size=int(protocol['batch_size']),seed=seed,device=device,logit_scale=logit_scale,patience=int(protocol['final_early_stopping_patience']))
    write_csv(final['history'],out/'projector_training.csv')
    selection={'dataset':dataset,'encoder_epoch':int(ck['epoch']),'projector':'linear_bias_false','ssl_encoder_frozen':True,'plip_text_branch_frozen':True,'trainable_component':'image_projection_head_only','loss':'symmetric_clip_cross_entropy','fixed_plip_logit_scale':logit_scale,'learning_rate':lr,'weight_decay':wd,'best_epoch':final['best_epoch'],'validation':final['val_metrics'],'selection_metric':'val_overall_mean_recall','test_used':False}
    torch.save({'projector':final['state'],'selection':selection,'input_dim':int(train_i.shape[1]),'output_dim':int(train_t.shape[1])},out/'best_image_projector.pt'); atomic_json_dump(selection,out/'projector_selection.json')
    # Exclusive marker BEFORE TEST images or TEST text are decoded/tokenized.
    with (out/'test_started.json').open('x') as f: json.dump({'projector_selected':True,'test_used_for_selection':False,**selection},f)
    test_i=_ssl_features(encoder,by['test'],root,c,device)
    test_t=_text_features(text_branch,text_projection,tokenizer,[r['text'] for r in by['test']],device,int(protocol['text_batch_size']))
    head=nn.Linear(train_i.shape[1],train_t.shape[1],bias=False).to(device); head.load_state_dict(final['state']); head.eval()
    with torch.inference_mode(): test_img=head(test_i.to(device)).cpu(); metrics=_retrieval_metrics(test_img,test_t)
    result={'method':c['method']['name'],'dataset':dataset,'encoder_epoch':int(ck['epoch']),'pairs':len(by['test']),'test':metrics,'test_evaluated_once':True,'selection':selection}
    atomic_json_dump(result,out/'test_retrieval_metrics.json')
    return result
