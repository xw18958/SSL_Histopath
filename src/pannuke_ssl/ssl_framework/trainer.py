from __future__ import annotations
import json,math,time
from pathlib import Path
from typing import Any
import torch
from pannuke_ssl.utils import atomic_json_dump,seed_everything,write_csv
from .data_validation import Validator,build_ssl_loader,module_sha
from .early import EarlyStopper

def save_checkpoint(path:Path,method,c:dict[str,Any],epoch:int,selection:dict[str,Any]):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.tmp')
    torch.save({'framework':'ssl_standard_v1','method_name':c['method']['name'],'epoch':int(epoch),'method_state':{k:v.detach().cpu() for k,v in method.state_dict().items()},'encoder_sha256':module_sha(method.encoder),'config':c,'selection':selection},tmp); tmp.replace(path)

def load_checkpoint(method,path:Path,device):
    x=torch.load(path,map_location=device,weights_only=False)
    if x.get('framework')!='ssl_standard_v1': raise ValueError('Not a standard SSL checkpoint')
    method.load_state_dict(x['method_state'],strict=True); return x

def _set_schedule(method,optimizer,step,total):
    vals=method.schedule(step,max(1,total))
    for g in optimizer.param_groups:
        g['lr']=float(vals['lr'])
        if not g.get('_no_weight_decay',False): g['weight_decay']=float(vals['weight_decay'])
    return vals

def train_ssl(c:dict[str,Any],out:Path,*,epochs:int|None=None,interval:int|None=None,early_stop:bool|None=None):
    from pannuke_ssl.ssl_methods.registry import build_method
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if (out/'pretrain_metrics.csv').exists() or (out/'run_summary.json').exists(): raise FileExistsError(f'Refuse overwrite: {out}')
    seed_everything(int(c['seed'])); torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=bool(c['training']['tf32']); torch.backends.cudnn.allow_tf32=bool(c['training']['tf32'])
    device=torch.device('cuda'); method=build_method(c,device); initial=module_sha(method.encoder); loader=build_ssl_loader(c,method); optimizer=method.build_optimizer(); params=method.optimizer_parameters(); validator=Validator(c,out,device)
    n_epochs=int(epochs or c['training']['max_epochs']); val_interval=int(interval or c['validation']['interval_epochs']); enabled=bool(c['early_stopping']['enabled'] if early_stop is None else early_stop); stopper=EarlyStopper(enabled=enabled,min_epochs=int(c['early_stopping']['min_epochs']),patience=int(c['early_stopping']['patience_monitors']),delta=float(c['early_stopping']['minimum_delta']))
    atomic_json_dump(c,out/'resolved_config.json'); total=n_epochs*len(loader); step=0; history=[]; reason='max_epochs'; started=time.perf_counter(); best=out/'checkpoints/best.pt'; last=out/'checkpoints/last.pt'
    for epoch in range(1,n_epochs+1):
        method.train_mode(); torch.cuda.reset_peak_memory_stats(); t0=time.perf_counter(); sums={}; seen=0; sched={}; after={}
        for batch in loader:
            optimizer.zero_grad(set_to_none=True); sched=_set_schedule(method,optimizer,step,total); result=method.training_step(batch,bf16=bool(c['training']['bf16']))
            if not torch.isfinite(result.loss): raise FloatingPointError(f'Non-finite SSL loss epoch={epoch} step={step}')
            result.loss.backward(); norm=torch.nn.utils.clip_grad_norm_(params,float('inf'))
            if not torch.isfinite(norm): raise FloatingPointError('Non-finite SSL gradients')
            clip=c['method'].get('gradient_clip_norm')
            if clip is not None: torch.nn.utils.clip_grad_norm_(params,float(clip))
            optimizer.step(); after=method.after_optimizer_step(step,total); b=method.batch_size(batch); seen+=b; sums['loss']=sums.get('loss',0.)+float(result.loss.detach())*b; sums['gradient_norm']=sums.get('gradient_norm',0.)+float(norm)*b
            for k,v in result.metrics.items(): sums[k]=sums.get(k,0.)+float(v)*b
            step+=1
        expected=int(c['data']['expected_ssl_images'])
        if seen!=expected: raise AssertionError(f'Every epoch must use {expected} images, got {seen}')
        sec=time.perf_counter()-t0; row={'epoch':epoch,'samples':seen,'seconds':sec,'samples_per_second':seen/sec,'learning_rate':sched.get('lr'),'weight_decay':sched.get('weight_decay'),'peak_gpu_memory_gib':torch.cuda.max_memory_allocated()/2**30,**{k:v/seen for k,v in sums.items()},**after}; history.append(row); write_csv(history,out/'pretrain_metrics.csv'); print(json.dumps({'pretrain':row}),flush=True)
        selection={'metric':'linear_val_macro_f1','minimum_delta':.005,'test_used':False}; save_checkpoint(last,method,c,epoch,selection)
        if epoch%val_interval==0:
            metrics=validator.evaluate(method.encoder,epoch); u=stopper.update(epoch,float(metrics['linear_val_macro_f1'])); print(json.dumps({'validation':metrics,'early_stopping':u}),flush=True)
            if u['improved']: save_checkpoint(best,method,c,epoch,{**selection,'value':u['best_score'],'best_epoch':u['best_epoch']})
            if u['should_stop']: reason='validation_plateau'; break
    if not best.exists(): save_checkpoint(best,method,c,int(history[-1]['epoch']),{'metric':'linear_val_macro_f1','test_used':False,'value':None,'best_epoch':int(history[-1]['epoch'])})
    summary={'method':c['method']['name'],'epochs_planned':n_epochs,'epochs_completed':int(history[-1]['epoch']),'stop_reason':reason,'best_epoch':stopper.best_epoch or int(history[-1]['epoch']),'best_validation_linear_macro_f1':stopper.best if math.isfinite(stopper.best) else None,'initial_encoder_sha256':initial,'ssl_images_per_epoch':int(c['data']['expected_ssl_images']),'labels_loaded_for_ssl':False,'test_used_for_ssl_selection':False,'source_metadata':c['method']['source_metadata'],'seconds':time.perf_counter()-started}; atomic_json_dump(summary,out/'run_summary.json'); return summary
