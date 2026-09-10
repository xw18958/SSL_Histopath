from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np,torch
from sklearn.metrics import classification_report,confusion_matrix
from torch.utils.data import DataLoader
from pannuke_ssl.data import PanNukeImageDataset,loader_kwargs
from pannuke_ssl.parquet import build_source_index,preload_images,read_metadata,verify_records
from pannuke_ssl.probe import _evaluate,_fit_probe
from pannuke_ssl.utils import atomic_json_dump,seed_everything,write_csv
from .trainer import load_checkpoint

@torch.inference_mode()
def _extract(encoder,rows,c,device):
    index=build_source_index(Path(c['data']['root'])); verify_records(rows,index); cache=preload_images(rows,index); loader=DataLoader(PanNukeImageDataset(rows,index,cache,include_label=True,include_key=True),**loader_kwargs(int(c['downstream']['batch_size']),int(c['downstream']['num_workers']),shuffle=False)); xs,ys,keys=[],[],[]
    for images,y,fold,idx in loader:
        x=images.to(device,dtype=torch.float32,non_blocking=True).div_(255.)
        with torch.autocast('cuda',dtype=torch.bfloat16): tokens=encoder(x)
        xs.append(tokens.float().mean(1).cpu()); ys.append(y.long().cpu()); keys.extend(zip(fold.tolist(),idx.tolist()))
    return torch.cat(xs),torch.cat(ys),np.asarray(keys,dtype=np.int64)

def run_downstream(c:dict[str,Any],checkpoint:Path,out:Path):
    from pannuke_ssl.ssl_methods.registry import build_method
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if (out/'test_started.json').exists(): raise FileExistsError('Test already started')
    seed_everything(int(c['seed'])); device=torch.device('cuda'); method=build_method(c,device); ck=load_checkpoint(method,Path(checkpoint),device); enc=method.encoder.eval(); [p.requires_grad_(False) for p in enc.parameters()]
    rows=read_metadata(Path(c['data']['metadata_csv'])); by={s:[r for r in rows if r['split']==s] for s in ('train','val','test')}
    # Only train/val are decoded before probe selection.
    tx,ty,_=_extract(enc,by['train'],c,device); vx,vy,_=_extract(enc,by['val'],c,device); mean=tx.mean(0,keepdim=True); std=tx.std(0,keepdim=True,unbiased=True).clamp_min(1e-6); feats={'train':((tx-mean)/std,ty),'val':((vx-mean)/std,vy)}; pc=c['downstream']['probe']; board=[]; best=None
    for lr in pc['learning_rates']:
        for wd in pc['weight_decays']:
            tr=_fit_probe(feats,learning_rate=float(lr),weight_decay=float(wd),maximum_epochs=int(pc['maximum_epochs']),patience=int(pc['early_stopping_patience']),seed=int(c['seed'])); board.append({k:v for k,v in tr.items() if k not in ('state','history')})
            if best is None or tr['val_macro_f1']>best['val_macro_f1']: best=tr
    write_csv(board,out/'probe_leaderboard.csv'); write_csv(best['history'],out/'selected_probe_metrics.csv'); torch.save({'classifier':{k:v.cpu() for k,v in best['state'].items()},'feature_mean':mean,'feature_std':std,'selection':{'learning_rate':best['learning_rate'],'weight_decay':best['weight_decay'],'best_epoch':best['best_epoch'],'validation_macro_f1':best['val_macro_f1'],'test_used':False}},out/'best_linear_probe.pt'); atomic_json_dump({'learning_rate':best['learning_rate'],'weight_decay':best['weight_decay'],'best_epoch':best['best_epoch'],'validation_macro_f1':best['val_macro_f1'],'test_used':False},out/'probe_selection.json')
    # Exclusive marker is created before any test image is decoded.
    with (out/'test_started.json').open('x') as f: json.dump({'encoder_epoch':int(ck['epoch']),'probe_selected':True},f)
    x,y,keys=_extract(enc,by['test'],c,device); state=torch.load(out/'best_linear_probe.pt',map_location='cpu',weights_only=False); clf=torch.nn.Linear(tx.shape[1],19).to(device).eval(); clf.load_state_dict(state['classifier']); loss,metrics,preds=_evaluate(clf,(x-state['feature_mean'])/state['feature_std'],y,device); np.savez_compressed(out/'test_predictions.npz',labels=y.numpy(),predictions=preds,keys=keys)
    report=classification_report(y.numpy(),preds,output_dict=True,zero_division=0); write_csv([{'class_id':i,**report[str(i)]} for i in range(19)],out/'test_per_class_metrics.csv'); np.savetxt(out/'test_confusion_matrix.csv',confusion_matrix(y.numpy(),preds,labels=list(range(19))),delimiter=',',fmt='%d')
    result={'method':c['method']['name'],'encoder_epoch':int(ck['epoch']),'probe_validation_macro_f1':float(best['val_macro_f1']),'test_loss':float(loss),'test':metrics,'test_images':int(y.numel()),'test_evaluated_once':True}; atomic_json_dump(result,out/'test_metrics.json'); return result
