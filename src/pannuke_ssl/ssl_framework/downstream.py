from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping
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
        with torch.autocast('cuda',dtype=torch.bfloat16): features=encoder(x)
        # Standard SSL encoders expose final patch tokens [B,T,D].  Frozen
        # external adapters may expose their pre-registered [B,D] readout.
        if features.ndim==3: features=features.mean(1)
        expected_dim=c['downstream'].get('feature_dim')
        if features.ndim!=2 or (expected_dim is not None and features.shape[1]!=int(expected_dim)):
            raise RuntimeError(f'Unexpected probe feature shape {tuple(features.shape)}; expected feature_dim={expected_dim}')
        xs.append(features.float().cpu()); ys.append(y.long().cpu()); keys.extend(zip(fold.tolist(),idx.tolist()))
    return torch.cat(xs),torch.cat(ys),np.asarray(keys,dtype=np.int64)

def _split_rows(c:dict[str,Any]):
    rows=read_metadata(Path(c['data']['metadata_csv'])); by={s:[r for r in rows if r['split']==s] for s in ('train','val','test')}
    expected={'train':int(c['downstream']['train_count']),'val':int(c['downstream']['validation_count']),'test':int(c['downstream']['test_count'])}
    observed={split:len(split_rows) for split,split_rows in by.items()}
    if observed!=expected:
        raise AssertionError(f'Fixed downstream split mismatch: observed={observed}, expected={expected}')
    return by

def _run_downstream(c:dict[str,Any],encoder,out:Path,*,encoder_epoch:int|str,encoder_metadata:Mapping[str,Any]|None=None):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if (out/'test_started.json').exists(): raise FileExistsError('Test already started')
    seed_everything(int(c['seed'])); device=torch.device('cuda'); enc=encoder.to(device).eval(); [p.requires_grad_(False) for p in enc.parameters()]
    by=_split_rows(c)
    # Only train/val are decoded before probe selection.
    tx,ty,_=_extract(enc,by['train'],c,device); vx,vy,_=_extract(enc,by['val'],c,device); mean=tx.mean(0,keepdim=True); std=tx.std(0,keepdim=True,unbiased=True).clamp_min(1e-6); feats={'train':((tx-mean)/std,ty),'val':((vx-mean)/std,vy)}; pc=c['downstream']['probe']; board=[]; best=None
    for lr in pc['learning_rates']:
        for wd in pc['weight_decays']:
            tr=_fit_probe(feats,learning_rate=float(lr),weight_decay=float(wd),maximum_epochs=int(pc['maximum_epochs']),patience=int(pc['early_stopping_patience']),seed=int(c['seed'])); board.append({k:v for k,v in tr.items() if k not in ('state','history')})
            if best is None or tr['val_macro_f1']>best['val_macro_f1']: best=tr
    write_csv(board,out/'probe_leaderboard.csv'); write_csv(best['history'],out/'selected_probe_metrics.csv'); torch.save({'classifier':{k:v.cpu() for k,v in best['state'].items()},'feature_mean':mean,'feature_std':std,'selection':{'learning_rate':best['learning_rate'],'weight_decay':best['weight_decay'],'best_epoch':best['best_epoch'],'validation_macro_f1':best['val_macro_f1'],'test_used':False}},out/'best_linear_probe.pt'); atomic_json_dump({'learning_rate':best['learning_rate'],'weight_decay':best['weight_decay'],'best_epoch':best['best_epoch'],'validation_macro_f1':best['val_macro_f1'],'test_used':False},out/'probe_selection.json')
    # Exclusive marker is created before any test image is decoded.
    marker={'encoder_epoch':encoder_epoch,'probe_selected':True}
    if encoder_metadata is not None: marker['encoder_metadata']=dict(encoder_metadata)
    with (out/'test_started.json').open('x') as f: json.dump(marker,f)
    x,y,keys=_extract(enc,by['test'],c,device); state=torch.load(out/'best_linear_probe.pt',map_location='cpu',weights_only=False); clf=torch.nn.Linear(tx.shape[1],19).to(device).eval(); clf.load_state_dict(state['classifier']); loss,metrics,preds=_evaluate(clf,(x-state['feature_mean'])/state['feature_std'],y,device); np.savez_compressed(out/'test_predictions.npz',labels=y.numpy(),predictions=preds,keys=keys)
    report=classification_report(y.numpy(),preds,output_dict=True,zero_division=0); write_csv([{'class_id':i,**report[str(i)]} for i in range(19)],out/'test_per_class_metrics.csv'); np.savetxt(out/'test_confusion_matrix.csv',confusion_matrix(y.numpy(),preds,labels=list(range(19))),delimiter=',',fmt='%d')
    result={'method':c['method']['name'],'encoder_epoch':encoder_epoch,'feature_dim':int(tx.shape[1]),'probe_validation_macro_f1':float(best['val_macro_f1']),'test_loss':float(loss),'test':metrics,'test_images':int(y.numel()),'test_evaluated_once':True}
    if encoder_metadata is not None: result['encoder_metadata']=dict(encoder_metadata)
    atomic_json_dump(result,out/'test_metrics.json'); return result

def run_downstream(c:dict[str,Any],checkpoint:Path,out:Path):
    """Run the existing standard downstream protocol from an SSL checkpoint."""
    from pannuke_ssl.ssl_methods.registry import build_method
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    device=torch.device('cuda'); method=build_method(c,device); ck=load_checkpoint(method,Path(checkpoint),device)
    return _run_downstream(c,method.encoder,out,encoder_epoch=int(ck['epoch']))

def run_frozen_downstream(
    c:dict[str,Any],
    encoder:torch.nn.Module,
    out:Path,
    *,
    encoder_metadata:Mapping[str,Any],
):
    """Apply the standard train/validation-selected probe to a frozen encoder.

    The encoder must already be an explicitly defined pretrained representation;
    this function performs no model or representation selection and preserves the
    framework's exclusive test marker semantics.
    """
    trainable=[name for name,param in encoder.named_parameters() if param.requires_grad]
    if trainable:
        raise AssertionError(f'Frozen downstream encoder has trainable parameters: {trainable[:5]}')
    if encoder.training:
        raise AssertionError('Frozen downstream encoder must be in evaluation mode')
    metadata=dict(encoder_metadata)
    metadata['parameters_requires_grad']=0
    return _run_downstream(c,encoder,out,encoder_epoch='pretrained',encoder_metadata=metadata)
