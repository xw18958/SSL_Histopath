from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping
import numpy as np,torch
from sklearn.metrics import classification_report,confusion_matrix
from torch.utils.data import DataLoader
from pannuke_ssl.data import PanNukeImageDataset,loader_kwargs
from pannuke_ssl.parquet import build_source_index,preload_images,verify_records
from .data_validation import validated_pannuke_split
from pannuke_ssl.probe import _evaluate,_fit_probe
from pannuke_ssl.utils import atomic_json_dump,seed_everything,write_csv
from .external_datasets import (
    EXTERNAL_DATASETS,
    PANNUKE_DATASET,
    ExternalProbeDataset,
    ExternalProbeImageDataset,
    load_external_manifest,
)
from .trainer import load_checkpoint

@torch.inference_mode()
def _extract(encoder,rows,c,device,external_dataset:ExternalProbeDataset|None=None):
    if external_dataset is None:
        index=build_source_index(Path(c['data']['root'])); verify_records(rows,index); cache=preload_images(rows,index)
        loader=DataLoader(PanNukeImageDataset(rows,index,cache,include_label=True,include_key=True),**loader_kwargs(int(c['downstream']['batch_size']),int(c['downstream']['num_workers']),shuffle=False))
        batches=((images,y,(fold,idx)) for images,y,fold,idx in loader)
    else:
        loader=DataLoader(ExternalProbeImageDataset(rows,external_dataset.root),**loader_kwargs(int(c['downstream']['batch_size']),int(c['downstream']['num_workers']),shuffle=False))
        batches=((images,y,(record_index,)) for images,y,record_index in loader)
    xs,ys,keys=[],[],[]
    for images,y,key_tensors in batches:
        x=images.to(device,dtype=torch.float32,non_blocking=True).div_(255.)
        with torch.autocast('cuda',dtype=torch.bfloat16): features=encoder(x)
        # Standard SSL encoders expose final patch tokens [B,T,D].  Frozen
        # external adapters may expose their pre-registered [B,D] readout.
        if features.ndim==3: features=features.mean(1)
        expected_dim=c['downstream'].get('feature_dim')
        if features.ndim!=2 or (expected_dim is not None and features.shape[1]!=int(expected_dim)):
            raise RuntimeError(f'Unexpected probe feature shape {tuple(features.shape)}; expected feature_dim={expected_dim}')
        xs.append(features.float().cpu()); ys.append(y.long().cpu())
        if external_dataset is None:
            fold,idx=key_tensors; keys.extend(zip(fold.tolist(),idx.tolist()))
        else:
            (record_index,)=key_tensors; keys.extend(record_index.tolist())
    return torch.cat(xs),torch.cat(ys),np.asarray(keys,dtype=np.int64)

def _split_rows(c:dict[str,Any],external_dataset:ExternalProbeDataset|None=None):
    if external_dataset is not None:
        by=external_dataset.split_rows
        if set(by)!={'train','val','test'} or any(not rows for rows in by.values()):
            raise AssertionError(f'External manifest has invalid split coverage: {external_dataset.split_counts}')
        return by
    rows,_=validated_pannuke_split(c)
    return {s:[r for r in rows if r['split']==s] for s in ('train','val','test')}

def _run_downstream(c:dict[str,Any],encoder,out:Path,*,encoder_epoch:int|str,encoder_metadata:Mapping[str,Any]|None=None,external_dataset:ExternalProbeDataset|None=None):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if (out/'test_started.json').exists(): raise FileExistsError('Test already started')
    seed_everything(int(c['seed'])); device=torch.device('cuda'); enc=encoder.to(device).eval(); [p.requires_grad_(False) for p in enc.parameters()]
    by=_split_rows(c,external_dataset)
    dataset_slug=PANNUKE_DATASET if external_dataset is None else external_dataset.slug
    class_names=tuple(str(index) for index in range(19)) if external_dataset is None else external_dataset.class_names
    num_classes=len(class_names)
    dataset_metadata=None if external_dataset is None else external_dataset.metadata()
    if dataset_metadata is not None: atomic_json_dump(dataset_metadata,out/'dataset_provenance.json')
    # Only train/val are decoded before probe selection.
    tx,ty,_=_extract(enc,by['train'],c,device,external_dataset); vx,vy,_=_extract(enc,by['val'],c,device,external_dataset); mean=tx.mean(0,keepdim=True); std=tx.std(0,keepdim=True,unbiased=True).clamp_min(1e-6); feats={'train':((tx-mean)/std,ty),'val':((vx-mean)/std,vy)}; pc=c['downstream']['probe']; board=[]; best=None
    for lr in pc['learning_rates']:
        for wd in pc['weight_decays']:
            tr=_fit_probe(feats,learning_rate=float(lr),weight_decay=float(wd),maximum_epochs=int(pc['maximum_epochs']),patience=int(pc['early_stopping_patience']),seed=int(c['seed']),num_classes=num_classes); board.append({k:v for k,v in tr.items() if k not in ('state','history')})
            if best is None or tr['val_macro_f1']>best['val_macro_f1']: best=tr
    write_csv(board,out/'probe_leaderboard.csv'); write_csv(best['history'],out/'selected_probe_metrics.csv'); torch.save({'classifier':{k:v.cpu() for k,v in best['state'].items()},'feature_mean':mean,'feature_std':std,'selection':{'learning_rate':best['learning_rate'],'weight_decay':best['weight_decay'],'best_epoch':best['best_epoch'],'validation_macro_f1':best['val_macro_f1'],'num_classes':num_classes,'test_used':False}},out/'best_linear_probe.pt'); atomic_json_dump({'learning_rate':best['learning_rate'],'weight_decay':best['weight_decay'],'best_epoch':best['best_epoch'],'validation_macro_f1':best['val_macro_f1'],'num_classes':num_classes,'test_used':False},out/'probe_selection.json')
    # Exclusive marker is created before any test image is decoded.
    marker={'encoder_epoch':encoder_epoch,'probe_selected':True,'dataset':dataset_slug,'num_classes':num_classes}
    if encoder_metadata is not None: marker['encoder_metadata']=dict(encoder_metadata)
    with (out/'test_started.json').open('x') as f: json.dump(marker,f)
    x,y,keys=_extract(enc,by['test'],c,device,external_dataset); state=torch.load(out/'best_linear_probe.pt',map_location='cpu',weights_only=False); clf=torch.nn.Linear(tx.shape[1],num_classes).to(device).eval(); clf.load_state_dict(state['classifier']); loss,metrics,preds=_evaluate(clf,(x-state['feature_mean'])/state['feature_std'],y,device); np.savez_compressed(out/'test_predictions.npz',labels=y.numpy(),predictions=preds,keys=keys)
    labels=list(range(num_classes)); report=classification_report(y.numpy(),preds,labels=labels,target_names=list(class_names),output_dict=True,zero_division=0); write_csv([{'class_id':i,'class_name':class_names[i],**report[class_names[i]]} for i in labels],out/'test_per_class_metrics.csv'); np.savetxt(out/'test_confusion_matrix.csv',confusion_matrix(y.numpy(),preds,labels=labels),delimiter=',',fmt='%d')
    if external_dataset is not None:
        lookup={int(row['record_index']):row for row in by['test']}
        write_csv([{'record_index':int(record_index),'relative_path':lookup[int(record_index)]['relative_path'],'class_id':int(label),'class_name':class_names[int(label)],'prediction_id':int(prediction),'prediction_name':class_names[int(prediction)]} for record_index,label,prediction in zip(keys,y.numpy(),preds)],out/'test_prediction_records.csv')
    result={'method':c['method']['name'],'dataset':dataset_slug,'encoder_epoch':encoder_epoch,'feature_dim':int(tx.shape[1]),'num_classes':num_classes,'class_names':list(class_names),'probe_validation_macro_f1':float(best['val_macro_f1']),'test_loss':float(loss),'test':metrics,'test_images':int(y.numel()),'test_evaluated_once':True}
    if dataset_metadata is not None: result['dataset_metadata']=dataset_metadata
    if encoder_metadata is not None: result['encoder_metadata']=dict(encoder_metadata)
    atomic_json_dump(result,out/'test_metrics.json'); return result

def _optional_dataset(dataset:str,c:dict[str,Any])->ExternalProbeDataset|None:
    if dataset==PANNUKE_DATASET: return None
    if dataset not in EXTERNAL_DATASETS: raise ValueError(f'Unknown downstream dataset {dataset!r}')
    return load_external_manifest(dataset,Path(c['output']['root']))

def run_downstream(c:dict[str,Any],checkpoint:Path,out:Path,*,dataset:str=PANNUKE_DATASET):
    """Run the existing standard downstream protocol from an SSL checkpoint."""
    from pannuke_ssl.ssl_methods.registry import build_method
    external_dataset=_optional_dataset(dataset,c)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    device=torch.device('cuda'); method=build_method(c,device); ck=load_checkpoint(method,Path(checkpoint),device)
    return _run_downstream(c,method.encoder,out,encoder_epoch=int(ck['epoch']),external_dataset=external_dataset)

def run_frozen_downstream(
    c:dict[str,Any],
    encoder:torch.nn.Module,
    out:Path,
    *,
    encoder_metadata:Mapping[str,Any],
    dataset:str=PANNUKE_DATASET,
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
    return _run_downstream(c,encoder,out,encoder_epoch='pretrained',encoder_metadata=metadata,external_dataset=_optional_dataset(dataset,c))
