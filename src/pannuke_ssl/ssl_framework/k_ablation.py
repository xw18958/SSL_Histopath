"""Predeclared K-only campaign; existing scientific fit/evaluation routines are reused."""
from __future__ import annotations
import itertools,json,math,os
from pathlib import Path
import torch
from pannuke_ssl.config import load_yaml
from pannuke_ssl.utils import atomic_json_dump,seed_everything
from .config import load_standard_config
from .external_datasets import EXTERNAL_DATASETS,load_external_manifest
from .image_text_datasets import IMAGE_TEXT_DATASETS,load_image_text_manifest
from .data_validation import module_sha,build_ssl_loader
from .downstream import _run_downstream,_extract,_tune_probe_sequential_greedy
from .retrieval_protocol import file_sha256,RETRIEVAL_VERSION
from .retrieval_repair import run_smoke,run_train_val,run_test
from .trainer import train_ssl,save_checkpoint,load_checkpoint,_set_schedule
from pannuke_ssl.ssl_methods.registry import build_method
from pannuke_ssl.probe import _fit_probe

ROOT=Path(__file__).resolve().parents[3]

def plan():
    p=load_yaml(ROOT/'configs/ssl_standard/simplex_k_ablation.yaml')
    if p['components']!=[8,16,32] or p['checkpoint_epochs']!=[100,150,200,250] or p['stop_epoch']!=250:
        raise ValueError('Campaign must contain three K values and four checkpoints stopping at 250')
    if p['schedule_epochs']!=300 or p['ssl_batch_size']!=128 or p['sigma']!=1.0 or p['ssl_learning_rate']!=.0005:
        raise ValueError('Frozen SSL comparison settings changed')
    if tuple(p['classification_datasets'])!=EXTERNAL_DATASETS or tuple(p['retrieval_datasets'])!=IMAGE_TEXT_DATASETS:
        raise ValueError('Campaign must cover all 15 classification and three retrieval datasets')
    if p['seed']!=20260903:raise ValueError('Original initialization seed changed')
    return p

def config(k):
    p=plan()
    if k not in p['components']:raise ValueError(f'Unplanned K={k}')
    c=load_standard_config('simplex_sigreg_lejepa')
    c['training'].update(max_epochs=p['stop_epoch'],schedule_epochs=p['schedule_epochs'],checkpoint_epochs=list(p['checkpoint_epochs']))
    c['method']['objective']['simplex_components']=k
    c['method']['optimizer']['peak_lr']=p['ssl_learning_rate']
    c['method']['source_metadata'].update(simplex_components_tuned=False,component_selection='predeclared_K_only_ablation')
    c['campaign']={'name':p['campaign'],'k':k,'expected_initial_encoder_sha256':p['initial_encoder_sha256']}
    return c

def matrix():
    p=plan()
    return [{'k':k,'epoch':e,'task':task,'dataset':d} for k in p['components'] for e in p['checkpoint_epochs']
            for task,ds in [('classification',p['classification_datasets']),('retrieval',p['retrieval_datasets'])] for d in ds]

def run_root(k):
    return Path(os.environ['SSL_ABLATION_RUN_ROOT'])/f'k{k}'

def preflight(k):
    c=config(k);mroot=Path(c['manifests']['root']);p=plan()
    from .data_validation import validated_pannuke_split
    rows,index=validated_pannuke_split(c)
    assert len(rows)==len(index)==7901
    manifests={}
    for d in p['classification_datasets']:
        m=load_external_manifest(d,mroot);manifests[d]=m.manifest_sha256
    for d in p['retrieval_datasets']:
        m=load_image_text_manifest(d,mroot);assert m['split_counts']=={'train':3267,'val':700,'test':700};manifests[d]=m['manifest_sha256']
    return {'status':'PASS','k':k,'pannuke_images':7901,'manifest_hashes':manifests,'evaluations':len(matrix()),'test_decoded':False}

def prepare(k):
    c=config(k);root=run_root(k);root.mkdir(parents=True,exist_ok=True)
    path=root/'campaign_config.json'
    if path.exists() and json.loads(path.read_text())!=c:raise RuntimeError('Existing campaign configuration differs')
    atomic_json_dump(c,path);atomic_json_dump({'plan':plan(),'evaluations':matrix()},root.parent/'campaign_plan.json')
    return {'status':'PREPARED','k':k,'root':str(root),'training_started':False}

def pretrain(k):
    c=config(k)
    seed_everything(int(c['seed']));m=build_method(c,torch.device('cpu'))
    actual=module_sha(m.encoder);del m
    if actual!=plan()['initial_encoder_sha256']:raise RuntimeError('Original initialization mismatch')
    return train_ssl(c,run_root(k)/'pretrain_full',use_all_data=True,validate=False,early_stop=False,
                     checkpoint_epochs=c['training']['checkpoint_epochs'],schedule_epochs=c['training']['schedule_epochs'])

def evaluate(k,epoch):
    p=plan()
    if epoch not in p['checkpoint_epochs']:raise ValueError('Unplanned checkpoint')
    c=config(k);root=run_root(k);checkpoint=root/'pretrain_full/checkpoints'/f'epoch_{epoch}.pt'
    checkpoint_hash=file_sha256(checkpoint);metadata={'k':k,'checkpoint_sha256':checkpoint_hash,'schedule_epochs':p['schedule_epochs']}
    pending=[];results=[]
    for d in p['classification_datasets']:
        out=root/'downstream_datasets'/d/f'epoch_{epoch}';done=out/'test_metrics.json'
        if done.exists():
            r=json.loads(done.read_text())
            if r.get('encoder_metadata')!=metadata or r['encoder_epoch']!=epoch:raise RuntimeError('Completed classification identity mismatch')
            results.append(r)
        else:pending.append((d,out))
    if pending:
        seed_everything(int(c['seed']));torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        method=build_method(c,torch.device('cuda'));ck=load_checkpoint(method,checkpoint,torch.device('cuda'))
        if ck['epoch']!=epoch or ck['config']['method']['objective']['simplex_components']!=k:raise RuntimeError('Checkpoint K/epoch mismatch')
        before=module_sha(method.encoder)
        for d,out in pending:
            results.append(_run_downstream(c,method.encoder,out,encoder_epoch=epoch,encoder_metadata=metadata,
                                          external_dataset=load_external_manifest(d,Path(c['manifests']['root']))))
        if module_sha(method.encoder)!=before:raise AssertionError('Classification changed frozen encoder weights')
        del method,ck;torch.cuda.empty_cache()
    for d in p['retrieval_datasets']:
        out=root/'image_text_retrieval'/d/f'epoch_{epoch}';done=out/'test_retrieval_metrics.json'
        if done.exists():
            r=json.loads(done.read_text())
            if r['checkpoint_sha256']!=checkpoint_hash or r['encoder_epoch']!=epoch or r['evaluation_protocol_version']!=RETRIEVAL_VERSION:
                raise RuntimeError('Completed retrieval identity mismatch')
        else:
            cache=Path(os.environ.get('SSL_ABLATION_SCRATCH','/tmp/ssl_k_ablation_20261007'))/'feature_work'/f'k{k}'
            run_train_val(c,checkpoint,out,dataset=d,cache_root=cache)
            r=run_test(c,checkpoint,out,dataset=d,cache_root=cache)
        results.append(r)
    if len(results)!=18:raise AssertionError('A checkpoint must produce 18 downstream evaluations')
    atomic_json_dump({'k':k,'epoch':epoch,'evaluations':18,'classification':15,'retrieval':3,'checkpoint_sha256':checkpoint_hash},root/'completion'/f'epoch_{epoch}.json')
    return {'status':'PASS','k':k,'epoch':epoch,'evaluations':18}

def smoke(k,*,all_downstream=False):
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Smoke requires exactly one visible CUDA GPU')
    c=config(k);seed_everything(int(c['seed']));torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=c['training']['tf32'];torch.backends.cudnn.allow_tf32=c['training']['tf32']
    device=torch.device('cuda');method=build_method(c,device)
    initial=module_sha(method.encoder);projector=module_sha(method.model.projector)
    if initial!=plan()['initial_encoder_sha256']:raise AssertionError('Original encoder initialization mismatch')
    loader=build_ssl_loader(c,method,use_all_data=True);opt=method.build_optimizer();losses=[]
    torch.cuda.reset_peak_memory_stats()
    for step,batch in enumerate(itertools.islice(loader,2)):
        assert batch['global_views'].shape[:2]==(128,2) and batch['local_views'].shape[:2]==(128,6)
        opt.zero_grad(set_to_none=True);_set_schedule(method,opt,step,300*len(loader))
        result=method.training_step(batch,bf16=True);assert torch.isfinite(result.loss);result.loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(method.optimizer_parameters(),float('inf'));assert torch.isfinite(norm)
        opt.step();losses.append(float(result.loss.detach()))
    scratch=Path(os.environ.get('SSL_ABLATION_SCRATCH','/tmp/ssl_k_ablation_20261007'))/'smoke'/os.environ.get('SSL_WORKER_NAME','worker')
    scratch.mkdir(parents=True,exist_ok=True);checkpoint=scratch/'smoke_only.pt';save_checkpoint(checkpoint,method,c,2,{'smoke_only':True,'test_used':False})
    report={'status':'PASS','k':k,'initial_encoder_sha256':initial,'initial_projector_sha256':projector,'ssl_batch_size':128,'ssl_steps':2,'ssl_losses':losses,'ssl_peak_memory_gib':torch.cuda.max_memory_allocated()/2**30,'classification':[],'retrieval':[],'test_touched':False}
    del loader,opt
    if all_downstream:
        method.encoder.eval();[param.requires_grad_(False) for param in method.encoder.parameters()];before=module_sha(method.encoder)
        torch.backends.cuda.matmul.allow_tf32=False
        def subset(rows):
            result=[];counts={}
            for row in rows:
                cid=row['class_id']
                if counts.get(cid,0)<2:result.append(row);counts[cid]=counts.get(cid,0)+1
            return result
        for d in plan()['classification_datasets']:
            m=load_external_manifest(d,Path(c['manifests']['root']));by=m.split_rows
            tx,ty,_=_extract(method.encoder,subset(by['train']),c,device,m);vx,vy,_=_extract(method.encoder,subset(by['val']),c,device,m)
            mean=tx.mean(0,keepdim=True);std=tx.std(0,keepdim=True,unbiased=True).clamp_min(1e-6)
            feats={'train':((tx-mean)/std,ty),'val':((vx-mean)/std,vy)}
            lr,wd,board,_=_tune_probe_sequential_greedy(feats,c['downstream']['probe'],seed=int(c['seed']),num_classes=m.num_classes)
            fit=_fit_probe(feats,learning_rate=lr,weight_decay=wd,maximum_epochs=2,patience=2,seed=int(c['seed']),num_classes=m.num_classes)
            assert len(board)==5 and math.isfinite(fit['val_macro_f1'])
            report['classification'].append({'dataset':d,'train_images':len(ty),'val_images':len(vy),'status':'PASS'})
        assert module_sha(method.encoder)==before;del method;torch.cuda.empty_cache()
        for d in plan()['retrieval_datasets']:
            r=run_smoke(c,checkpoint,scratch/(d+'_retrieval_smoke.json'),dataset=d)
            assert r['status']=='PASS' and r['test_touched'] is False;report['retrieval'].append({'dataset':d,'status':'PASS'})
    atomic_json_dump(report,scratch/'report.json');return report
