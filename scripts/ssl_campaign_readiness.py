"""Record verified inputs and refuse stale readiness after code or input changes."""
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path
from ssl_campaign_inputs import sha,verify
from ssl_campaign_job import atomic

ROOT=Path(__file__).resolve().parents[1]

def source():
    files={}
    for folder in ('src','scripts','configs','manifests'):
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.yaml','.json','.sha256'):
                files[p.relative_to(ROOT).as_posix()]=sha(p)
    files['pannuke19_metadata.csv']=sha(ROOT/'pannuke19_metadata.csv')
    return files

def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def data_stats(ledger,data):
    stats=[]
    for row in ledger['files']:
        path=(data/row['path']);s=path.stat()
        if s.st_size!=row['bytes']:raise RuntimeError('Input size changed: '+row['path'])
        stats.append([row['path'],s.st_size,s.st_mtime_ns])
    return fingerprint(stats)

def models():
    root=Path(os.environ['SSL_MODEL_ROOT'])/'plip_model'
    return {name:sha(root/name) for name in ('config.json','model.safetensors','tokenizer.json','tokenizer_config.json')}

def record(k,report):
    import torch
    from pannuke_ssl.ssl_framework.k_ablation import preflight
    scratch=Path(os.environ['SSL_ABLATION_SCRATCH']);ledger=json.loads((scratch/'inputs.json').read_text());data=Path(os.environ['SSL_DATA_ROOT'])
    checked=verify(ledger,ROOT,data);flight=preflight(k)
    smoke=json.loads((scratch/'smoke'/os.environ['SSL_WORKER_NAME']/'report.json').read_text())
    if smoke['status']!='PASS' or smoke['test_touched'] or smoke['k']!=k:raise RuntimeError('GPU smoke did not pass for this K')
    current=source();r={'status':'PASS','worker':os.environ['SSL_WORKER_NAME'],'source_sha256':fingerprint(current),'source_files':current,
        'input_ledger_sha256':fingerprint(ledger),'input_verification':checked,'data_stat_sha256':data_stats(ledger,data),
        'model_files':models(),'smoke':smoke,'preflight':flight,'python_version':sys.version,'torch_version':torch.__version__,'training_started':False}
    atomic(r,report);return r

def check(report):
    r=json.loads(report.read_text());scratch=Path(os.environ['SSL_ABLATION_SCRATCH']);ledger=json.loads((scratch/'inputs.json').read_text())
    if r['status']!='PASS' or r['source_sha256']!=fingerprint(source()):raise RuntimeError('Verified source changed; repeat smoke tests')
    if r['input_ledger_sha256']!=fingerprint(ledger):raise RuntimeError('Input inventory changed')
    if r['data_stat_sha256']!=data_stats(ledger,Path(os.environ['SSL_DATA_ROOT'])):raise RuntimeError('Data changed; repeat byte verification')
    if r['model_files']!=models():raise RuntimeError('Model assets changed')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Exactly one CUDA GPU must be visible per worker')
    if sys.version!=r['python_version'] or torch.__version__!=r['torch_version']:raise RuntimeError('Runtime changed')
    free,total=torch.cuda.mem_get_info()
    minimum_gib=20 if os.environ.get('SSL_WORKER_ROLE','pretrain')=='pretrain' else 16
    if free<minimum_gib*2**30:raise RuntimeError(f'Insufficient free GPU memory: {free/2**30:.1f} GiB; {minimum_gib} GiB required for this worker role')
    r['free_gpu_memory_gib']=free/2**30
    # Leave the detailed per-file map in the saved report, not every console update.
    r.pop('source_files');return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path);p.add_argument('--record',type=int,choices=[8,16,32]);p.add_argument('--mkdir',type=Path);p.add_argument('--json-file',type=Path);p.add_argument('--install-checkpoint',type=Path);p.add_argument('--destination',type=Path);p.add_argument('--sha256');a=p.parse_args()
    if a.mkdir:a.mkdir.mkdir(parents=True,exist_ok=True);r={'status':'PASS'}
    elif a.json_file:r=json.loads(a.json_file.read_text())
    elif a.install_checkpoint:
        if sha(a.install_checkpoint)!=a.sha256:raise RuntimeError('Transferred checkpoint checksum mismatch')
        if a.destination.exists() and sha(a.destination)!=a.sha256:raise RuntimeError('Refuse overwrite of a different checkpoint')
        a.install_checkpoint.replace(a.destination);r={'status':'PASS','sha256':a.sha256}
    elif a.record:r=record(a.record,a.report)
    else:r=check(a.report)
    print(json.dumps(r),flush=True)

if __name__=='__main__':main()
