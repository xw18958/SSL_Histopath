"""Four single-GPU workers: three fixed-budget SSL runs and a shared downstream queue.

Nothing trains unless the explicit ``start`` action is used. A machine-local JSON
file keeps storage and Python paths out of the scientific campaign definition.
"""
from __future__ import annotations
import argparse,base64,fcntl,json,shlex,subprocess,time
from pathlib import Path
import yaml
from ssl_campaign_job import atomic,digest

ROOT=Path(__file__).resolve().parents[1]
SSH=['ssh','-n','-oBatchMode=yes','-oConnectTimeout=15','-oServerAliveInterval=30','-oServerAliveCountMax=3']
TRANSFER_SSH=[part for part in SSH if part!='-n']

def settings(path):
    machines=json.loads(path.read_text());p=yaml.safe_load((ROOT/'configs/ssl_standard/simplex_k_ablation.yaml').read_text())
    workers=machines['workers']
    if set(workers)!=set(p['allocation']):raise ValueError('Exactly the four declared GPU workers are required')
    for name,w in workers.items():
        allocation=p['allocation'][name]
        if (w['host'],w['gpu'],w.get('pretrain_k'))!=(allocation['host'],allocation['gpu'],allocation.get('pretrain_k')):raise ValueError(f'Allocation mismatch: {name}')
    if p['components']!=[8,16,32] or p['checkpoint_epochs']!=[100,150,200,250] or p['stop_epoch']!=250:raise ValueError('Campaign changed')
    return machines,p

def env(w):
    return {'CUDA_VISIBLE_DEVICES':str(w['gpu']),'SSL_PROJECT_ROOT':w['project'],'SSL_DATA_ROOT':w['data'],
            'SSL_MODEL_ROOT':w['models'],'SSL_RUN_ROOT':w['run_root'],'SSL_ABLATION_RUN_ROOT':w['run_root'],
            'SSL_ABLATION_SCRATCH':w['scratch'],'SSL_WORKER_NAME':w['name'],
            'PYTHONPATH':w['project']+'/src'+(':'+w['extra_pythonpath'] if w.get('extra_pythonpath') else ''),
            'OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'}

def command(w,args):
    return 'cd '+shlex.quote(w['project'])+' && '+shlex.join(['env',*(k+'='+v for k,v in env(w).items()),w['python'],*args])

def call(w,args,*,timeout=120):
    cmd=command(w,args)
    cp=subprocess.run(['bash','-c',cmd] if w.get('local') else SSH+['xwan0900@'+w['host'],cmd],text=True,capture_output=True,timeout=timeout)
    if cp.returncode:raise RuntimeError(f'{w["name"]}: {cp.stderr[-2000:]} {cp.stdout[-2000:]}')
    # Scientific commands may print progress first; their last line is JSON.
    return json.loads(cp.stdout.strip().splitlines()[-1])

def job(w,action,job_root=None,spec=None):
    args=['scripts/ssl_campaign_job.py',action]
    if job_root:args+=['--root',str(job_root)]
    if spec:args+=['--spec-base64',base64.b64encode(json.dumps(spec).encode()).decode()]
    return call(w,args)

def launch(w,action,k,epoch=None):
    name=f'{action}_k{k}'+(f'_e{epoch}' if epoch else '')
    args=[w['python'],'-u','scripts/run_simplex_k_ablation.py',action,'--k',str(k)]
    if epoch:args+=['--epoch',str(epoch)]
    spec={'project':w['project'],'command':args,'env':env(w),'job_root':w['scratch']+'/jobs/'+name}
    job(w,'launch',spec=spec)
    return {'root':spec['job_root'],'action':action,'k':k,'epoch':epoch}

def bundle_queue(p):
    return [(k,e) for e in p['checkpoint_epochs'] for k in p['components']]

def audit(w):
    return call(w,['scripts/ssl_campaign_readiness.py','--report',w['scratch']+'/readiness.json'])

def check(m,p):
    reports={name:audit(w) for name,w in m['workers'].items()}
    if any(r['status']!='PASS' for r in reports.values()):raise RuntimeError('A worker is not ready')
    hashes={r['source_sha256'] for r in reports.values()};inputs={r['input_ledger_sha256'] for r in reports.values()}
    if len(hashes)!=1 or len(inputs)!=1:raise RuntimeError('Workers have different code or input inventories')
    if len({json.dumps(r['model_files'],sort_keys=True) for r in reports.values()})!=1:raise RuntimeError('PLIP assets differ across workers')
    # Keep each full build string in its audit; compare the Python release
    # across hosts, since Anaconda and uv package the same release differently.
    if len({(r['python_version'].split()[0],r['torch_version']) for r in reports.values()})!=1:raise RuntimeError('Python/PyTorch releases differ')
    for name,r in reports.items():
        s=r['smoke']
        if r['worker']!=name or s['k']!=m['workers'][name].get('pretrain_k',8) or s['ssl_steps']!=2 or s['ssl_batch_size']!=128 or s['initial_encoder_sha256']!=p['initial_encoder_sha256'] or s['test_touched']:
            raise RuntimeError(f'Incorrect smoke identity: {name}')
    projectors={r['smoke']['initial_projector_sha256'] for r in reports.values()}
    if len(projectors)!=1:raise RuntimeError('Initial projectors differ')
    downstream=reports['gpu2-358-g0']['smoke']
    if {r['dataset'] for r in downstream['classification']}!=set(p['classification_datasets']) or {r['dataset'] for r in downstream['retrieval']}!=set(p['retrieval_datasets']):raise RuntimeError('Full downstream smoke coverage missing')
    return {'status':'PASS','workers':reports,'training_started':False,'ssl_runs':3,'checkpoint_bundles':12,'evaluations':216,'classification':180,'retrieval':36}

def pull(w,relative,destination):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    source=w['run_root']+'/'+relative
    if w.get('local'):
        if Path(source).resolve()!=destination.resolve():
            subprocess.run(['rsync','-a','--checksum',source,str(destination)],check=True)
    else:subprocess.run(['rsync','-a','--checksum','-e',shlex.join(TRANSFER_SSH),'xwan0900@'+w['host']+':'+source,str(destination)],check=True)

def push_file(w,source,relative):
    target=w['run_root']+'/'+relative
    if w.get('local') and Path(source).resolve()==Path(target).resolve():return
    call(w,['scripts/ssl_campaign_readiness.py','--mkdir',str(Path(target).parent)])
    temp=target+'.transfer'
    destination=temp if w.get('local') else 'xwan0900@'+w['host']+':'+temp
    subprocess.run(['rsync','-a','--checksum','-e',shlex.join(TRANSFER_SSH),str(source),destination],check=True)
    call(w,['scripts/ssl_campaign_readiness.py','--install-checkpoint',temp,'--destination',target,'--sha256',digest(Path(source))])

def collect(w,k,e,master,p):
    for task,ds in [('downstream_datasets',p['classification_datasets']),('image_text_retrieval',p['retrieval_datasets'])]:
        for d in ds:
            rel=f'k{k}/{task}/{d}/epoch_{e}/';pull(w,rel,master/rel)
    rel=f'k{k}/completion/epoch_{e}.json';pull(w,rel,master/rel)
    done=json.loads((master/rel).read_text())
    if (done['k'],done['epoch'],done['evaluations'])!=(k,e,18):raise RuntimeError('Incomplete downstream bundle')

def verify_complete(master,p):
    classifications=[];retrievals=[]
    for k,e in bundle_queue(p):
        done=json.loads((master/f'k{k}/completion/epoch_{e}.json').read_text())
        ck=digest(master/f'k{k}/pretrain_full/checkpoints/epoch_{e}.pt')
        if done['checkpoint_sha256']!=ck:raise RuntimeError('Completion checkpoint mismatch')
        for d in p['classification_datasets']:
            r=json.loads((master/f'k{k}/downstream_datasets/{d}/epoch_{e}/test_metrics.json').read_text())
            if r['encoder_metadata']['checkpoint_sha256']!=ck or not r['test_evaluated_once']:raise RuntimeError('Classification mismatch')
            classifications.append({'k':k,'epoch':e,'dataset':d,**r['test']})
        for d in p['retrieval_datasets']:
            r=json.loads((master/f'k{k}/image_text_retrieval/{d}/epoch_{e}/test_retrieval_metrics.json').read_text())
            if r['checkpoint_sha256']!=ck:raise RuntimeError('Retrieval mismatch')
            retrievals.append({'k':k,'epoch':e,'dataset':d,**r['test']})
    if (len(classifications),len(retrievals))!=(180,36):raise RuntimeError('Missing results')
    atomic({'classification':classifications,'retrieval':retrievals},master/'all_results.json')
    import csv
    for name,rows in [('classification_results',classifications),('retrieval_results',retrievals)]:
        with (master/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def start(m,p):
    master=Path(m['master_root']);master.mkdir(parents=True,exist_ok=True)
    with (master/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (master/'campaign_started.json').exists():raise RuntimeError('Campaign already started; preserve outputs and diagnose before any restart')
        ready=check(m,p);atomic(ready,master/'launch_readiness.json')
        with (master/'campaign_started.json').open('x') as f:json.dump({'started_at':time.time(),'plan':p},f,indent=2)
        workers=m['workers'];running={};pretrain={w['pretrain_k']:w for w in workers.values() if w.get('pretrain_k')}
        completed=[];available=set();backed=set();pretrain_finished=set()
        try:
            for name,w in workers.items():
                if w.get('pretrain_k'):running[name]=launch(w,'pretrain',w['pretrain_k'])
            while len(completed)!=12 or len(pretrain_finished)!=3:
                for name,j in list(running.items()):
                    w=workers[name];s=job(w,'status',j['root'])
                    if s['state'] in ('failed','lost','missing'):raise RuntimeError(f'{name} {j}: {s}')
                    if s['state']=='complete':
                        if j['action']=='pretrain':
                            k=j['k'];summary=call(w,['scripts/ssl_campaign_readiness.py','--json-file',w['run_root']+f'/k{k}/pretrain_full/run_summary.json'])
                            if summary['epochs_completed']!=250 or summary['saved_checkpoint_epochs']!=[100,150,200,250]:raise RuntimeError('Wrong pretraining budget')
                            for f in ['run_summary.json','resolved_config.json','pretrain_metrics.csv']:pull(w,f'k{k}/pretrain_full/{f}',master/f'k{k}/pretrain_full/{f}')
                            pretrain_finished.add(k)
                        else:
                            collect(w,j['k'],j['epoch'],master,p);completed.append((j['k'],j['epoch']))
                        del running[name]
                for k,e in bundle_queue(p):
                    if (k,e) in backed:continue
                    source=pretrain[k];rel=f'k{k}/pretrain_full/checkpoints/epoch_{e}.pt'
                    info=call(source,['scripts/ssl_campaign_job.py','inspect','--file',source['run_root']+'/'+rel])
                    if info['exists']:
                        local=master/rel;pull(source,rel,local)
                        if digest(local)!=info['sha256']:raise RuntimeError('Checkpoint transfer mismatch')
                        backed.add((k,e));available.add((k,e))
                for name,w in workers.items():
                    if name in running:continue
                    tasks=[x for x in bundle_queue(p) if x in available and x not in completed and not any((j['k'],j['epoch'])==x for j in running.values() if j['action']=='downstream')]
                    if tasks:
                        k,e=tasks[0];rel=f'k{k}/pretrain_full/checkpoints/epoch_{e}.pt';push_file(w,master/rel,rel)
                        running[name]=launch(w,'downstream',k,e)
                atomic({'state':'running','pretraining_finished':sorted(pretrain_finished),'checkpoint_bundles_backed_up':len(backed),'completed_bundles':len(completed),'completed_evaluations':18*len(completed),'total_evaluations':216,'running':running,'updated_at':time.time()},master/'status.json')
                time.sleep(30)
            verify_complete(master,p);atomic({'state':'complete','ssl_runs':3,'checkpoint_bundles':12,'evaluations':216,'classification':180,'retrieval':36,'finished_at':time.time()},master/'status.json')
        except Exception as e:
            atomic({'state':'failed','error':str(e),'running':running,'completed_bundles':completed,'updated_at':time.time()},master/'status.json')
            # Detached jobs already running are preserved; no automatic reruns.
            raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','check','launch','start','status']);parser.add_argument('--machines',type=Path,required=True);a=parser.parse_args();m,p=settings(a.machines)
    if a.action=='prepare':
        for w in m['workers'].values():
            for k in p['components']:call(w,['scripts/run_simplex_k_ablation.py','prepare','--k',str(k)])
        r={'status':'PREPARED','training_started':False,'ssl_runs':3,'evaluations':216}
    elif a.action=='check':r=check(m,p);atomic(r,Path(m['master_root'])/'ready_to_start.json')
    elif a.action=='status':
        file=Path(m['master_root'])/'status.json';r=json.loads(file.read_text()) if file.exists() else {'state':'not_started'}
    elif a.action=='launch':
        master=Path(m['master_root'])
        if (master/'campaign_started.json').exists():raise RuntimeError('Campaign already started')
        check(m,p)
        with (master/'launch_requested.json').open('x') as f:json.dump({'requested_at':time.time()},f)
        controller=m['workers']['jinman-g0']
        with (master/'coordinator.log').open('ab') as log:
            child=subprocess.Popen([controller['python'],'-u',str(Path(__file__).resolve()),'start','--machines',str(a.machines.resolve())],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        r={'state':'launch_requested','pid':child.pid,'log':str(master/'coordinator.log')};atomic(r,master/'launcher.json')
    else:start(m,p);r={'state':'complete'}
    print(json.dumps(r),flush=True)

if __name__=='__main__':main()
