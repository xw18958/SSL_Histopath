"""Small detached-job helper. Job IDs and scientific output directories are exclusive."""
from __future__ import annotations
import argparse,base64,hashlib,json,os,subprocess,sys,time
from pathlib import Path

def atomic(value,path):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def status(root):
    if (root/'return.json').exists():return json.loads((root/'return.json').read_text())
    if not (root/'started.json').exists():return {'state':'missing'}
    s=json.loads((root/'started.json').read_text())
    try:os.kill(s['pid'],0)
    except ProcessLookupError:return {'state':'lost','pid':s['pid'],'log':str(root/'job.log')}
    return {'state':'running','pid':s['pid'],'log':str(root/'job.log')}

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','run','status','inspect']);p.add_argument('--spec-base64');p.add_argument('--root',type=Path);p.add_argument('--file',type=Path)
    a=p.parse_args()
    if a.action=='inspect':
        r={'exists':a.file.is_file()}
        if r['exists']:r.update(bytes=a.file.stat().st_size,sha256=digest(a.file))
    elif a.action=='launch':
        spec=json.loads(base64.b64decode(a.spec_base64));root=Path(spec['job_root']);root.mkdir(parents=True,exist_ok=False)
        (root/'spec.json').write_text(json.dumps(spec,indent=2)+'\n')
        with (root/'runner.log').open('ab') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run','--root',str(root)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        r={'state':'launched','pid':child.pid,'root':str(root)}
    elif a.action=='status':r=status(a.root)
    else:
        spec=json.loads((a.root/'spec.json').read_text());atomic({'pid':os.getpid(),'started_at':time.time()},a.root/'started.json')
        try:
            env=dict(os.environ);env.update(spec['env'])
            with (a.root/'job.log').open('ab') as log:
                result=subprocess.run(spec['command'],cwd=spec['project'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            r={'state':'complete' if result.returncode==0 else 'failed','returncode':result.returncode,'finished_at':time.time(),'log':str(a.root/'job.log')}
        except Exception as e:r={'state':'failed','error':str(e),'finished_at':time.time(),'log':str(a.root/'job.log')}
        atomic(r,a.root/'return.json')
    print(json.dumps(r),flush=True)

if __name__=='__main__':main()
