"""Inventory or verify the exact existing campaign inputs, without rebuilding splits."""
from __future__ import annotations
import argparse,concurrent.futures,hashlib,json,os
from pathlib import Path
import yaml

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def relative(root, path):
    # Validate the resolved target, but preserve the path used by the manifest.
    # IPATH/Images is an alias on the source server; staging its target name
    # alone would leave the original manifest path missing on a new server.
    path.resolve().relative_to(root.resolve())
    return Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(root))).as_posix()

def inventory(project,data):
    files=set();manifests={};counts={}
    for registry,subdir in [('external_probe_datasets.yaml','dataset_manifests'),('image_text_retrieval_datasets.yaml','image_text_manifests')]:
        spec=yaml.safe_load((project/'configs/ssl_standard'/registry).read_text())['datasets']
        for slug,c in spec.items():
            mpath=project/'manifests/ssl_standard'/subdir/(slug+'.json');m=json.loads(mpath.read_text())
            expected=Path(str(mpath)+'.sha256').read_text().split()[0]
            if sha(mpath)!=expected:raise ValueError(f'Manifest checksum mismatch: {slug}')
            manifests[relative(project,mpath)]=expected
            root=data/str(c['root']).removeprefix('${SSL_DATA_ROOT}/')
            counts[slug]=len(m['records'])
            for row in m['records']:
                key={'file':'relative_path','parquet':'parquet_file','hdf5':'hdf5_file'}[row.get('storage','file')]
                files.add(relative(data,root/row[key]))
    files.update(relative(data,p) for p in (data/'PanNuke/data').glob('fold*-of-*.parquet'))
    pannuke=[p for p in files if p.startswith('PanNuke/data/')]
    if len(pannuke)!=6:raise ValueError(f'Expected six PanNuke parquet files, found {len(pannuke)}')
    def record(name):
        p=data/name
        return {'path':name,'bytes':p.stat().st_size,'sha256':sha(p)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:rows=list(ex.map(record,sorted(files)))
    return {'schema':1,'files':rows,'bytes':sum(r['bytes'] for r in rows),'dataset_records':counts,
            'manifest_hashes':manifests,'metadata_sha256':sha(project/'pannuke19_metadata.csv'),'test_decoded':False}

def verify(ledger,project,data):
    for name,digest in ledger['manifest_hashes'].items():
        if sha(project/name)!=digest:raise ValueError(f'Manifest mismatch: {name}')
    if sha(project/'pannuke19_metadata.csv')!=ledger['metadata_sha256']:raise ValueError('PanNuke metadata mismatch')
    def check(r):
        p=data/r['path']
        if p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:raise ValueError(f'Data mismatch: {r["path"]}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(check,ledger['files']))
    return {'status':'PASS','files':len(ledger['files']),'bytes':ledger['bytes'],'ledger_sha256':hashlib.sha256(json.dumps(ledger,sort_keys=True).encode()).hexdigest(),'test_decoded':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['inventory','verify']);p.add_argument('--project',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--ledger',type=Path,required=True);p.add_argument('--report',type=Path)
    a=p.parse_args()
    if a.action=='inventory':
        r=inventory(a.project,a.data);a.ledger.parent.mkdir(parents=True,exist_ok=True);a.ledger.write_text(json.dumps(r,indent=2)+'\n')
        a.ledger.with_suffix('.files0').write_bytes(b''.join(x['path'].encode()+b'\0' for x in r['files']))
        report={'files':len(r['files']),'bytes':r['bytes'],'dataset_records':r['dataset_records']}
    else:report=verify(json.loads(a.ledger.read_text()),a.project,a.data)
    if a.report:a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
