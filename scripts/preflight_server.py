from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
import yaml

from pannuke_ssl.config import set_dotted
from pannuke_ssl.ssl_framework import load_standard_config
from pannuke_ssl.ssl_framework.external_datasets import load_external_manifest, ready_external_datasets
from pannuke_ssl.ssl_framework.run_management import attach_run_context
from pannuke_ssl.ssl_framework.runtime_paths import require_runtime_environment, runtime_identity

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / 'configs/ssl_standard/experiment_registry.yaml'


def _entry(run_id: str) -> dict:
    entries = yaml.safe_load(REGISTRY.read_text(encoding='utf-8'))['experiments']
    if run_id not in entries:
        raise ValueError(f'Unknown experiment {run_id!r}')
    return dict(entries[run_id])


def main() -> None:
    p=argparse.ArgumentParser(description='Preflight one registered experiment on the current server')
    p.add_argument('--experiment', required=True)
    p.add_argument('--action', choices=('pretrain','downstream','downstream-suite','image-retrieval','image-retrieval-suite'), default='pretrain')
    p.add_argument('--dataset')
    args=p.parse_args()
    checks=[]
    def check(name: str, ok: bool, detail):
        checks.append({'check':name,'ok':bool(ok),'detail':detail})

    env=require_runtime_environment()
    spec=_entry(args.experiment)
    c=load_standard_config(spec['method'])
    for key,value in dict(spec.get('overrides') or {}).items():
        set_dotted(c,key,value)
    c,run_id,run_root=attach_run_context(c,args.experiment)
    ident=runtime_identity()
    check('canonical_git_commit', ident['git_commit'] != 'unknown', ident['git_commit'])
    check('source_tree_clean', ident['source_tree_clean'], ident['source_tree_clean'])
    for name,path in (
        ('pannuke_data',Path(c['data']['root'])),
        ('metadata_csv',Path(c['data']['metadata_csv'])),
        ('backbone_config',Path(c['backbone']['config_dir'])),
        ('manifest_root',Path(c['manifests']['root'])),
    ):
        check(name,path.exists(),str(path))
    output_base=Path(c['output']['root'])
    output_base.mkdir(parents=True,exist_ok=True)
    check('run_root_writable',os.access(output_base,os.W_OK),str(run_root))
    check('cuda_available',torch.cuda.is_available(),torch.cuda.device_count())
    if torch.cuda.is_available():
        free,total=torch.cuda.mem_get_info()
        check('gpu_identity',True,{'name':torch.cuda.get_device_name(0),'free_gib':round(free/2**30,2),'total_gib':round(total/2**30,2)})
    datasets=[]
    if args.action in ('downstream','image-retrieval'):
        if not args.dataset: raise ValueError('--dataset required for single-dataset evaluation preflight')
        datasets=[args.dataset]
    elif args.action.endswith('-suite'):
        datasets=list(ready_external_datasets(tier='main'))
    for dataset in datasets:
        try:
            ds=load_external_manifest(dataset,Path(c['manifests']['root']))
            check(f'manifest:{dataset}',True,{'sha256':ds.manifest_sha256,'splits':ds.split_counts})
        except Exception as exc:
            check(f'manifest:{dataset}',False,str(exc))
    checkpoints=[run_root/'pretrain_full/checkpoints'/f'epoch_{epoch}.pt' for epoch in c['training']['checkpoint_epochs']]
    if args.action=='pretrain':
        occupied=(run_root/'pretrain_full/run_summary.json').exists() or (run_root/'pretrain_full/pretrain_metrics.csv').exists()
        check('pretrain_output_clear',not occupied,str(run_root/'pretrain_full'))
    else:
        check('required_checkpoints',all(path.is_file() for path in checkpoints),[str(p) for p in checkpoints if not p.is_file()])
    result={'experiment':run_id,'method':c['method']['name'],'action':args.action,'runtime':ident,'checks':checks,'passed':all(x['ok'] for x in checks)}
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 2)

if __name__=='__main__': main()
