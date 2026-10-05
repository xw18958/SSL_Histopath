from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

from pannuke_ssl.ssl_framework.runtime_paths import require_runtime_environment
from pannuke_ssl.ssl_framework.run_management import sanitize_run_id


def join_target(base: str, suffix: str) -> str:
    if ':' in base and not base.startswith('/'):
        host,path=base.split(':',1)
        return f"{host}:{path.rstrip('/')}/{suffix.strip('/')}/"
    return str(Path(base)/suffix) + '/'


def main() -> None:
    p=argparse.ArgumentParser(description='Push/pull one portable SSL run to the configured results master')
    p.add_argument('direction',choices=('push','pull'))
    p.add_argument('--run-id',required=True)
    p.add_argument('--execute',action='store_true',help='Actually copy; default is rsync dry-run')
    args=p.parse_args()
    run_id=sanitize_run_id(args.run_id)
    env=require_runtime_environment()
    master=os.environ.get('SSL_RESULTS_MASTER')
    if not master:
        p.error('SSL_RESULTS_MASTER is not configured in .ssl_server.env')
    local=str(Path(env['SSL_RUN_ROOT'])/'ssl_standard/runs'/run_id) + '/'
    remote=join_target(master,f'ssl_standard/runs/{run_id}')
    src,dst=(local,remote) if args.direction=='push' else (remote,local)
    cmd=['rsync','-a','--partial','--prune-empty-dirs',
         '--exclude','pretrain_full/checkpoints/last.pt',
         '--exclude','pretrain_full/checkpoints/best.pt']
    if not args.execute: cmd.append('--dry-run')
    cmd += [src,dst]
    print(' '.join(shlex.quote(x) for x in cmd),flush=True)
    subprocess.run(cmd,check=True)

if __name__=='__main__': main()
