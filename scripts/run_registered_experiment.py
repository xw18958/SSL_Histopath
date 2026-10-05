from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/ssl_standard/experiment_registry.yaml"


def registry() -> dict[str, dict]:
    document = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    return dict(document["experiments"])


def main() -> None:
    p = argparse.ArgumentParser(description="Run a canonical registered SSL experiment on this server")
    p.add_argument("action", nargs="?", choices=("pretrain","downstream","downstream-suite","image-text-retrieval","image-text-retrieval-suite","report"))
    p.add_argument("--experiment")
    p.add_argument("--dataset")
    p.add_argument("--suite-tier", choices=("main","supplementary","all"), default="main")
    p.add_argument("--list", action="store_true")
    p.add_argument("--ignore-tuned", action="store_true")
    args = p.parse_args()
    entries = registry()
    if args.list:
        for run_id, spec in entries.items():
            print(f"{run_id}\t{spec['method']}\t{spec.get('role','')}\t{spec.get('overrides',{})}")
        return
    if not args.experiment or not args.action:
        p.error("--experiment and action are required unless --list is used")
    if args.experiment not in entries:
        p.error(f"Unknown experiment {args.experiment!r}")
    spec = entries[args.experiment]
    cmd = [sys.executable, str(ROOT/'scripts/run_ssl_standard.py'), args.action, '--method', spec['method'], '--run-id', args.experiment]
    overrides = dict(spec.get('overrides') or {})
    if 'method.objective.simplex_components' in overrides:
        cmd += ['--simplex-components', str(int(overrides['method.objective.simplex_components']))]
    sigma = overrides.get('method.objective.simplex_sigma')
    if sigma is not None and float(sigma) != 1.0:
        p.error('Registry violates fixed simplex sigma=1.0 protocol')
    if args.dataset:
        cmd += ['--dataset', args.dataset]
    if args.action.endswith('-suite'):
        cmd += ['--suite-tier', args.suite_tier]
    if args.ignore_tuned:
        cmd += ['--ignore-tuned']
    print('EXEC', ' '.join(cmd), flush=True)
    os.execv(sys.executable, cmd)


if __name__ == '__main__':
    main()
