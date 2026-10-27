#!/usr/bin/env python3
"""MAE-only adapter to the UNCHANGED patch retrieval/few-shot benchmark.

The historical LeJEPA/K64 matrix aggregation stays untouched. MAE uses the
identical manifest, image preprocessing, test-isolation and ranking code.
Input checkpoint comes from SSL_MAE_CKPT_ROOT, strictly verified in base.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import patch_benchmark as benchmark

def mae_checkpoint_path(method: str, epoch: int) -> Path:
    if method != "mae" or epoch not in (100, 200):
        raise ValueError("Approved MAE benchmark supports epochs 100 and 200 only")
    ckpt_root = os.environ.get("SSL_MAE_CKPT_ROOT")
    if not ckpt_root:
        raise EnvironmentError("SSL_MAE_CKPT_ROOT must name approved MAE checkpoint directory")
    path = Path(ckpt_root) / f"epoch_{epoch}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Approved MAE checkpoint unavailable: {path}")
    return path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("extract","evaluate","extract-evaluate"))
    parser.add_argument("--dataset", required=True, choices=benchmark.EXTERNAL_DATASETS)
    parser.add_argument("--epoch", type=int, required=True, choices=(100,200))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", default="main", choices=("main","original"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-per-class", type=int, default=0)
    args = parser.parse_args()
    if args.smoke_per_class and args.smoke_per_class < 10:
        parser.error("Smoke subsets must contain at least 10/class")
    # Assign just the checkpoint resolver; use all existing benchmark functions
    # as-is (including hash validation, row ordering and full test isolation).
    benchmark.checkpoint_path = mae_checkpoint_path
    if args.action in ("extract", "extract-evaluate"):
        benchmark.extract_one(args.out,"mae",args.epoch,args.dataset,
                              device_name=args.device,variant=args.variant,
                              smoke_per_class=args.smoke_per_class)
    if args.action in ("evaluate", "extract-evaluate"):
        benchmark.evaluate_one(args.out,"mae",args.epoch,args.dataset,
                               variant=args.variant,
                               smoke_per_class=args.smoke_per_class,
                               repeats=benchmark.REPEATS)

if __name__ == "__main__":
    main()
