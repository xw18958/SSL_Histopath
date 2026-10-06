from __future__ import annotations
import argparse
import gc
import json
import time
from pathlib import Path

import torch
from pannuke_ssl.ssl_framework.retrieval_repair import run_train_val, run_test, run_smoke
from pannuke_ssl.ssl_framework.runtime_paths import require_runtime_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("smoke", "train-val", "test"))
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", default="100,150,200,250,300")
    parser.add_argument("--datasets", default="arch,ipath")
    args = parser.parse_args()
    require_runtime_environment()
    config = json.loads((args.source_run / "pretrain_full/resolved_config.json").read_text())
    epochs = [int(x) for x in args.epochs.split(",")]
    datasets = args.datasets.split(",")
    for epoch in epochs:
        checkpoint = args.source_run / "pretrain_full/checkpoints" / f"epoch_{epoch}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        for dataset in datasets:
            started = time.time()
            print(json.dumps({"event": "start", "phase": args.phase, "epoch": epoch, "dataset": dataset}), flush=True)
            out = args.output_root / "image_text_retrieval" / dataset / f"epoch_{epoch}"
            if args.phase == "smoke":
                result = run_smoke(config, checkpoint, args.output_root / "smoke" / f"{dataset}_{epoch}.json", dataset=dataset)
            else:
                call = run_train_val if args.phase == "train-val" else run_test
                result = call(config, checkpoint, out, dataset=dataset, cache_root=args.output_root / "cache")
            print(json.dumps({"event": "complete", "phase": args.phase, "epoch": epoch, "dataset": dataset,
                              "seconds": time.time() - started, "selection": result.get("validation"),
                              "test": result.get("test"), "status": result.get("status", "PASS")}), flush=True)
            del result
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
