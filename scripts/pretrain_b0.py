#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from pannuke_ssl.config import load_yaml
from pannuke_ssl.training import run_pretraining


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain the B0 PanNuke degradation predictor")
    parser.add_argument("--config", default="configs/b0.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.resume:
        config["train"]["resume"] = args.resume
    print(json.dumps(run_pretraining(config), indent=2))


if __name__ == "__main__":
    main()
