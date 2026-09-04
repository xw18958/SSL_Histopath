#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from pannuke_ssl.config import load_yaml
from pannuke_ssl.tuning import run_tuning


def main() -> None:
    parser = argparse.ArgumentParser(description="Method-registered SSL hyperparameter tuning")
    parser.add_argument("--config", default="configs/tune_b0.yaml")
    args = parser.parse_args()
    print(json.dumps(run_tuning(load_yaml(args.config)), indent=2))


if __name__ == "__main__":
    main()
