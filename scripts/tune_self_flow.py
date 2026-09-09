#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from pannuke_ssl.config import load_yaml
from pannuke_ssl.self_flow_tuning import run_self_flow_tuning


def main() -> None:
    parser = argparse.ArgumentParser(description="B0-budget-equivalent Self-Flow tuning")
    parser.add_argument("--config", default="configs/tune_self_flow.yaml")
    args = parser.parse_args()
    print(json.dumps(run_self_flow_tuning(load_yaml(args.config)), indent=2))


if __name__ == "__main__":
    main()
