#!/usr/bin/env python
"""Run B1's locked 300-epoch, validation-selected duration experiment."""
from __future__ import annotations

import argparse
import json

from pannuke_ssl.b1_training import run_b1_pretraining
from pannuke_ssl.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the B1 residual discrete-velocity duration pilot")
    parser.add_argument("--config", default="configs/b1_duration_pilot.yaml")
    args = parser.parse_args()
    print(json.dumps(run_b1_pretraining(load_yaml(args.config)), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
