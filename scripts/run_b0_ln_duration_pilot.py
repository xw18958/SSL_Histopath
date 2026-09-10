#!/usr/bin/env python
"""Run the isolated B0-LN adaptive duration pilot (validation only)."""
from __future__ import annotations

import argparse
import json

from pannuke_ssl.b0_ln_training import run_b0_ln_pretraining
from pannuke_ssl.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the B0-LN duration pilot")
    parser.add_argument("--config", default="configs/b0_ln_duration_pilot.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    print(json.dumps(run_b0_ln_pretraining(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
