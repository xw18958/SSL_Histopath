#!/usr/bin/env python
"""Run the validation-only B0 SSL-duration experiment (no test evaluation)."""

from __future__ import annotations

import argparse
import json

from pannuke_ssl.config import load_yaml
from pannuke_ssl.training import run_pretraining


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 300-epoch B0 duration pilot")
    parser.add_argument("--config", default="configs/b0_duration_pilot.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if not bool(config.get("monitor", {}).get("enabled", False)):
        raise ValueError("The duration-pilot config must enable validation-only monitoring")
    print(json.dumps(run_pretraining(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
