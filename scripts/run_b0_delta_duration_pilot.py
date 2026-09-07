#!/usr/bin/env python
"""Run the validation-only 300-epoch B0-Delta duration experiment."""
from __future__ import annotations

import argparse
import json

from pannuke_ssl.b0_delta_training import run_b0_delta_pretraining
from pannuke_ssl.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run B0-Delta with the locked B0 duration protocol"
    )
    parser.add_argument(
        "--config",
        default="configs/b0_delta_duration_pilot.yaml",
    )
    args = parser.parse_args()
    config = load_yaml(args.config)
    print(
        json.dumps(
            run_b0_delta_pretraining(config),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
