#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from pannuke_ssl.config import load_yaml
from pannuke_ssl.probe import run_linear_probe


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune and evaluate a frozen B0 linear probe")
    parser.add_argument("--config", default="configs/linear_probe.yaml")
    args = parser.parse_args()
    print(json.dumps(run_linear_probe(load_yaml(args.config)), indent=2))


if __name__ == "__main__":
    main()
