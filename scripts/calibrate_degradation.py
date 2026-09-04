#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from pannuke_ssl.calibration import run_calibration
from pannuke_ssl.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate global PanNuke degradation endpoints using VIF")
    parser.add_argument("--config", default="configs/calibration.yaml")
    args = parser.parse_args()
    print(json.dumps(run_calibration(load_yaml(args.config)), indent=2))


if __name__ == "__main__":
    main()
