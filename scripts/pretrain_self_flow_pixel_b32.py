#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from pannuke_ssl.config import load_yaml
from pannuke_ssl.self_flow_training import run_self_flow_pretraining


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain fair Self-Flow-Pixel-B/32 on PanNuke")
    parser.add_argument("--config", default="configs/self_flow_pixel_b32_duration_pilot.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    print(json.dumps(run_self_flow_pretraining(config), indent=2))


if __name__ == "__main__":
    main()
