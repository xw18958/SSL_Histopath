#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from pannuke_ssl.config import load_yaml
from pannuke_ssl.self_flow_probe import run_self_flow_linear_probe


def main() -> None:
    parser = argparse.ArgumentParser(description="Final frozen linear probe for Self-Flow-Pixel-B/32")
    parser.add_argument("--config", default="configs/self_flow_pixel_b32_duration_final_probe.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    print(json.dumps(run_self_flow_linear_probe(config), indent=2))


if __name__ == "__main__":
    main()
