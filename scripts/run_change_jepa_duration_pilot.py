from __future__ import annotations

import argparse
import json
from pathlib import Path

from pannuke_ssl.change_jepa_training import run_change_jepa_pretraining
from pannuke_ssl.config import load_yaml


ROOT = Path("/raid1/xwan0900/SSL_proj")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/change_jepa_duration_pilot.yaml"),
    )
    args = parser.parse_args()
    result = run_change_jepa_pretraining(load_yaml(args.config))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
