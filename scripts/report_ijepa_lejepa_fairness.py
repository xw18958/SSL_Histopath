import argparse
import json

from pannuke_ssl.config import load_yaml
from pannuke_ssl.ijepa_lejepa_fairness_source import comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ijepa_lejepa_fairness.yaml")
    args = parser.parse_args()
    print(json.dumps(comparison(load_yaml(args.config)), indent=2), flush=True)
