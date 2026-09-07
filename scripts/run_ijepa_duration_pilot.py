import argparse
import json
from pannuke_ssl.config import load_yaml
from pannuke_ssl.ijepa_training import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ijepa_duration_pilot.yaml")
    args = parser.parse_args()
    print(json.dumps(run(load_yaml(args.config)),indent=2),flush=True)
