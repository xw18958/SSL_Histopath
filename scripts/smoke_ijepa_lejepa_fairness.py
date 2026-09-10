import json

from pannuke_ssl.config import load_yaml
from pannuke_ssl.ijepa_lejepa_fairness_source import smoke


if __name__ == "__main__":
    config = load_yaml("configs/ijepa_lejepa_fairness.yaml")
    print(json.dumps(smoke(config), indent=2), flush=True)
