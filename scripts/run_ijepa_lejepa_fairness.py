from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from pannuke_ssl.config import load_yaml
from pannuke_ssl.ijepa_lejepa_fairness_source import (
    SETTINGS,
    comparison,
    require_smoke,
    run_setting,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ijepa_lejepa_fairness.yaml")
    parser.add_argument("--setting", choices=("all", *SETTINGS), default="all")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    require_smoke(config)

    if args.setting != "all":
        result = run_setting(config, args.setting)
        print(json.dumps(result, indent=2), flush=True)
        return

    if args.child:
        raise ValueError("--child requires one concrete --setting")

    script = str(Path(__file__).resolve())
    for setting in SETTINGS:
        command = [
            sys.executable,
            script,
            "--config",
            args.config,
            "--setting",
            setting,
            "--child",
        ]
        print(json.dumps({"launching": setting, "command": command}), flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            print(
                json.dumps(
                    {"setting": setting, "returncode": completed.returncode, "continuing": True}
                ),
                flush=True,
            )

    result = comparison(config)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
