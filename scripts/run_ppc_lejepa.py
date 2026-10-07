from __future__ import annotations

import argparse
import json

from pannuke_ssl.ssl_framework.ppc_campaign import (
    evaluate,
    functional_smoke,
    preflight,
    prepare,
    pretrain,
    tune,
)


def main():
    parser = argparse.ArgumentParser(description="PPC-LeJEPA functional-drift 250/250 campaign")
    parser.add_argument(
        "action",
        choices=["prepare", "preflight", "functional-smoke", "tune", "pretrain", "downstream"],
    )
    parser.add_argument("--epoch", type=int, choices=[100, 150, 200, 250])
    args = parser.parse_args()
    if args.action == "downstream" and args.epoch is None:
        parser.error("--epoch is required for downstream")

    if args.action == "prepare":
        result = prepare()
    elif args.action == "preflight":
        result = preflight()
    elif args.action == "functional-smoke":
        result = functional_smoke()
    elif args.action == "tune":
        result = tune()
    elif args.action == "pretrain":
        result = pretrain()
    else:
        result = evaluate(args.epoch)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
