"""Inspect/freeze immutable external classification manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pannuke_ssl.ssl_framework import load_standard_config
from pannuke_ssl.ssl_framework.external_datasets import (
    EXTERNAL_DATASETS,
    external_dataset_status,
    prepare_external_manifest,
    ready_external_datasets,
)


def main() -> None:
    default_output = Path(load_standard_config("lejepa")["manifests"]["root"])
    parser = argparse.ArgumentParser(description="Prepare immutable external linear-probe manifests")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dataset", choices=EXTERNAL_DATASETS)
    mode.add_argument("--ready-suite", action="store_true", help="Prepare every protocol-ready dataset")
    mode.add_argument("--list", action="store_true", help="Show planned/ready/blocked dataset protocols")
    parser.add_argument("--tier", choices=("main", "supplementary", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=default_output)
    parser.add_argument("--force", action="store_true", help="Deliberately replace existing manifest/checksum files")
    args = parser.parse_args()
    if args.list:
        print(json.dumps(external_dataset_status(), indent=2), flush=True)
        return
    if args.dataset:
        result = prepare_external_manifest(args.dataset, args.output_root, force=args.force)
        print(json.dumps(result, indent=2), flush=True)
        return
    tier = None if args.tier == "all" else args.tier
    datasets = ready_external_datasets(tier=tier)
    results = {dataset: prepare_external_manifest(dataset, args.output_root, force=args.force) for dataset in datasets}
    print(json.dumps({"datasets": datasets, "results": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
