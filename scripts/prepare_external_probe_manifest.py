"""Explicitly freeze a balanced optional-dataset split before probe evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pannuke_ssl.ssl_framework.external_datasets import EXTERNAL_DATASETS, prepare_external_manifest


PROJECT_ROOT = Path("/raid1/xwan0900/SSL_proj")
OUTPUT_ROOT = PROJECT_ROOT / "outputs/ssl_standard"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare one immutable optional linear-probe manifest")
    parser.add_argument("--dataset", required=True, choices=EXTERNAL_DATASETS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true", help="Deliberately replace an existing manifest and checksum")
    args = parser.parse_args()
    print(json.dumps(prepare_external_manifest(args.dataset, args.output_root, force=args.force), indent=2), flush=True)


if __name__ == "__main__":
    main()
