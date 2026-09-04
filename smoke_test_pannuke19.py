#!/usr/bin/env python3
"""End-to-end validation for balanced PanNuke classification metadata and loaders."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

from pannuke19_dataset import build_dataloaders, build_source_index, read_metadata, verify_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()

    rows = read_metadata(args.metadata)
    index = build_source_index(args.data_root)
    verify_records(rows, index)
    assert len(rows) == 2546
    assert Counter(row["class_id"] for row in rows) == Counter({class_id: 134 for class_id in range(19)})
    assert Counter(row["split"] for row in rows) == Counter({"train": 2052, "val": 247, "test": 247})

    cached = build_dataloaders(args.metadata, args.data_root, batch_size=8, num_workers=0, cache_in_ram=True)
    uncached = build_dataloaders(args.metadata, args.data_root, batch_size=8, num_workers=0, cache_in_ram=False)
    for split in ("train", "val", "test"):
        images, labels = next(iter(cached[split]))
        assert images.shape == (8, 3, 256, 256)
        assert images.dtype == torch.uint8
        assert int(labels.min()) >= 0 and int(labels.max()) <= 18
        assert cached[split].dataset[0][1] == uncached[split].dataset[0][1]
    print("Smoke test passed: metadata, cached loader, and uncached loader are valid.")


if __name__ == "__main__":
    main()
