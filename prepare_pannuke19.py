#!/usr/bin/env python3
"""Create a deterministic, balanced PanNuke tissue-classification index."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pannuke19_dataset import SourceRecord, build_source_index, read_metadata, verify_records


SEED = 20260903
SAMPLES_PER_CLASS = 134
SPLIT_COUNTS = {"train": 108, "val": 13, "test": 13}
CSV_FIELDS = ["fold", "sample_index", "tissue_label", "class_id", "split"]


def select_balanced_records(index: dict[tuple[int, int], SourceRecord]) -> list[dict[str, object]]:
    """Sample and split every class deterministically from all source folds."""
    by_class: dict[int, list[SourceRecord]] = defaultdict(list)
    for record in index.values():
        by_class[record.class_id].append(record)

    expected_ids = set(range(19))
    if set(by_class) != expected_ids:
        raise ValueError(f"Expected class IDs 0--18, found {sorted(by_class)}")

    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    for class_id in sorted(by_class):
        candidates = sorted(by_class[class_id], key=lambda r: (r.fold, r.sample_index))
        if len(candidates) < SAMPLES_PER_CLASS:
            raise ValueError(
                f"Class {class_id} has {len(candidates)} samples; need {SAMPLES_PER_CLASS}."
            )
        chosen = [candidates[i] for i in rng.permutation(len(candidates))[:SAMPLES_PER_CLASS]]
        chosen = [chosen[i] for i in rng.permutation(SAMPLES_PER_CLASS)]
        cursor = 0
        for split, count in SPLIT_COUNTS.items():
            for record in chosen[cursor : cursor + count]:
                rows.append(
                    {
                        "fold": record.fold,
                        "sample_index": record.sample_index,
                        "tissue_label": record.tissue_label,
                        "class_id": record.class_id,
                        "split": split,
                    }
                )
            cursor += count

    rng.shuffle(rows)
    return rows


def validate_selection(rows: list[dict[str, object]], index: dict[tuple[int, int], SourceRecord]) -> None:
    if len(rows) != 19 * SAMPLES_PER_CLASS:
        raise AssertionError(f"Expected 2546 rows, got {len(rows)}")

    class_counts = Counter(int(row["class_id"]) for row in rows)
    if class_counts != Counter({class_id: SAMPLES_PER_CLASS for class_id in range(19)}):
        raise AssertionError(f"Bad class counts: {class_counts}")

    split_counts = Counter(str(row["split"]) for row in rows)
    expected_splits = {split: count * 19 for split, count in SPLIT_COUNTS.items()}
    if split_counts != Counter(expected_splits):
        raise AssertionError(f"Bad split counts: {split_counts}")

    for class_id in range(19):
        per_split = Counter(str(row["split"]) for row in rows if int(row["class_id"]) == class_id)
        if per_split != Counter(SPLIT_COUNTS):
            raise AssertionError(f"Class {class_id} has bad split counts: {per_split}")

    verify_records(rows, index)


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]]) -> None:
    print(f"Selected samples: {len(rows)}")
    print("Split sizes:")
    for split in SPLIT_COUNTS:
        print(f"  {split}: {sum(row['split'] == split for row in rows)}")
    print("Per-class distributions (class_id, label, train, val, test, total):")
    labels = {int(row["class_id"]): str(row["tissue_label"]) for row in rows}
    for class_id in range(19):
        counts = Counter(str(row["split"]) for row in rows if int(row["class_id"]) == class_id)
        print(
            f"  {class_id:2d} {labels[class_id]:16s} "
            f"{counts['train']:3d} {counts['val']:3d} {counts['test']:3d} {sum(counts.values()):3d}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing PanNuke Parquet shards")
    parser.add_argument("--output", type=Path, default=Path("pannuke19_metadata.csv"))
    args = parser.parse_args()

    index = build_source_index(args.data_root)
    rows = select_balanced_records(index)
    validate_selection(rows, index)
    write_csv(rows, args.output)

    # Re-read the only persisted metadata artifact and validate it again.
    persisted_rows = read_metadata(args.output)
    validate_selection(persisted_rows, index)
    print_summary(persisted_rows)
    print(f"Metadata written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
