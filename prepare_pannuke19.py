#!/usr/bin/env python3
"""Create a deterministic PanNuke split with balanced validation and test sets."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pannuke19_dataset import SourceRecord, build_source_index, read_metadata, verify_records


SEED = 20260903
NUM_CLASSES = 19
SOURCE_COUNT = 7901
VAL_PER_CLASS = 42
TEST_PER_CLASS = 42
EXPECTED_SPLIT_COUNTS = {
    "train": SOURCE_COUNT - NUM_CLASSES * (VAL_PER_CLASS + TEST_PER_CLASS),
    "val": NUM_CLASSES * VAL_PER_CLASS,
    "test": NUM_CLASSES * TEST_PER_CLASS,
}
CSV_FIELDS = ["fold", "sample_index", "tissue_label", "class_id", "split"]


def _row(record: SourceRecord, split: str) -> dict[str, object]:
    return {
        "fold": record.fold,
        "sample_index": record.sample_index,
        "tissue_label": record.tissue_label,
        "class_id": record.class_id,
        "split": split,
    }


def select_balanced_records(index: dict[tuple[int, int], SourceRecord]) -> list[dict[str, object]]:
    """Use every source image; balance only validation and test across the 19 classes."""
    if len(index) != SOURCE_COUNT:
        raise ValueError(f"Expected {SOURCE_COUNT} PanNuke source records, found {len(index)}")

    by_class: dict[int, list[SourceRecord]] = defaultdict(list)
    for record in index.values():
        by_class[record.class_id].append(record)

    expected_ids = set(range(NUM_CLASSES))
    if set(by_class) != expected_ids:
        raise ValueError(f"Expected class IDs 0--{NUM_CLASSES - 1}, found {sorted(by_class)}")

    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    held_out_per_class = VAL_PER_CLASS + TEST_PER_CLASS

    for class_id in sorted(by_class):
        candidates = sorted(by_class[class_id], key=lambda r: (r.fold, r.sample_index))
        if len(candidates) < held_out_per_class:
            raise ValueError(
                f"Class {class_id} has {len(candidates)} samples; need at least {held_out_per_class}."
            )
        order = rng.permutation(len(candidates))
        validation = [candidates[i] for i in order[:VAL_PER_CLASS]]
        test = [candidates[i] for i in order[VAL_PER_CLASS:held_out_per_class]]
        train = [candidates[i] for i in order[held_out_per_class:]]
        rows.extend(_row(record, "train") for record in train)
        rows.extend(_row(record, "val") for record in validation)
        rows.extend(_row(record, "test") for record in test)

    rng.shuffle(rows)
    return rows


def validate_selection(rows: list[dict[str, object]], index: dict[tuple[int, int], SourceRecord]) -> None:
    if len(index) != SOURCE_COUNT:
        raise AssertionError(f"Expected {SOURCE_COUNT} source records, got {len(index)}")
    if len(rows) != SOURCE_COUNT:
        raise AssertionError(f"Expected {SOURCE_COUNT} metadata rows, got {len(rows)}")

    verify_records(rows, index)
    row_keys = {(int(row["fold"]), int(row["sample_index"])) for row in rows}
    if len(row_keys) != SOURCE_COUNT or row_keys != set(index):
        raise AssertionError("Metadata must partition every PanNuke source image exactly once")

    split_counts = Counter(str(row["split"]) for row in rows)
    if split_counts != Counter(EXPECTED_SPLIT_COUNTS):
        raise AssertionError(f"Bad split counts: {split_counts}")

    source_class_counts = Counter(record.class_id for record in index.values())
    metadata_class_counts = Counter(int(row["class_id"]) for row in rows)
    if metadata_class_counts != source_class_counts:
        raise AssertionError("Metadata class counts do not match the full PanNuke source")

    for class_id in range(NUM_CLASSES):
        per_split = Counter(str(row["split"]) for row in rows if int(row["class_id"]) == class_id)
        if per_split["val"] != VAL_PER_CLASS or per_split["test"] != TEST_PER_CLASS:
            raise AssertionError(f"Class {class_id} has bad held-out counts: {per_split}")
        expected_train = source_class_counts[class_id] - VAL_PER_CLASS - TEST_PER_CLASS
        if per_split["train"] != expected_train:
            raise AssertionError(
                f"Class {class_id} train count {per_split['train']} != expected remaining {expected_train}"
            )


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]]) -> None:
    print(f"Selected samples: {len(rows)}")
    print("Split sizes:")
    for split in ("train", "val", "test"):
        print(f"  {split}: {sum(row['split'] == split for row in rows)}")
    print("Per-class distributions (class_id, label, train, val, test, total):")
    labels = {int(row["class_id"]): str(row["tissue_label"]) for row in rows}
    for class_id in range(NUM_CLASSES):
        counts = Counter(str(row["split"]) for row in rows if int(row["class_id"]) == class_id)
        print(
            f"  {class_id:2d} {labels[class_id]:16s} "
            f"{counts['train']:4d} {counts['val']:3d} {counts['test']:3d} {sum(counts.values()):4d}"
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

    persisted_rows = read_metadata(args.output)
    validate_selection(persisted_rows, index)
    print_summary(persisted_rows)
    print(f"Metadata written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
