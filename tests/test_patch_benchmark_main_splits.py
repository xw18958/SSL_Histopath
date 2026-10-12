"""Regression tests for the versioned, image-disjoint main evaluation splits."""
from __future__ import annotations

import collections
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from patch_benchmark import EXTERNAL_DATASETS, EPOCHS, METHODS, load_manifest, record_identity


@pytest.mark.parametrize("dataset", EXTERNAL_DATASETS)
def test_main_split_provenance(dataset):
    orig = load_manifest(dataset, "original")
    main = load_manifest(dataset, "main")
    assert main.num_classes == orig.num_classes
    assert list(main.class_names) == list(orig.class_names)
    src = collections.Counter((record_identity(r), r["class_id"]) for r in orig.manifest["records"])
    for r in main.manifest["records"]:
        key = (record_identity(r), r["class_id"])
        assert src[key] > 0
        src[key] -= 1
    counts = collections.Counter(r["class_id"] for r in main.split_rows["train"])
    assert all(counts[c] >= 10 for c in range(main.num_classes))
    assert len(main.split_rows["test"]) > 0
    assert main.manifest_sha256
    assert main.manifest["preprocessing"] == orig.manifest["preprocessing"]
    for s in ("train", "val", "test"):
        assert len(main.split_rows[s]) > 0


def test_brea_image_only_no_patient_isolation_requirement():
    orig = load_manifest("breakhis_8subtype", "original")
    main = load_manifest("breakhis_8subtype", "main")
    assert len(orig.manifest["records"]) == len(main.manifest["records"]) == 3552
    assert main.manifest["remaining_duplicate_content_groups_cross_split"] == 0
    assert main.manifest["original_duplicate_content_groups_cross_split"] > 0
    oldcounts = collections.Counter((r["class_id"], r["split"]) for r in orig.manifest["records"])
    newcounts = collections.Counter((r["class_id"], r["split"]) for r in main.manifest["records"])
    assert oldcounts == newcounts
    assert all(x["split"] == "val" for group in main.manifest["conflicting_label_records"]
               for x in group)


@pytest.mark.parametrize("dataset", ("endometrial_4class", "oral_oscc"))
def test_filter_does_not_move_original_records(dataset):
    orig = load_manifest(dataset, "original")
    main = load_manifest(dataset, "main")
    ref = {record_identity(r): (r["split"], r["class_id"]) for r in orig.manifest["records"]}
    for r in main.manifest["records"]:
        assert (r["split"], r["class_id"]) == ref[record_identity(r)]
    assert 0 < main.manifest["removed_records_count"] < 100
    assert main.manifest["remaining_duplicate_content_groups_cross_split"] == 0
    grouping = collections.defaultdict(set)
    for r in main.manifest["records"]:
        if r.get("group_id"):
            grouping[r["group_id"]].add(r["split"])
    assert all(len(split) == 1 for split in grouping.values())


def test_required_full_matrix():
    assert len(METHODS) == 2 and len(EPOCHS) == 5
    assert len(EXTERNAL_DATASETS) == 15
    assert len(METHODS) * len(EPOCHS) * len(EXTERNAL_DATASETS) == 150
