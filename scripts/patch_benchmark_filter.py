#!/usr/bin/env python3
"""Versioned, image-only split repair for all non-BreaKHis datasets.

Unlike BreaKHis, do not move source-grouped images between splits: drop only
the minimum conflicting image records needed to prevent exact pixel-identity
overlap while preserving existing slide/patient group isolation.
"""
from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

from patch_benchmark import ROOT, SPLITS, dump_json, load_manifest, sha_file
from pannuke_ssl.ssl_framework.external_datasets import ExternalProbeImageDataset

VERSION = 1
MAIN_EXCEPTIONS = ("endometrial_4class", "oral_oscc")


def fingerprint(image):
    return hashlib.sha256(f"{image.width}x{image.height}|".encode() + image.tobytes()).hexdigest()


def build(dataset, *, overwrite=False):
    if dataset == "breakhis_8subtype":
        raise ValueError("Use the dedicated BreaKHis class-balance-preserving repair")
    original = load_manifest(dataset, "original")
    name = f"{dataset}_image_disjoint_v{VERSION}"
    path = ROOT / "manifests/patch_benchmark_v1" / (name + ".json")
    if path.is_file() and not overwrite:
        j = json.loads(path.read_text())
        if j["parent_manifest_sha256"] != original.manifest_sha256:
            raise RuntimeError("Source manifest changed since repairing")
        return path

    rows = [dict(r) for r in original.manifest["records"]]
    data = ExternalProbeImageDataset(rows, original.root)
    buckets = collections.defaultdict(list)
    for i, row in enumerate(rows):
        im = data._open_record(row)
        buckets[fingerprint(im)].append(i)
    collisions = [g for g in buckets.values() if len({rows[i]["split"] for i in g}) > 1]
    conflicts = [g for g in buckets.values() if len({rows[i]["class_id"] for i in g}) > 1]
    targets = [g for g in buckets.values() if (
        len({rows[i]["split"] for i in g}) > 1 or
        len({rows[i]["class_id"] for i in g}) > 1
    )]
    if not targets:
        raise RuntimeError(f"{dataset}: no image or label conflicts found; no repair needed")
    excluded = set()
    provenance = []
    for indices in targets:
        classes = {int(rows[i]["class_id"]) for i in indices}
        if len(classes) != 1:
            # Label-conflicting identical images cannot be trusted; exclude
            # all instances of the affected image from benchmark evaluation.
            remove = list(indices)
            rule = "conflicting-label-identity-excluded-entirely"
        else:
            # Leave all original train images when possible, then validation;
            # no other image changes split or group.
            keep_split = min((rows[i]["split"] for i in indices),
                             key=lambda s: ("train", "val", "test").index(s))
            remove = [i for i in indices if rows[i]["split"] != keep_split]
            rule = "retain-existing-" + keep_split
        for i in remove:
            excluded.add(i)
            provenance.append({
                "original_row_index": i,
                "identity": str(rows[i].get("relative_path", rows[i].get("source_name",
                           rows[i].get("hdf5_index", "")))),
                "original_split": rows[i]["split"],
                "class_id": int(rows[i]["class_id"]),
                "source_group_id": rows[i].get("group_id"),
                "rule": rule,
            })
    retained = [r for i, r in enumerate(rows) if i not in excluded]
    # Verify no remaining pixel fingerprint spans multiple splits without
    # re-decoding input images.
    for key, group in buckets.items():
        live = [i for i in group if i not in excluded]
        if len({rows[i]["split"] for i in live}) > 1:
            raise RuntimeError("A cross-split duplicate survives")
    old_group_splits = collections.defaultdict(set)
    new_group_splits = collections.defaultdict(set)
    for row in rows:
        if row.get("group_id") is not None:
            old_group_splits[str(row["group_id"])].add(row["split"])
    for row in retained:
        if row.get("group_id") is not None:
            new_group_splits[str(row["group_id"])].add(row["split"])
    if any(len(v) > 1 for v in new_group_splits.values()):
        raise RuntimeError("Repair broke group isolation")
    if any(len(v) > 1 for v in old_group_splits.values()):
        raise RuntimeError("Existing source-group violation found: cannot claim a grouped repair")
    class_counter = collections.Counter((int(r["class_id"]), r["split"]) for r in retained)
    for cid in range(original.num_classes):
        if class_counter[(cid, "train")] < 10 or class_counter[(cid, "val")] < 1 or class_counter[(cid, "test")] < 1:
            raise RuntimeError("Repair made a class/split unusable")
    payload = {
        "schema": "patch_benchmark_exact_image_filter_v1",
        "dataset": dataset, "split_version": name,
        "parent_manifest_sha256": original.manifest_sha256,
        "source_overlap_policy": "no exact RGB image content across train, val, test",
        "image_fingerprint_algorithm": "SHA256(WxH + decoded RGB pixel bytes)",
        "original_duplicate_content_groups_cross_split": len(collisions),
        "original_conflicting_label_image_groups": len(conflicts),
        "remaining_duplicate_content_groups_cross_split": 0,
        "remaining_conflicting_label_image_groups_in_eval": 0,
        "removed_records_count": len(excluded),
        "removed_records": provenance,
        "unchanged_source_group_assignment": True,
        "records": retained,
        "split_counts": dict(collections.Counter(r["split"] for r in retained)),
        "class_names": list(original.class_names),
    }
    dump_json(path, payload)
    print(json.dumps({
        "dataset": dataset, "path": str(path), "sha256": sha_file(path),
        "cross_split_duplicate_groups_fixed": len(collisions),
        "conflicting_label_image_groups_excluded": len(conflicts),
        "removed_image_records": len(excluded),
        "unchanged_patient_slide_split_assignments": True,
        "minimum_training_samples_per_class": min(
            class_counter[(cid, "train")] for cid in range(original.num_classes)),
    }, indent=2), flush=True)
    return path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python patch_benchmark_filter.py DATASET [--overwrite]")
    build(sys.argv[1], overwrite="--overwrite" in sys.argv[2:])
