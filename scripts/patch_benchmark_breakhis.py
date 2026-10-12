#!/usr/bin/env python3
"""One-time, deterministic image-only duplicate repair for BreaKHis.

Retains the original dataset and exactly the same images, classes, and
train/val/test class counts; corrects exact-content duplicates crossing splits
through within-class swaps of individual (nonduplicate) images.
"""
from __future__ import annotations
import collections
import hashlib
import json
from pathlib import Path
import numpy as np

from patch_benchmark import (
    ROOT, SPLITS, dump_json, load_manifest, sha_file,
)
from pannuke_ssl.ssl_framework.external_datasets import ExternalProbeImageDataset

SLUG = "breakhis_8subtype"
NAME = "breakhis_image_disjoint_v1"


def image_fingerprints(rows, root):
    loader = ExternalProbeImageDataset(rows, root)
    signatures = []
    for r in rows:
        img = loader._open_record(r)
        digest = hashlib.sha256(
            f"{img.width}x{img.height}|".encode() + img.tobytes()
        ).hexdigest()
        signatures.append(digest)
    return signatures


def build(*, overwrite=False):
    original = load_manifest(SLUG, "original")
    target = ROOT / "manifests/patch_benchmark_v1" / (NAME + ".json")
    if target.is_file() and not overwrite:
        payload = json.loads(target.read_text())
        if payload["parent_manifest_sha256"] != original.manifest_sha256:
            raise RuntimeError("Parent manifest changed; refuse stale repaired split")
        return target
    rows = [dict(r) for r in original.manifest["records"]]
    old = [r["split"] for r in rows]
    signatures = image_fingerprints(rows, original.root)
    buckets = collections.defaultdict(list)
    for i, h in enumerate(signatures):
        buckets[h].append(i)
    crossing = [ids for ids in buckets.values()
                if len({old[i] for i in ids}) > 1]
    conflicting = [ids for ids in buckets.values()
                   if len({rows[i]["class_id"] for i in ids}) > 1]
    if not crossing:
        print("No duplicate content crosses any splits; no repair needed.")
    changed = []
    reserved = set()
    targets = [ids for ids in buckets.values() if (
        len({old[i] for i in ids}) > 1 or
        len({rows[i]["class_id"] for i in ids}) > 1
    )]
    for group in targets:
        labels = {rows[i]["class_id"] for i in group}
        if len(labels) != 1:
            print("LABEL_CONFLICT", json.dumps([{"class_id": rows[i]["class_id"], "split": rows[i]["split"], "path": rows[i]["relative_path"]} for i in group]), flush=True)
        # Prefer retaining existing gallery membership to avoid test copies in gallery.
        # Conflicting-label exact copies are retained only in validation;
        # neither the reference gallery nor test queries should contain them.
        dest = "val" if len(labels) > 1 else min((old[i] for i in group), key=lambda s: (
            {"train": 0, "val": 1, "test": 2}[s], s
        ))
        for i in group:
            if old[i] == dest:
                continue
            if i in reserved:
                raise AssertionError("Image visited twice")
            src = old[i]
            cls = rows[i]["class_id"]
            # Swap with a stable, truly singleton image from dest and same class
            # to retain the exact train/val/test/class counts without patient grouping.
            candidates = [
                j for j, r in enumerate(rows)
                if j not in reserved and j not in group
                and len(buckets[signatures[j]]) == 1
                and r["class_id"] == cls and r["split"] == dest
            ]
            if not candidates:
                raise RuntimeError("Unable to preserve split/class sizes")
            j = min(candidates, key=lambda t: rows[t]["relative_path"])
            rows[i]["split"] = dest
            rows[j]["split"] = src
            reserved.update((i, j))
            changed.append({
                "duplicate_original_split": src, "duplicate_new_split": dest,
                "duplicate_path": rows[i]["relative_path"],
                "swapped_original_split": dest, "swapped_new_split": src,
                "swapped_path": rows[j]["relative_path"],
            })
    new_buckets = collections.defaultdict(set)
    for row, signature in zip(rows, signatures):
        new_buckets[signature].add(row["split"])
    remaining = [h for h, splits in new_buckets.items() if len(splits) > 1]
    if remaining:
        raise RuntimeError("Pixel duplicate repair failed; remaining=%d" % len(remaining))
    original_counter = collections.Counter((r["split"], r["class_id"])
                                           for r in original.manifest["records"])
    new_counter = collections.Counter((r["split"], r["class_id"]) for r in rows)
    if original_counter != new_counter:
        raise RuntimeError("The image-level repair changed class balance")
    for r, oldr in zip(rows, original.manifest["records"]):
        if {k: v for k, v in r.items() if k != "split"} != {
            k: v for k, v in oldr.items() if k != "split"
        }:
            raise RuntimeError("A source image or label was modified")
    payload = {
        "schema": NAME, "dataset": SLUG, "split_version": NAME,
        "parent_manifest_sha256": original.manifest_sha256,
        "dataset_root_spec": original.config.root_spec,
        "class_names": list(original.class_names),
        "source_overlap_policy": "no identical RGB content across train/val/test; same patients allowed",
        "image_fingerprint_algorithm": "SHA256(WxH + RGB pixel bytes)",
        "original_duplicate_content_groups_cross_split": len(crossing),
        "conflicting_label_fingerprint_groups": len(conflicting),
        "conflicting_label_records": [
            [{"class_id": rows[i]["class_id"], "relative_path": rows[i]["relative_path"], "split": rows[i]["split"]} for i in group]
            for group in conflicting
        ],
        "repaired_record_swaps": changed,
        "remaining_duplicate_content_groups_cross_split": 0,
        "records": rows,
        "split_counts": dict(collections.Counter(r["split"] for r in rows)),
        "source_manifest_sha256": original.manifest_sha256,
    }
    dump_json(target, payload)
    print(json.dumps({
        "path": str(target), "parent_sha256": original.manifest_sha256,
        "new_sha256": sha_file(target),
        "cross_split_content_duplicates_fixed": len(crossing),
        "image_pair_swaps": len(changed),
        "per_class_split_counts_preserved": True,
        "remaining_content_overlap": len(remaining)
    }, indent=2))
    return target


if __name__ == "__main__":
    build()
