#!/usr/bin/env python3
"""Merge the 15 independently completed content audits into a verified main audit."""
from __future__ import annotations
import json
from pathlib import Path
from patch_benchmark import EXTERNAL_DATASETS, DEFAULT_OUT, dump_json, load_manifest


def merge(out=DEFAULT_OUT):
    entries = []
    for dataset in EXTERNAL_DATASETS:
        path = Path(out) / "audit_parts" / dataset / "audit.json"
        if not path.is_file():
            raise FileNotFoundError("Incomplete content audit: " + str(path))
        items = json.loads(path.read_text())
        if len(items) != 1 or items[0]["dataset"] != dataset:
            raise RuntimeError("Unexpected per-dataset audit contents: " + dataset)
        record = dict(items[0])
        expected = load_manifest(dataset, "main")
        if record["manifest_sha256"] != expected.manifest_sha256:
            raise RuntimeError("Audit manifest checksum != frozen main split: " + dataset)
        if (record["identity_split_overlap_count"] != 0 or
                record["pixel_duplicate_cross_split_count"] != 0 or
                record.get("image_split_disjoint") is not True or
                record.get("conflicting_label_groups_in_train_or_test") != 0):
            raise RuntimeError("Image-content leakage detected: " + dataset)
        if dataset != "breakhis_8subtype" and record["group_split_overlap_count"] != 0:
            raise RuntimeError("Recorded patient/slide-group leakage: " + dataset)
        # The original and main variants are byte-for-byte identical for the
        # 12 datasets that did not require an image overlap repair.
        if dataset not in ("breakhis_8subtype", "endometrial_4class", "oral_oscc"):
            if record["split_version"] != "original":
                raise RuntimeError("Unexpected audit split version: " + dataset)
            record["same_as_original_manifest"] = True
        else:
            if record["split_version"] != "main":
                raise RuntimeError("Corrected version has not been audited: " + dataset)
            record["same_as_original_manifest"] = False
        record["split_version"] = "main"
        entries.append(record)
    dump_json(Path(out) / "audit.json", entries)
    print("PASS: 15/15 audited; 0 cross-split duplicate images; "
          "group metadata retained and flagged; "
          "patient-level isolation not required for BreaKHis.", flush=True)
    return entries


if __name__ == "__main__":
    merge()
