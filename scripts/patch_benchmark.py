#!/usr/bin/env python3
"""Frozen-backbone pathology patch retrieval and prototype few-shot benchmark.

No SSL optimization, projector inference, probe training, checkpoint selection,
feature standardisation on test, or test-set hyperparameter tuning.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from pannuke_ssl.ssl_framework.external_datasets import (
    EXTERNAL_DATASETS, ExternalProbeDataset, load_external_manifest,
)
from pannuke_ssl.ssl_framework.runtime_paths import load_server_environment

METHODS = ("lejepa", "simplex_k64")
EPOCHS = (100, 150, 200, 250, 300)
SEED = 20260903
TRAIN_RUN = "ssl_standard/runs/ssl-lejepa-s20260903/pretrain_full/checkpoints"
SIMPLEX_RUN = ("retrieval_repair_20261006/original_state/"
               "ssl-simplex-k64-s20260903/artifacts/pretrain_full/checkpoints")
DEFAULT_OUT = ROOT / "outputs/patch_benchmark_v1"
SPLITS = ("train", "val", "test")
SHOTS = (1, 5, 10)
REPEATS = 100


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_json(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf8")).hexdigest()


def dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf8"
    ) as tmp:
        json.dump(value, tmp, indent=2, sort_keys=True, allow_nan=False)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def checkpoint_path(method: str, epoch: int) -> Path:
    if method not in METHODS or epoch not in EPOCHS:
        raise ValueError("Unsupported method / epoch")
    env = load_server_environment()
    run_root = Path(env["SSL_RUN_ROOT"])
    name = "ssl-lejepa-s20260903" if method == "lejepa" else "ssl-simplex-k64-s20260903"
    candidates = [
        run_root / "ssl_standard/runs" / name / "pretrain_full/checkpoints" / ("epoch_%s.pt" % epoch),
        run_root / "retrieval_repair_20261006/original_state" / name /
            "artifacts/pretrain_full/checkpoints" / ("epoch_%s.pt" % epoch),
    ]
    present = [path for path in candidates if path.is_file()]
    if not present:
        raise FileNotFoundError("Checkpoint absent on this server: %s" % candidates)
    if len(present) > 1:
        hashes = {sha_file(path) for path in present}
        if len(hashes) != 1:
            raise RuntimeError("Ambiguous differing copies of original checkpoint: %s" % present)
    return present[0]


def sha_quick(path: Path) -> str:
    return sha_file(path)


MAIN_VARIANTS = {
    "breakhis_8subtype": "breakhis_image_disjoint_v1",
    "endometrial_4class": "endometrial_4class_image_disjoint_v1",
    "oral_oscc": "oral_oscc_image_disjoint_v1",
}


def load_manifest(dataset: str, variant: str = "original") -> ExternalProbeDataset:
    load_server_environment()
    base = load_external_manifest(dataset, ROOT / "manifests/ssl_standard")
    if variant == "main":
        variant = MAIN_VARIANTS.get(dataset, "original")
    if variant == "original":
        return base
    allowed = (
        (dataset == "breakhis_8subtype" and variant in (
            "breakhis_patient_v1", "breakhis_image_disjoint_v1"))
        or (dataset in ("endometrial_4class", "oral_oscc") and
            variant == dataset + "_image_disjoint_v1")
    )
    if not allowed:
        raise ValueError("Unsupported dataset/split combination")
    path = ROOT / "manifests/patch_benchmark_v1" / (variant + ".json")
    payload = json.loads(path.read_text())
    if payload["parent_manifest_sha256"] != base.manifest_sha256:
        raise ValueError("Versioned split parent checksum mismatch")
    # Compare the exact source identifiers/classes, excluding only split and
    # (on the optional BreaKHis grouped branch) derived group_id.
    def image_key(r, *, extra_group=False):
        return sha_json({k: v for k, v in r.items()
                         if k != "split" and not (extra_group and k == "group_id")})
    source = collections.Counter(image_key(r) for r in base.manifest["records"])
    for row in payload["records"]:
        key = image_key(row, extra_group=variant == "breakhis_patient_v1")
        if source[key] < 1:
            raise ValueError("Versioned split changed/repeated an image or its class")
        source[key] -= 1
        if row["split"] not in SPLITS:
            raise ValueError("Invalid split")
    if variant in ("breakhis_patient_v1", "breakhis_image_disjoint_v1"):
        if len(payload["records"]) != len(base.manifest["records"]):
            raise ValueError("BreaKHis alternative split must retain all source images")
    else:
        if len(payload["records"]) >= len(base.manifest["records"]) or (
            payload.get("removed_records_count") !=
            len(base.manifest["records"]) - len(payload["records"])):
            raise ValueError("Invalid exact-image filtered version")
    if variant == "breakhis_patient_v1":
        groups = collections.defaultdict(set)
        for row in payload["records"]:
            groups[row["group_id"]].add(row["split"])
        if any(len(parts) > 1 for parts in groups.values()):
            raise ValueError("Optional BreaKHis patient split contains overlapping source groups")
    else:
        if payload.get("remaining_duplicate_content_groups_cross_split") != 0:
            raise ValueError("Image-level split was not verified duplicate-free")
        if variant not in ("breakhis_image_disjoint_v1",):
            groups = collections.defaultdict(set)
            for row in payload["records"]:
                if row.get("group_id") is not None:
                    groups[row["group_id"]].add(row["split"])
            if any(len(splits) != 1 for splits in groups.values()):
                raise ValueError("Filtered split violated source-group isolation")
    merged = {**base.manifest, **payload}
    # Keep preprocessing and source provenance from the validated, original
    # manifest. Correct selected counts for filtered evaluation subsets.
    counts = collections.Counter((r["class_name"], r["split"]) for r in payload["records"])
    merged["selected_class_counts"] = {
        cls: sum(counts[cls, split] for split in SPLITS) for cls in base.class_names
    }
    merged["selected_class_split_counts"] = {
        cls: {split: counts[cls, split] for split in SPLITS} for cls in base.class_names
    }
    merged["balancing"] = {**base.manifest.get("balancing", {}),
                           "benchmark_versioned_image_dedup": True,
                           "excluded_image_count": (len(base.manifest["records"]) -
                                                    len(payload["records"]))}
    return replace(base, manifest=merged, manifest_path=path,
                   manifest_sha256=sha_file(path))


def records_by_split(ds: ExternalProbeDataset):
    rows = ds.split_rows
    assert set(rows) == set(SPLITS) and all(rows[s] for s in SPLITS)
    return rows


def record_identity(row):
    storage = row.get("storage", "file")
    if storage == "file":
        return ("file", row["relative_path"])
    if storage == "parquet":
        return ("parquet", row["parquet_file"], int(row["row_group"]), int(row["row_in_group"]))
    if storage == "hdf5":
        return ("hdf5", row["hdf5_file"], int(row["hdf5_index"]))
    raise ValueError("Unknown image storage: %s" % storage)


def patient_breakhis(row):
    parts = Path(row["relative_path"]).parts
    if len(parts) < 4 or not parts[-3].startswith("SOB_"):
        raise ValueError("Cannot derive BreaKHis source grouping: %s" % row["relative_path"])
    return parts[-3]


def build_breakhis_variant(force=False):
    base = load_manifest("breakhis_8subtype")
    path = ROOT / "manifests/patch_benchmark_v1/breakhis_patient_v1.json"
    if path.exists() and not force:
        existing = json.loads(path.read_text())
        if existing.get("parent_manifest_sha256") != base.manifest_sha256:
            raise RuntimeError("Stale BreaKHis alternate split")
        return path
    rows = [dict(r) for r in base.manifest["records"]]
    byclass = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        p = patient_breakhis(r)
        r["group_id"] = p
        byclass[int(r["class_id"])][p].append(r)
    for cls, groups in sorted(byclass.items()):
        if len(groups) < 3:
            raise RuntimeError("Class %s has fewer than three source groups" % cls)
        # Independently of images or embeddings, partition each class's source
        # groups. For three groups, all three splits have one group.
        rng = np.random.default_rng(SEED + cls + 81001)
        ids = list(sorted(groups))
        rng.shuffle(ids)
        target_test = max(1, round(len(ids) * 0.1))
        target_val = max(1, round(len(ids) * 0.1))
        if len(ids) - target_test - target_val < 1:
            raise RuntimeError("Cannot retain training group")
        for i, patient in enumerate(ids):
            split = "test" if i < target_test else (
                "val" if i < target_test + target_val else "train"
            )
            for r in groups[patient]:
                r["split"] = split
    group_splits = collections.defaultdict(set)
    class_splits = collections.Counter()
    for r in rows:
        group_splits[r["group_id"]].add(r["split"])
        class_splits[(r["class_id"], r["split"])] += 1
    if any(len(v) != 1 for v in group_splits.values()):
        raise RuntimeError("Split contains overlapping source groups")
    if min(class_splits[(c, s)] for c in range(8) for s in SPLITS) < 10:
        raise RuntimeError("Patient split insufficient for 10-shot classification")
    payload = {
        "schema": "patch_benchmark_grouped_v1",
        "dataset": "breakhis_8subtype",
        "split_version": "breakhis_patient_v1",
        "grouping": "image parent folder; patient surrogate, source patient ID not independently verified",
        "parent_manifest_sha256": base.manifest_sha256,
        "class_names": list(base.class_names),
        "records": rows,
        "split_counts": {s: sum(r["split"] == s for r in rows) for s in SPLITS},
        "caution": "Grouped split is not class-balanced and is reported separately.",
    }
    dump_json(path, payload)
    return path


def audit_dataset(slug: str, *, variant="original", content_check=False, visual_scan=False):
    from PIL import Image
    import io
    ds = load_manifest(slug, variant)
    by = records_by_split(ds)
    identity_split = collections.defaultdict(set)
    group_split = collections.defaultdict(set)
    missing_groups = 0
    class_counts = collections.Counter()
    for split, rows in by.items():
        for r in rows:
            identity_split[record_identity(r)].add(split)
            grp = r.get("group_id")
            if grp is None and slug == "breakhis_8subtype":
                grp = patient_breakhis(r)
            if grp is None:
                missing_groups += 1
            else:
                group_split[str(grp)].add(split)
            class_counts[(r["class_id"], split)] += 1
    identity_overlaps = [repr(k) for k, s in identity_split.items() if len(s) > 1]
    group_overlaps = [g for g, s in group_split.items() if len(s) > 1]
    exact_hash_conflicts = []
    label_conflicts = []
    active_label_conflicts = []
    visual_candidates = []
    if content_check or visual_scan:
        # Content checks decode every record using the same source reader as
        # feature extraction; no image is sampled away silently.
        from patch_benchmark_reader import CachedBenchmarkImageDataset
        pixel_splits = collections.defaultdict(set)
        fingerprint_labels = collections.defaultdict(set)
        image_hash_splits = collections.defaultdict(set)
        # Audit in physical-storage order, not class/split order. This avoids
        # decoding each 400 MB HF-Parquet row group dozens of times on BACH.
        rows_for_scan = [(split, row) for split, rows in by.items() for row in rows]
        rows_for_scan.sort(key=lambda item: (
            item[1].get("storage", "file"),
            item[1].get("parquet_file", item[1].get("relative_path", item[1].get("hdf5_file", ""))),
            int(item[1].get("row_group", -1)),
            int(item[1].get("row_in_group", item[1].get("hdf5_index", -1))),
        ))
        loader = CachedBenchmarkImageDataset([r for _, r in rows_for_scan], ds.root)
        for split, row in rows_for_scan:
            im = loader._open_record(row)
            if content_check:
                fingerprint = hashlib.sha256(f"{im.width}x{im.height}|".encode() + im.tobytes()).hexdigest()
                pixel_splits[fingerprint].add(split)
                fingerprint_labels[fingerprint].add(int(row["class_id"]))
            if visual_scan:
                # Candidates are warnings only, never asserted to be identical.
                mono = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
                bits = (mono[:, :-1] > mono[:, 1:]).ravel()
                key = np.packbits(bits).tobytes().hex()
                image_hash_splits[key].add(split)
        exact_hash_conflicts = [k for k, s in pixel_splits.items() if len(s) > 1]
        label_conflicts = [k for k, y in fingerprint_labels.items() if len(y) > 1]
        active_label_conflicts = [k for k in label_conflicts if pixel_splits[k] & {"train", "test"}]
        visual_candidates = [k for k, s in image_hash_splits.items() if len(s) > 1]
    unknown = missing_groups > 0
    leakage = bool(identity_overlaps or group_overlaps or exact_hash_conflicts)
    return {
        "dataset": slug, "split_version": variant, "manifest_sha256": ds.manifest_sha256,
        "total_images": sum(map(len, by.values())),
        "split_counts": {s: len(by[s]) for s in SPLITS},
        "minimum_class_train": min(class_counts[c, "train"] for c in range(ds.num_classes)),
        "missing_source_group_ids": missing_groups,
        "groups_derived_from_paths": slug == "breakhis_8subtype" and variant == "original",
        "identity_split_overlap_count": len(identity_overlaps),
        "identity_split_overlap_samples": identity_overlaps[:12],
        "group_split_overlap_count": len(group_overlaps),
        "group_split_overlap_samples": group_overlaps[:12],
        "pixel_duplicate_cross_split_count": len(exact_hash_conflicts) if content_check else None,
        "same_image_different_label_groups": len(label_conflicts) if content_check else None,
        "conflicting_label_groups_in_train_or_test": len(active_label_conflicts) if content_check else None,
        "image_split_disjoint": (not identity_overlaps and not exact_hash_conflicts) if content_check else None,
        "patient_isolation_required": not (slug == "breakhis_8subtype" and variant in ("main", "original", "breakhis_image_disjoint_v1")),
        "visual_hash_cross_split_candidates": len(visual_candidates) if visual_scan else None,
        "source_independence": "overlap_detected" if leakage else (
            "unverified" if unknown else "group-disjoint-by-recorded-ID"
        ),
        "warning": "Source-group information absent; patient/slide independence cannot be verified" if unknown else (
            "Recorded/derived source-group overlap" if leakage else None
        ),
    }


def run_audit(out: Path, datasets, *, variant="main", content_check=False, visual_scan=False):
    rows = [audit_dataset(d, variant=variant, content_check=content_check, visual_scan=visual_scan)
            for d in datasets]
    dump_json(out / "audit.json", rows)
    print(json.dumps(rows, indent=2))
    return rows


def run_preflight(out: Path, datasets, *, hash_checkpoints=False):
    load_server_environment()
    entries = {}
    for method in METHODS:
        for epoch in EPOCHS:
            path = checkpoint_path(method, epoch)
            if not path.is_file() or path.stat().st_size < 1_000_000:
                raise FileNotFoundError("Missing/undersized source checkpoint: %s" % path)
            entries["%s_%s" % (method, epoch)] = {
                "path": str(path), "size_bytes": path.stat().st_size,
                "sha256": sha_file(path) if hash_checkpoints else None,
            }
    manifests = {}
    for dataset in datasets:
        ds = load_manifest(dataset, "main")
        r = audit_dataset(dataset, variant="main")
        if r["identity_split_overlap_count"] or r["minimum_class_train"] < 10:
            raise RuntimeError("Invalid evaluation manifest: %s" % dataset)
        manifests[dataset] = {
            "sha256": ds.manifest_sha256, "total": len(ds.manifest["records"]),
            "source_independence": r["source_independence"],
        }
    info = {
        "checkpoints": entries, "datasets": manifests,
        "expected_total_full_feature_jobs": len(METHODS) * len(EPOCHS) * len(datasets),
        "expected_retrieval_results": len(METHODS) * len(EPOCHS) * len(datasets),
        "expected_fewshot_settings": len(METHODS) * len(EPOCHS) * len(datasets) * len(SHOTS),
        "feature_source": "frozen backbone before projector; mean of final patch tokens",
    }
    dump_json(out / "preflight.json", info)
    print(json.dumps(info, indent=2))
    return info


def feature_path(out: Path, method: str, epoch: int, dataset: str,
                 *, variant="original", smoke_per_class=0):
    part = ("smoke_%d" % smoke_per_class) if smoke_per_class else "full"
    return out / "features" / part / variant / method / ("epoch_%d" % epoch) / (dataset + ".npz")


def selected_rows(ds, smoke_per_class=0):
    by = records_by_split(ds)
    if not smoke_per_class:
        return by
    chosen = {}
    for split, rows in by.items():
        cap = smoke_per_class if split == "train" else max(3, min(5, smoke_per_class))
        counts = collections.Counter()
        selected = []
        for r in rows:
            key = int(r["class_id"])
            if counts[key] < cap:
                selected.append(r)
                counts[key] += 1
        if len(counts) != ds.num_classes or min(counts.values()) < (10 if split == "train" else 1):
            raise ValueError("Smoke subset lacks required classes/10-shot capacity")
        chosen[split] = selected
    return chosen


def row_order_hash(by):
    return sha_json({
        split: [[int(r["record_index"]), int(r["class_id"]), record_identity(r)]
                for r in by[split]]
        for split in SPLITS
    })


def save_feature_cache(path: Path, arrays: dict, metadata: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".npz", dir=path.parent, delete=False
    ) as handle:
        tmp_path = Path(handle.name)
    try:
        np.savez_compressed(tmp_path, metadata=np.array(json.dumps(metadata, sort_keys=True)),
                            **arrays)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_feature_cache(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as obj:
        meta = json.loads(str(obj["metadata"].item()))
        data = {s: (np.array(obj[s + "_features"], dtype=np.float32),
                    np.array(obj[s + "_labels"], dtype=np.int64),
                    np.array(obj[s + "_indices"], dtype=np.int64)) for s in SPLITS}
    for s, (x, y, idx) in data.items():
        if x.ndim != 2 or len(x) != len(y) or len(x) != len(idx):
            raise RuntimeError("Invalid cache shapes %s %s" % (path, s))
        if not np.isfinite(x).all() or np.any(np.linalg.norm(x, axis=1) < 1e-12):
            raise RuntimeError("Invalid/degenerate cached features")
        if len(set(map(int, idx))) != len(idx):
            raise RuntimeError("Duplicate feature source indices")
    if len({data[s][0].shape[1] for s in SPLITS}) != 1:
        raise RuntimeError("Inconsistent feature dimensions")
    return meta, data


def extract_one(out, method_name, epoch, dataset, *, device_name="cuda:0",
                variant="original", smoke_per_class=0):
    import torch
    from patch_benchmark_reader import extract_frozen_features
    from pannuke_ssl.ssl_methods.registry import build_method
    from pannuke_ssl.ssl_framework.trainer import load_checkpoint

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for extraction")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    ds = load_manifest(dataset, variant)
    by = selected_rows(ds, smoke_per_class)
    expected_rows = row_order_hash(by)
    checkpoint = checkpoint_path(method_name, epoch)
    ckhash = sha_file(checkpoint)
    path = feature_path(out, method_name, epoch, dataset, variant=variant,
                        smoke_per_class=smoke_per_class)
    if path.exists():
        meta, existing = load_feature_cache(path)
        if (meta["checkpoint_sha256"] != ckhash or
                meta["manifest_sha256"] != ds.manifest_sha256 or
                meta["row_order_sha256"] != expected_rows or
                meta["smoke_per_class"] != smoke_per_class):
            raise RuntimeError("Stale/incompatible cache: %s; do not silently overwrite" % path)
        print("CACHE_OK", path, flush=True)
        return meta

    ck_cpu = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ck_cpu["framework"] != "ssl_standard_v1" or ck_cpu["epoch"] != epoch:
        raise RuntimeError("Checkpoint framework/epoch mismatch")
    expect = "lejepa" if method_name == "lejepa" else "simplex_sigreg_lejepa"
    if ck_cpu["method_name"] != expect:
        raise RuntimeError("Checkpoint method mismatch")
    config = ck_cpu["config"]
    if config["method"]["name"] != expect:
        raise RuntimeError("Stored checkpoint configuration mismatch")
    if method_name == "simplex_k64" and int(config["method"]["objective"]["simplex_components"]) != 64:
        raise RuntimeError("This is not the K=64 Simplex checkpoint")
    # Relocate only physical model storage for host-specific layouts.
    config["backbone"]["config_dir"] = str(Path(load_server_environment()["SSL_MODEL_ROOT"]) / "plip_model")
    config["downstream"]["num_workers"] = min(int(config["downstream"].get("num_workers", 4)), 4)
    config["downstream"]["batch_size"] = min(int(config["downstream"].get("batch_size", 128)), 128)
    mdl = build_method(config, device)
    loaded = load_checkpoint(mdl, checkpoint, device)
    if int(loaded["epoch"]) != epoch:
        raise RuntimeError("Incorrect loaded checkpoint epoch")
    encoder = mdl.encoder.eval().to(device)
    encoder.requires_grad_(False)
    if any(p.requires_grad for p in encoder.parameters()) or encoder.training:
        raise RuntimeError("Encoder is not strictly frozen")
    arrays = {}
    for split in SPLITS:
        with torch.inference_mode():
            x, labels, indices = extract_frozen_features(encoder, by[split], config, device, ds)
        if x.ndim != 2 or not torch.isfinite(x).all():
            raise RuntimeError("Invalid encoder representations")
        arrays[split + "_features"] = x.numpy().astype(np.float32)
        arrays[split + "_labels"] = labels.numpy().astype(np.int64)
        arrays[split + "_indices"] = indices.astype(np.int64)
        if not np.array_equal(arrays[split + "_indices"],
                              [r["record_index"] for r in by[split]]):
            raise RuntimeError("Data loader re-ordered evaluation samples")
        if not np.array_equal(arrays[split + "_labels"],
                              [r["class_id"] for r in by[split]]):
            raise RuntimeError("Data loader label mismatch")
    metadata = {
        "protocol": "patch_benchmark_v1",
        "dataset": dataset, "method": method_name, "epoch": epoch,
        "split_variant": variant, "smoke_per_class": smoke_per_class,
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": ckhash,
        "manifest_path": str(ds.manifest_path),
        "manifest_sha256": ds.manifest_sha256, "row_order_sha256": expected_rows,
        "class_names": list(ds.class_names),
        "feature_dim": int(arrays["train_features"].shape[1]),
        "split_sizes": {s: int(len(arrays[s + "_labels"])) for s in SPLITS},
        "encoder": "pre-projector backbone mean final patch tokens",
        "preprocessing": ds.manifest.get("preprocessing", {}),
        "frozen": True, "no_test_augmentation": True,
        "inference_bfloat16": True,
        "source_checkpoint_method": loaded["method_name"],
    }
    save_feature_cache(path, arrays, metadata)
    del mdl, encoder
    torch.cuda.empty_cache()
    print("EXTRACTED", method_name, epoch, dataset, str(path), flush=True)
    return metadata


def normalize(features):
    x = np.asarray(features, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("Feature contains zero vector")
    return x / norm


def retrieval_metrics(gallery, gallery_y, query, query_y, *, query_chunk=256):
    ref = normalize(gallery)
    q = normalize(query)
    gy = np.asarray(gallery_y, dtype=np.int64)
    qy = np.asarray(query_y, dtype=np.int64)
    if len(ref) < 5:
        raise ValueError("Retrieval requires gallery >=5")
    classes = sorted(set(map(int, qy)))
    if set(classes) - set(map(int, gy)):
        raise ValueError("Query contains classes absent from gallery")
    aps = []
    successes = {k: [] for k in (1, 3, 5)}
    votes = []
    for begin in range(0, len(q), query_chunk):
        sims = q[begin:begin+query_chunk] @ ref.T
        for row_i, scores in enumerate(sims):
            y = int(qy[begin+row_i])
            order = np.argsort(-scores, kind="stable")
            ranked_y = gy[order]
            relevant = ranked_y == y
            n_relevant = int(np.count_nonzero(relevant))
            if n_relevant == 0:
                raise ValueError("No relevant gallery examples")
            precision_at_hits = np.flatnonzero(relevant) + 1
            ap = np.mean(np.arange(1, n_relevant + 1, dtype=np.float64)
                         / precision_at_hits)
            aps.append((y, float(ap)))
            for k in successes:
                successes[k].append(bool(np.any(relevant[:k])))
            nearest = ranked_y[:5]
            vote_counts = collections.Counter(int(c) for c in nearest)
            maximum = max(vote_counts.values())
            tied = [c for c, n in vote_counts.items() if n == maximum]
            winner = min(tied, key=lambda c: (
                -float(np.sum(scores[order[:5]][nearest == c])), c
            ))
            votes.append(winner == y)
    ap_by_class = {c: float(np.mean([ap for y, ap in aps if y == c]))
                   for c in classes}
    return {
        "mAP": float(np.mean([ap for _, ap in aps])),
        "macro_mAP": float(np.mean(list(ap_by_class.values()))),
        "Acc@1": float(np.mean(successes[1])),
        "Acc@3": float(np.mean(successes[3])),
        "Acc@5": float(np.mean(successes[5])),
        "majority_vote_Acc@5": float(np.mean(votes)),
        "AP_by_class": ap_by_class,
        "num_gallery": len(ref), "num_queries": len(q),
    }


def support_selections(dataset, variant, train_indices, train_y, *, shots=SHOTS,
                       repeats=REPEATS, seed=SEED):
    labels = np.asarray(train_y, dtype=np.int64)
    idx = np.asarray(train_indices, dtype=np.int64)
    classes = sorted(set(map(int, labels)))
    if len(set(map(int, idx))) != len(idx):
        raise RuntimeError("Nonunique training record indices")
    per_class = {c: np.sort(idx[labels == c]) for c in classes}
    base_seed = int(sha_json([dataset, variant, seed])[:16], 16)
    output = {}
    for shot in shots:
        if any(len(v) < shot for v in per_class.values()):
            raise ValueError("Insufficient class capacity for %d shot" % shot)
        values = []
        for rep in range(repeats):
            rng = np.random.default_rng(np.random.SeedSequence(
                [base_seed & 0xffffffff, base_seed >> 32, int(shot), rep]
            ))
            support = []
            for cls in classes:
                support.extend(map(int, rng.choice(per_class[cls], shot, replace=False)))
            values.append(support)
        output[str(shot)] = values
    return output


def ci_summary(values):
    from scipy.stats import t
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError("Invalid few-shot scores")
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    radius = float(t.ppf(0.975, len(a)-1) * std / np.sqrt(len(a))) if len(a) > 1 else 0.0
    return {"mean": mean, "std": std, "ci95_lower": mean-radius,
            "ci95_upper": mean+radius, "n_support_subsets": len(a)}


def fewshot_metrics(train, train_y, train_indices, test, test_y, selections):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    tx = normalize(train)
    qx = normalize(test)
    train_y = np.asarray(train_y, dtype=np.int64)
    test_y = np.asarray(test_y, dtype=np.int64)
    idx_to_pos = {int(x): i for i, x in enumerate(train_indices)}
    classes = np.asarray(sorted(set(map(int, train_y))), dtype=np.int64)
    if set(map(int, test_y)) - set(map(int, classes)):
        raise ValueError("Test contains class absent from training")
    results = {}
    for shot_str, subsets in selections.items():
        scores = {"macro_f1": [], "balanced_accuracy": [], "accuracy": []}
        for chosen_ids in subsets:
            chosen = [idx_to_pos[int(i)] for i in chosen_ids]
            ys = train_y[chosen]
            if collections.Counter(map(int, ys)) != {int(c): int(shot_str) for c in classes}:
                raise RuntimeError("Support subset has an incorrect per-class count")
            prototypes = np.vstack([np.mean(tx[np.asarray(chosen)[ys == c]], axis=0)
                                    for c in classes])
            prototypes = normalize(prototypes)
            pred = classes[np.argmax(qx @ prototypes.T, axis=1)]
            scores["macro_f1"].append(float(f1_score(
                test_y, pred, labels=classes, average="macro", zero_division=0)))
            scores["balanced_accuracy"].append(float(balanced_accuracy_score(test_y, pred)))
            scores["accuracy"].append(float(accuracy_score(test_y, pred)))
        results[shot_str] = {
            "metrics": {name: ci_summary(vals) for name, vals in scores.items()},
            "per_subset": scores,
        }
    return results


def evaluate_one(out, method, epoch, dataset, *, variant="original",
                 smoke_per_class=0, repeats=REPEATS):
    ds = load_manifest(dataset, variant)
    by = selected_rows(ds, smoke_per_class)
    path = feature_path(out, method, epoch, dataset, variant=variant,
                        smoke_per_class=smoke_per_class)
    meta, arrays = load_feature_cache(path)
    if (meta["dataset"] != dataset or meta["method"] != method or meta["epoch"] != epoch
            or meta["manifest_sha256"] != ds.manifest_sha256
            or meta["row_order_sha256"] != row_order_hash(by)
            or meta["split_variant"] != variant or meta["smoke_per_class"] != smoke_per_class
            or not meta["frozen"] or not meta["no_test_augmentation"]
            or meta["encoder"] != "pre-projector backbone mean final patch tokens"):
        raise RuntimeError("Fairness/provenance check failed on feature cache")
    for s, (x, y, idx) in arrays.items():
        if not np.array_equal(idx, [r["record_index"] for r in by[s]]):
            raise RuntimeError("Feature rows differ from frozen manifest")
        if not np.array_equal(y, [r["class_id"] for r in by[s]]):
            raise RuntimeError("Feature labels differ from frozen manifest")
        if x.shape[1] != meta["feature_dim"]:
            raise RuntimeError("Dimension metadata mismatch")
    tx, ty, ti = arrays["train"]
    qx, qy, qi = arrays["test"]
    support = support_selections(dataset, variant, ti, ty, repeats=repeats)
    supp_sha = sha_json(support)
    retrieval = retrieval_metrics(tx, ty, qx, qy)
    fewshot = fewshot_metrics(tx, ty, ti, qx, qy, support)
    output = {
        "dataset": dataset, "method": method, "epoch": epoch,
        "split_variant": variant, "smoke_per_class": smoke_per_class,
        "feature_file": str(path), "feature_checkpoint_sha256": meta["checkpoint_sha256"],
        "manifest_sha256": meta["manifest_sha256"],
        "row_order_sha256": meta["row_order_sha256"],
        "feature_dim": meta["feature_dim"],
        "test_indices_sha256": sha_json(qi.tolist()),
        "train_indices_sha256": sha_json(ti.tolist()),
        "support_subsets_sha256": supp_sha, "support_subsets": support,
        "retrieval": retrieval, "fewshot": fewshot,
        "normalization": "per-image L2; no distribution-level normalization",
        "query_test_only": True, "gallery_train_only": True,
        "no_test_selection": True,
    }
    part = ("smoke_%d" % smoke_per_class) if smoke_per_class else "full"
    result_path = out / "results" / part / variant / method / ("epoch_%d" % epoch) / (dataset + ".json")
    if result_path.exists():
        original = json.loads(result_path.read_text())
        if (original["feature_checkpoint_sha256"] != output["feature_checkpoint_sha256"]
                or original["support_subsets_sha256"] != supp_sha):
            raise RuntimeError("Existing result conflicts with frozen input")
    dump_json(result_path, output)
    print("EVALUATED", method, epoch, dataset, result_path, flush=True)
    return output


def compare_and_aggregate(out: Path, datasets, *, variant="original",
                          smoke_per_class=0, partial=False):
    part = ("smoke_%d" % smoke_per_class) if smoke_per_class else "full"
    folder = out / "results" / part / variant
    found = {}
    missing = []
    for method in METHODS:
        for epoch in EPOCHS:
            for dataset in datasets:
                path = folder / method / ("epoch_%d" % epoch) / (dataset + ".json")
                if not path.exists():
                    missing.append("%s/%d/%s" % (method, epoch, dataset))
                    continue
                r = json.loads(path.read_text())
                assert (r["dataset"], r["method"], r["epoch"], r["split_variant"],
                        r["smoke_per_class"]) == (
                    dataset, method, epoch, variant, smoke_per_class)
                found[(method, epoch, dataset)] = r
    if missing and not partial:
        raise RuntimeError("Incomplete evaluation: %d missing: %s" % (len(missing), missing[:25]))
    comparison = []
    detailed = []
    metrics = ("mAP", "macro_mAP", "Acc@1", "Acc@3", "Acc@5", "majority_vote_Acc@5")
    for epoch in EPOCHS:
        for dataset in datasets:
            key0 = ("lejepa", epoch, dataset)
            key1 = ("simplex_k64", epoch, dataset)
            if key0 not in found or key1 not in found:
                continue
            a, b = found[key0], found[key1]
            # These checks enforce that neither model used a different query,
            # gallery, split, support selection, or number of classes.
            for field in ("manifest_sha256", "row_order_sha256", "test_indices_sha256",
                          "train_indices_sha256", "support_subsets_sha256",
                          "split_variant", "feature_dim"):
                if a[field] != b[field]:
                    raise RuntimeError("FAIRNESS FAILURE %s/%s: %s" % (dataset, epoch, field))
            if a["support_subsets"] != b["support_subsets"]:
                raise RuntimeError("Support image identity mismatch")
            for metric in metrics:
                comparison.append({
                    "dataset": dataset, "epoch": epoch, "task": "retrieval",
                    "setting": "all", "metric": metric,
                    "lejepa": float(a["retrieval"][metric]),
                    "simplex_k64": float(b["retrieval"][metric]),
                    "delta_pp": 100 * (float(b["retrieval"][metric]) -
                                       float(a["retrieval"][metric])),
                })
            for shot in SHOTS:
                tag = str(shot)
                for metric in ("macro_f1", "balanced_accuracy", "accuracy"):
                    ax = a["fewshot"][tag]["per_subset"][metric]
                    bx = b["fewshot"][tag]["per_subset"][metric]
                    if len(ax) != len(bx):
                        raise RuntimeError("Paired subset length mismatch")
                    delta = np.asarray(bx) - np.asarray(ax)
                    comparison.append({
                        "dataset": dataset, "epoch": epoch, "task": "fewshot",
                        "setting": tag + "-shot", "metric": metric,
                        "lejepa": float(np.mean(ax)), "simplex_k64": float(np.mean(bx)),
                        "delta_pp": float(100 * np.mean(delta)),
                        "paired_delta_ci95_pp": [
                            100 * ci_summary(delta)["ci95_lower"],
                            100 * ci_summary(delta)["ci95_upper"]
                        ],
                        "repeats": len(ax),
                        "lejepa_std": ci_summary(ax)["std"],
                        "simplex_k64_std": ci_summary(bx)["std"],
                    })
    # Equally weighted averages over entire datasets (not test images).
    grouped = collections.defaultdict(list)
    for r in comparison:
        grouped[(r["epoch"], r["task"], r["setting"], r["metric"])].append(r)
    summary = []
    for (epoch, task, setting, metric), rows in sorted(grouped.items()):
        avg_a = float(np.mean([r["lejepa"] for r in rows]))
        avg_b = float(np.mean([r["simplex_k64"] for r in rows]))
        direct_delta = 100 * (avg_b - avg_a)
        row_deltas = float(np.mean([r["delta_pp"] for r in rows]))
        if not np.isclose(direct_delta, row_deltas, atol=1e-8):
            raise RuntimeError("Aggregate difference recomputation failed")
        summary.append({
            "epoch": epoch, "task": task, "setting": setting, "metric": metric,
            "datasets_present": len(rows), "datasets_expected": len(datasets),
            "lejepa_mean": avg_a, "simplex_k64_mean": avg_b,
            "delta_pp": direct_delta,
        })
    if not partial and len(found) != 2 * len(EPOCHS) * len(datasets):
        raise AssertionError("Unexpected final coverage")
    audit_path = out / "audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.is_file() else []
    warning = {r["dataset"]: r["source_independence"] for r in audit
               if r["split_version"] == variant}
    for row in comparison:
        row["source_independence"] = warning.get(row["dataset"], "unverified")
    export = out / "reports" / part / variant
    export.mkdir(parents=True, exist_ok=True)
    dump_json(export / "comparison.json", comparison)
    dump_json(export / "summary.json", summary)
    dump_json(export / "verification.json", {
        "protocol": "patch_benchmark_v1", "split_variant": variant,
        "method_count": 2, "checkpoints_per_method": 5,
        "dataset_count": len(datasets), "expected_feature_jobs": 10 * len(datasets),
        "completed_jobs": len(found), "missing": missing,
        "comparison_rows": len(comparison), "aggregation_recomputed": True,
        "all_available_pairs_fairness_checked": True,
        "support_splits_shared_across_methods": True,
        "source_independence_warnings": warning,
        "final_complete": not bool(missing),
    })
    with (export / "comparison.csv").open("w", newline="") as handle:
        columns = ("dataset", "epoch", "task", "setting", "metric", "lejepa",
                   "simplex_k64", "delta_pp", "source_independence")
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in comparison:
            writer.writerow(r)
    with (export / "summary.csv").open("w", newline="") as handle:
        if summary:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    print(json.dumps({
        "jobs": len(found), "missing": len(missing),
        "paired_comparisons": len(comparison), "summary_rows": len(summary),
        "report_directory": str(export)
    }, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=(
        "preflight", "audit", "group-breakhis", "extract", "evaluate",
        "extract-evaluate", "aggregate"
    ))
    parser.add_argument("--dataset", choices=(*EXTERNAL_DATASETS, "all"), default="all")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--epoch", type=int, choices=EPOCHS)
    parser.add_argument("--variant", default="main", choices=(
        "main", "original", "breakhis_patient_v1", "breakhis_image_disjoint_v1"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-per-class", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--hash-checkpoints", action="store_true")
    parser.add_argument("--content-check", action="store_true")
    parser.add_argument("--visual-scan", action="store_true")
    parser.add_argument("--partial", action="store_true",
                        help="Allow incomplete matrix ONLY for smoke report")
    args = parser.parse_args()
    if args.repeats != REPEATS and args.smoke_per_class == 0:
        parser.error("Full results require exactly 100 predetermined support subsets")
    if args.partial and args.smoke_per_class == 0:
        parser.error("Full benchmark reporting must require complete matrix")
    if args.smoke_per_class and args.smoke_per_class < 10:
        parser.error("Smoke requires at least 10 training images/class")
    if args.action in ("extract", "evaluate", "extract-evaluate") and (
            args.method is None or args.epoch is None):
        parser.error("Feature/evaluation action requires --method and --epoch")
    datasets = list(EXTERNAL_DATASETS) if args.dataset == "all" else [args.dataset]
    if args.variant not in ("original", "main") and datasets != ["breakhis_8subtype"]:
        parser.error("Special BreaKHis variant requires --dataset breakhis_8subtype")
    if args.action == "group-breakhis":
        print(build_breakhis_variant())
    elif args.action == "preflight":
        run_preflight(args.out, datasets, hash_checkpoints=args.hash_checkpoints)
    elif args.action == "audit":
        run_audit(args.out, datasets, variant=args.variant, content_check=args.content_check,
                  visual_scan=args.visual_scan)
    elif args.action in ("extract", "evaluate", "extract-evaluate"):
        for dataset in datasets:
            if args.action in ("extract", "extract-evaluate"):
                extract_one(args.out, args.method, args.epoch, dataset,
                            device_name=args.device, variant=args.variant,
                            smoke_per_class=args.smoke_per_class)
            if args.action in ("evaluate", "extract-evaluate"):
                evaluate_one(args.out, args.method, args.epoch, dataset,
                             variant=args.variant,
                             smoke_per_class=args.smoke_per_class,
                             repeats=args.repeats)
    elif args.action == "aggregate":
        compare_and_aggregate(args.out, datasets, variant=args.variant,
                              smoke_per_class=args.smoke_per_class,
                              partial=args.partial)


if __name__ == "__main__":
    main()
