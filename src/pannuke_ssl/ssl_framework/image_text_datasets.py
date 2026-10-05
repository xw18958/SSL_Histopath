"""Leakage-aware image-text pair inventories and immutable split manifests.

This module deliberately does not define the cross-modal alignment model.  The
image-only SSL encoders are not natively aligned with text, so a downstream
alignment/projection protocol must be explicitly frozen before retrieval
metrics are scientifically meaningful.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from pannuke_ssl.config import load_yaml
from pannuke_ssl.utils import atomic_json_dump
from .runtime_paths import expand_runtime_string, expand_runtime_paths


IMAGE_TEXT_DATASETS = ("arch", "ipath")
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _PROJECT_ROOT / "configs/ssl_standard/image_text_retrieval_datasets.yaml"
_MANIFEST_SCHEMA_VERSION = 2
_IMAGE_SUFFIXES = frozenset((".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"))


class ImageTextProtocolNotReadyError(RuntimeError):
    pass


def _configs() -> dict[str, dict[str, Any]]:
    document = load_yaml(_CONFIG_PATH)
    datasets = document.get("datasets") if isinstance(document, dict) else None
    if not isinstance(datasets, dict) or set(datasets) != set(IMAGE_TEXT_DATASETS):
        raise RuntimeError(f"Invalid image-text registry: {_CONFIG_PATH}")
    return {slug: dict(datasets[slug]) for slug in IMAGE_TEXT_DATASETS}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksum(path: Path, checksum: str) -> None:
    destination = Path(f"{path}.sha256")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=destination.parent, delete=False, encoding="utf-8") as handle:
        handle.write(f"{checksum}  {path.name}\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def _image_by_stem(directory: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            if path.stem in result:
                raise ValueError(f"Duplicate image stem in {directory}: {path.stem}")
            result[path.stem] = path
    return result


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _arch_records(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    empty_text: list[str] = []
    books_root = root / "extracted/books_set/books_set"
    books = json.loads((books_root / "captions.json").read_text(encoding="utf-8"))
    images = _image_by_stem(books_root / "images")
    book_rows: list[dict[str, Any]] = []
    for key, value in books.items():
        uuid = str(value["uuid"])
        path = images.get(uuid)
        if path is None:
            missing.append(f"books:{uuid}")
            continue
        text = str(value["caption"]).strip()
        if not text:
            empty_text.append(f"books:{uuid}")
            continue
        book_rows.append({
            "source": "books",
            "source_row": str(key),
            "relative_path": path.relative_to(root).as_posix(),
            "text": text,
            "figure_id": str(value.get("figure_id", "")),
            "uuid": uuid,
        })
    # Connected components merge rows sharing either figure ID or exact caption,
    # preventing both panel leakage and duplicated-caption leakage.
    uf = _UnionFind(len(book_rows))
    seen_figure: dict[str, int] = {}
    seen_caption: dict[str, int] = {}
    for i, row in enumerate(book_rows):
        for value, table in ((row["figure_id"], seen_figure), (row["text"], seen_caption)):
            if value in table:
                uf.union(i, table[value])
            else:
                table[value] = i
    roots: dict[int, int] = {}
    for i, row in enumerate(book_rows):
        root_id = uf.find(i)
        roots.setdefault(root_id, len(roots))
        records.append({**row, "group_id": f"books:{roots[root_id]:05d}"})

    pubmed_root = root / "extracted/pubmed_set/pubmed_set"
    pubmed = json.loads((pubmed_root / "captions.json").read_text(encoding="utf-8"))
    images = _image_by_stem(pubmed_root / "images")
    caption_to_group: dict[str, str] = {}
    for key, value in pubmed.items():
        uuid = str(value["uuid"])
        path = images.get(uuid)
        if path is None:
            missing.append(f"pubmed:{uuid}")
            continue
        text = str(value["caption"]).strip()
        if not text:
            empty_text.append(f"pubmed:{uuid}")
            continue
        group_id = caption_to_group.setdefault(text, f"pubmed:{len(caption_to_group):05d}")
        records.append({
            "source": "pubmed",
            "source_row": str(key),
            "relative_path": path.relative_to(root).as_posix(),
            "text": text,
            "uuid": uuid,
            "group_id": group_id,
        })
    groups = Counter(row["group_id"] for row in records)
    return records, {
        "caption_rows": len(books) + len(pubmed),
        "paired_rows": len(records),
        "missing_images": missing,
        "empty_text_rows": empty_text,
        "groups": len(groups),
        "multi_record_groups": sum(value > 1 for value in groups.values()),
        "max_group_size": max(groups.values()) if groups else 0,
        "books_caption_rows": len(books),
        "pubmed_caption_rows": len(pubmed),
    }


def _ipath_records(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    csv_path = root / "Descriptions.csv"
    if not csv_path.is_file():
        csv_path = root / "raw/Descriptions.csv"
    frame = pd.read_csv(csv_path)
    image_root = root / "Images"
    if not image_root.is_dir():
        image_root = root / "extracted/IPATH/Images"
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    empty_text: list[str] = []
    description_to_group: dict[str, str] = {}
    for source_row, row in enumerate(frame.to_dict("records")):
        image_id = str(row["Image_ID"])
        path = image_root / image_id
        if not path.is_file():
            missing.append(image_id)
            continue
        raw_text = row["Description"]
        text = "" if pd.isna(raw_text) else " ".join(str(raw_text).split())
        if not text:
            empty_text.append(image_id)
            continue
        group_id = description_to_group.setdefault(text, f"description:{len(description_to_group):05d}")
        records.append({
            "source": "ipath",
            "source_row": int(source_row),
            "relative_path": path.relative_to(root).as_posix(),
            "text": text,
            "sub_pathology_types": str(row["Sub_Pathology_Types"]),
            "image_id": image_id,
            "group_id": group_id,
        })
    groups = Counter(row["group_id"] for row in records)
    return records, {
        "description_rows": len(frame),
        "paired_rows": len(records),
        "unique_descriptions": len(description_to_group),
        "missing_images": missing,
        "empty_text_rows": empty_text,
        "groups": len(groups),
        "multi_record_groups": sum(value > 1 for value in groups.values()),
        "max_group_size": max(groups.values()) if groups else 0,
    }


def inspect_image_text_dataset(slug: str) -> dict[str, Any]:
    if slug not in IMAGE_TEXT_DATASETS:
        raise ValueError(f"Unknown image-text dataset {slug!r}")
    config = _configs()[slug]
    root_spec = str(config["root"])
    root = Path(expand_runtime_string(root_spec))
    records, provenance = (_arch_records(root) if slug == "arch" else _ipath_records(root))
    return {
        "dataset": slug,
        "root": str(root),
        "configured_train_size": int(config["train_size"]),
        "configured_val_size": int(config["val_size"]),
        "configured_test_size": int(config["test_size"]),
        "configured_target_total": int(config["target_total"]),
        "configured_split_ratio": list(config["split_ratio"]),
        "split_unit": str(config["split_unit"]),
        "alignment_protocol": config.get("alignment_protocol"),
        "retrieval_evaluation_ready": config.get("alignment_protocol") is not None,
        **provenance,
    }


def _select_groups_exact(group_sizes: Mapping[str, int], target: int, seed: int) -> set[str]:
    if target < 0:
        raise ValueError("Target record count cannot be negative")
    if target == 0:
        return set()
    items = list(group_sizes.items())
    random.Random(seed).shuffle(items)
    # DP stores only one predecessor per reachable record count.  Targets are
    # small (700 by current protocol), so this remains cheap even for IPATH.
    parent: dict[int, tuple[int, str]] = {0: (-1, "")}
    for group_id, size in items:
        if size > target:
            continue
        for current in sorted(tuple(parent), reverse=True):
            nxt = current + int(size)
            if nxt > target or nxt in parent:
                continue
            parent[nxt] = (current, group_id)
        if target in parent:
            break
    if target not in parent:
        raise ValueError(f"Cannot obtain exactly {target} records without breaking group boundaries")
    selected: set[str] = set()
    cursor = target
    while cursor:
        previous, group_id = parent[cursor]
        selected.add(group_id)
        cursor = previous
    return selected


def _group_split(
    records: list[dict[str, Any]],
    *,
    val_count: int,
    test_count: int,
    train_count: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["group_id"])].append(row)
    sizes = {group_id: len(rows) for group_id, rows in grouped.items()}
    test_groups = _select_groups_exact(sizes, test_count, seed + 2)
    remaining = {k: v for k, v in sizes.items() if k not in test_groups}
    val_groups = _select_groups_exact(remaining, val_count, seed + 1)
    remaining = {k: v for k, v in remaining.items() if k not in val_groups}
    if train_count is None:
        train_groups = set(remaining)
    else:
        train_groups = _select_groups_exact(remaining, train_count, seed)
    split_of = {group_id: "test" for group_id in test_groups}
    split_of.update({group_id: "val" for group_id in val_groups})
    split_of.update({group_id: "train" for group_id in train_groups})
    selected = [{**row, "split": split_of[row["group_id"]]} for row in records if row["group_id"] in split_of]
    counts = Counter(row["split"] for row in selected)
    expected_train = sum(remaining.values()) if train_count is None else train_count
    expected = {"train": expected_train, "val": val_count, "test": test_count}
    if dict(counts) != expected:
        raise AssertionError(f"Group split count mismatch: observed={dict(counts)}, expected={expected}")
    for group_id, rows in defaultdict(list, ((gid, []) for gid in split_of)).items():
        del rows
    if any(len({row["split"] for row in selected if row["group_id"] == group_id}) != 1 for group_id in split_of):
        raise AssertionError("Image-text group leakage detected")
    return sorted(selected, key=lambda x: (x["split"], x["group_id"], x["relative_path"])), expected


def prepare_image_text_manifest(
    slug: str,
    output_root: str | Path,
    *,
    val_count: int | None = None,
    test_count: int | None = None,
    train_count: int | None = None,
    seed: int = 20260903,
    allow_missing_images: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    if slug not in IMAGE_TEXT_DATASETS:
        raise ValueError(f"Unknown image-text dataset {slug!r}")
    config = _configs()[slug]
    root_spec = str(config["root"])
    root = Path(expand_runtime_string(root_spec))
    records, provenance = (_arch_records(root) if slug == "arch" else _ipath_records(root))
    missing = list(provenance.get("missing_images", ()))
    if missing and not allow_missing_images:
        raise ImageTextProtocolNotReadyError(
            f"{slug} has {len(missing)} metadata rows without a matched local image; "
            "resolve them or pass --allow-missing-images explicitly before freezing a split"
        )
    target_train = int(config["train_size"] if train_count is None else train_count)
    target_val = int(config["val_size"] if val_count is None else val_count)
    target_test = int(config["test_size"] if test_count is None else test_count)
    selected, split_counts = _group_split(
        records,
        val_count=target_val,
        test_count=target_test,
        train_count=target_train,
        seed=int(seed),
    )
    if train_count is None and val_count is None and test_count is None:
        expected_total = int(config["target_total"])
        if sum(split_counts.values()) != expected_total:
            raise AssertionError(
                f"Configured {slug} 7:1.5:1.5 split must total {expected_total}, got {split_counts}"
            )
    manifest = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "dataset": slug,
        "dataset_root_spec": root_spec,
        "seed": int(seed),
        "split_unit": config["split_unit"],
        "split_counts": split_counts,
        "group_leakage": False,
        "missing_image_rows_excluded": len(missing),
        "empty_text_rows_excluded": len(provenance.get("empty_text_rows", ())),
        "alignment_protocol": config.get("alignment_protocol"),
        "retrieval_evaluation_ready": config.get("alignment_protocol") is not None,
        "provenance": provenance,
        "records": selected,
    }
    path = Path(output_root) / "image_text_manifests" / f"{slug}.json"
    checksum = Path(f"{path}.sha256")
    if path.exists() or checksum.exists():
        if not force:
            raise FileExistsError(f"Manifest already exists: {path}")
        path.unlink(missing_ok=True)
        checksum.unlink(missing_ok=True)
    atomic_json_dump(manifest, path)
    digest = _sha256_file(path)
    _write_checksum(path, digest)
    return {
        "dataset": slug,
        "manifest_file": str(path),
        "manifest_sha256": digest,
        "split_counts": split_counts,
        "group_leakage": False,
        "alignment_protocol": config.get("alignment_protocol"),
        "retrieval_evaluation_ready": config.get("alignment_protocol") is not None,
    }


def load_image_text_manifest(slug: str, output_root: str | Path) -> dict[str, Any]:
    """Load and validate a frozen image-text manifest without touching held-out content."""
    if slug not in IMAGE_TEXT_DATASETS:
        raise ValueError(f"Unknown image-text dataset {slug!r}")
    config = _configs()[slug]
    path = Path(output_root) / "image_text_manifests" / f"{slug}.json"
    checksum_path = Path(f"{path}.sha256")
    if not path.is_file() or not checksum_path.is_file():
        raise FileNotFoundError(f"Missing frozen image-text manifest/checksum for {slug}: {path}")
    expected = checksum_path.read_text(encoding="utf-8").strip().split(maxsplit=1)[0]
    observed = _sha256_file(path)
    if expected != observed:
        raise RuntimeError(f"Image-text manifest checksum mismatch: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if int(document.get("schema_version", -1)) != _MANIFEST_SCHEMA_VERSION or document.get("dataset") != slug:
        raise ValueError("Image-text manifest identity/schema mismatch")
    if document.get("dataset_root_spec") != str(config["root"]):
        raise ValueError("Image-text manifest root-spec mismatch")
    if document.get("alignment_protocol") != config.get("alignment_protocol"):
        raise ValueError("Image-text manifest alignment protocol does not match the frozen config")
    records = document.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Image-text manifest has no records")
    seen=set(); groups={}; split_rows={split:[] for split in ("train","val","test")}
    root = Path(expand_runtime_string(str(config["root"])))
    for index,row in enumerate(records):
        split=str(row.get("split")); rel=str(row.get("relative_path","")); gid=str(row.get("group_id",""))
        if split not in split_rows or not rel or not gid or not str(row.get("text","")).strip():
            raise ValueError(f"Invalid image-text record at index {index}")
        if rel in seen: raise ValueError(f"Duplicate image-text image record: {rel}")
        seen.add(rel); groups.setdefault(gid,set()).add(split)
        if not (root/rel).is_file(): raise FileNotFoundError(root/rel)
        split_rows[split].append({**row,"record_index":index})
    leaking=[g for g,v in groups.items() if len(v)>1]
    if leaking: raise ValueError(f"Image-text group leakage detected: {leaking[:5]}")
    observed_counts={k:len(v) for k,v in split_rows.items()}
    if observed_counts != document.get("split_counts"):
        raise ValueError(f"Image-text split count mismatch: {observed_counts}")
    protocol=expand_runtime_paths(config.get("alignment_protocol"))
    return {"dataset":slug,"root":root,"manifest_path":path,"manifest_sha256":observed,
            "split_rows":split_rows,"split_counts":observed_counts,"alignment_protocol":protocol,
            "provenance":document.get("provenance",{})}
