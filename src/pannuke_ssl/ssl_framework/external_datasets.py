"""Frozen, manifest-backed optional datasets for downstream linear probes.

PanNuke remains the framework default.  This module deliberately keeps the
other datasets outside the SSL/pretraining data path: callers must prepare a
split manifest explicitly, then evaluation verifies its sidecar checksum
before it opens an image.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from pannuke_ssl.config import load_yaml
from pannuke_ssl.utils import atomic_json_dump


PANNUKE_DATASET = "pannuke19"
EXTERNAL_DATASETS = ("crc_val_he_7k", "breakhis_8subtype")
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _PROJECT_ROOT / "configs/ssl_standard/external_probe_datasets.yaml"
_IMAGE_SUFFIXES = frozenset((".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"))
_SPLITS = ("train", "val", "test")
_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExternalDatasetConfig:
    slug: str
    root: Path
    class_names: tuple[str, ...]
    split_policy: str
    expected_images: int


@dataclass(frozen=True)
class ExternalProbeDataset:
    config: ExternalDatasetConfig
    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]

    @property
    def slug(self) -> str:
        return self.config.slug

    @property
    def root(self) -> Path:
        return self.config.root

    @property
    def class_names(self) -> tuple[str, ...]:
        return self.config.class_names

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def split_rows(self) -> dict[str, list[dict[str, Any]]]:
        by = {split: [] for split in _SPLITS}
        for record_index, record in enumerate(self.manifest["records"]):
            row = dict(record)
            row["record_index"] = record_index
            by[str(row["split"])].append(row)
        return by

    @property
    def split_counts(self) -> dict[str, int]:
        return {split: len(rows) for split, rows in self.split_rows.items()}

    def metadata(self) -> dict[str, Any]:
        return {
            "dataset": self.slug,
            "manifest_file": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "class_names": list(self.class_names),
            "num_classes": self.num_classes,
            "split_policy": self.config.split_policy,
            "split_counts": self.split_counts,
            "source_image_count": self.manifest["source_image_count"],
            "source_class_counts": self.manifest["source_class_counts"],
            "selected_class_counts": self.manifest["selected_class_counts"],
            "selected_class_split_counts": self.manifest["selected_class_split_counts"],
            "balancing": self.manifest["balancing"],
            "preprocessing": self.manifest["preprocessing"],
        }


def _dataset_configs() -> dict[str, ExternalDatasetConfig]:
    document = load_yaml(_CONFIG_PATH)
    datasets = document.get("datasets") if isinstance(document, dict) else None
    if not isinstance(datasets, dict) or set(datasets) != set(EXTERNAL_DATASETS):
        raise RuntimeError(f"Invalid external probe dataset registry: {_CONFIG_PATH}")
    result: dict[str, ExternalDatasetConfig] = {}
    for slug in EXTERNAL_DATASETS:
        value = datasets[slug]
        classes = tuple(str(item) for item in value["class_names"])
        if len(classes) < 2 or len(classes) != len(set(classes)):
            raise RuntimeError(f"External dataset {slug} has invalid class names")
        result[slug] = ExternalDatasetConfig(
            slug=slug,
            root=Path(value["root"]),
            class_names=classes,
            split_policy=str(value["split_policy"]),
            expected_images=int(value["expected_images"]),
        )
    return result


def external_dataset_config(slug: str, *, dataset_root: Path | None = None) -> ExternalDatasetConfig:
    if slug not in EXTERNAL_DATASETS:
        raise ValueError(f"Unknown optional downstream dataset {slug!r}; expected one of {EXTERNAL_DATASETS}")
    config = _dataset_configs()[slug]
    return config if dataset_root is None else ExternalDatasetConfig(
        slug=config.slug,
        root=Path(dataset_root),
        class_names=config.class_names,
        split_policy=config.split_policy,
        expected_images=config.expected_images,
    )


def manifest_directory(output_root: str | Path) -> Path:
    return Path(output_root) / "dataset_manifests"


def manifest_path(slug: str, output_root: str | Path) -> Path:
    if slug not in EXTERNAL_DATASETS:
        raise ValueError(f"Only optional datasets have manifests, not {slug!r}")
    return manifest_directory(output_root) / f"{slug}.json"


def manifest_checksum_path(path: str | Path) -> Path:
    return Path(f"{Path(path)}.sha256")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksum(path: Path, checksum: str) -> None:
    destination = manifest_checksum_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=destination.parent, delete=False, encoding="utf-8") as handle:
        handle.write(f"{checksum}  {path.name}\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def _image_paths(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES)


def _count_by_class(records: Iterable[Mapping[str, Any]], class_names: tuple[str, ...]) -> dict[str, int]:
    counts = {name: 0 for name in class_names}
    for record in records:
        counts[str(record["class_name"])] += 1
    return counts


def _count_by_class_split(records: Iterable[Mapping[str, Any]], class_names: tuple[str, ...]) -> dict[str, dict[str, int]]:
    counts = {name: {split: 0 for split in _SPLITS} for name in class_names}
    for record in records:
        counts[str(record["class_name"])][str(record["split"])] += 1
    return counts


def _stratified_image_records(config: ExternalDatasetConfig, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scanned: dict[str, list[Path]] = {}
    for class_name in config.class_names:
        paths = _image_paths(config.root / class_name)
        if not paths:
            raise FileNotFoundError(f"No images for CRC class {class_name}: {config.root / class_name}")
        scanned[class_name] = paths
    source_class_counts = {name: len(paths) for name, paths in scanned.items()}
    quota = min(source_class_counts.values())
    if quota < 3:
        raise ValueError("CRC minimum class count cannot support train/validation/test")
    train_count = int(math.floor(quota * 0.8))
    val_count = (quota - train_count) // 2
    test_count = quota - train_count - val_count
    if (train_count, val_count, test_count) != (271, 34, 34):
        raise ValueError(
            "CRC registered dataset must use its fixed 339-image class quota and 271/34/34 split; "
            f"found quota={quota}, split={(train_count, val_count, test_count)}"
        )
    records: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(config.class_names):
        shuffled = list(scanned[class_name])
        random.Random(seed + class_id).shuffle(shuffled)
        selected = shuffled[:quota]
        assignments = (
            ("train", selected[:train_count]),
            ("val", selected[train_count : train_count + val_count]),
            ("test", selected[train_count + val_count :]),
        )
        for split, rows in assignments:
            for path in rows:
                records.append({
                    "relative_path": path.relative_to(config.root).as_posix(),
                    "class_id": class_id,
                    "class_name": class_name,
                    "split": split,
                })
    ordered = sorted(records, key=lambda value: (value["split"], value["class_id"], value["relative_path"]))
    return ordered, {
        "source_image_count": sum(source_class_counts.values()),
        "source_class_counts": source_class_counts,
        "balancing": {
            "policy": "deterministic_equal_class_sample_before_split",
            "seed": seed,
            "per_class_quota": quota,
            "per_class_split_counts": {"train": train_count, "val": val_count, "test": test_count},
        },
    }


def _parse_breakhis_subtype(root: Path, path: Path, class_names: tuple[str, ...]) -> str:
    parts = path.relative_to(root).parts
    try:
        sob_index = parts.index("SOB")
        subtype = parts[sob_index + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"Cannot parse expected BreakHis SOB/subtype structure: {path}") from error
    if subtype not in class_names:
        raise ValueError(f"Unexpected BreakHis subtype {subtype!r}: {path}")
    return subtype


def _breakhis_records(config: ExternalDatasetConfig, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[Path]] = {name: [] for name in config.class_names}
    for path in sorted(config.root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        subtype = _parse_breakhis_subtype(config.root, path, config.class_names)
        grouped[subtype].append(path)
    source_class_counts = {name: len(paths) for name, paths in grouped.items()}
    quota = min(source_class_counts.values())
    if quota != 444:
        raise ValueError(f"BreakHis registered dataset must use its fixed 444-image subtype quota, found {quota}")
    train_count, val_count, test_count = 356, 44, 44
    records: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(config.class_names):
        shuffled = sorted(grouped[class_name])
        random.Random(seed + class_id).shuffle(shuffled)
        selected = shuffled[:quota]
        assignments = (
            ("train", selected[:train_count]),
            ("val", selected[train_count : train_count + val_count]),
            ("test", selected[train_count + val_count :]),
        )
        for split, paths in assignments:
            for path in paths:
                records.append({
                    "relative_path": path.relative_to(config.root).as_posix(),
                    "class_id": class_id,
                    "class_name": class_name,
                    "split": split,
                })
    ordered = sorted(records, key=lambda value: (value["split"], value["class_id"], value["relative_path"]))
    return ordered, {
        "source_image_count": sum(source_class_counts.values()),
        "source_class_counts": source_class_counts,
        "balancing": {
            "policy": "deterministic_equal_subtype_image_sample_before_split",
            "seed": seed,
            "per_class_quota": quota,
            "per_class_split_counts": {"train": train_count, "val": val_count, "test": test_count},
            "split_unit": "image",
            "patient_isolation_enforced": False,
        },
    }


def _manifest_payload(config: ExternalDatasetConfig, records: Iterable[Mapping[str, Any]], seed: int, provenance: Mapping[str, Any]) -> dict[str, Any]:
    rows = [dict(record) for record in records]
    counts = {split: sum(record["split"] == split for record in rows) for split in _SPLITS}
    source_count = int(provenance["source_image_count"])
    if source_count != config.expected_images:
        raise ValueError(
            f"{config.slug} scan found {source_count} images, expected {config.expected_images}; "
            "refusing to freeze an incomplete manifest"
        )
    class_split_counts = _count_by_class_split(rows, config.class_names)
    if any(len({class_split_counts[name][split] for name in config.class_names}) != 1 for split in _SPLITS):
        raise AssertionError("Every split must be class-balanced")
    return {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "dataset": config.slug,
        "dataset_root": str(config.root),
        "seed": int(seed),
        "class_names": list(config.class_names),
        "class_to_id": {name: index for index, name in enumerate(config.class_names)},
        "split_policy": config.split_policy,
        "split_counts": counts,
        "source_image_count": source_count,
        "source_class_counts": dict(provenance["source_class_counts"]),
        "selected_class_counts": _count_by_class(rows, config.class_names),
        "selected_class_split_counts": class_split_counts,
        "balancing": dict(provenance["balancing"]),
        "preprocessing": {
            "input": "raw_rgb_uint8",
            "resize": "resize_short_side_to_256_bicubic",
            "crop": "center_crop_256x256",
            "adapter_input": "raw_rgb_float_0_1_before_adapter_normalization",
        },
        "records": rows,
    }


def prepare_external_manifest(
    slug: str,
    output_root: str | Path,
    *,
    seed: int = 20260903,
    force: bool = False,
    dataset_root: Path | None = None,
) -> dict[str, Any]:
    """Explicitly scan an optional dataset and freeze its split manifest.

    Evaluation never calls this function.  Refusing an existing manifest by
    default avoids silently changing test membership during fast experiments.
    """
    config = external_dataset_config(slug, dataset_root=dataset_root)
    if not config.root.is_dir():
        raise FileNotFoundError(f"Optional dataset root does not exist: {config.root}")
    path = manifest_path(slug, output_root)
    checksum = manifest_checksum_path(path)
    if path.exists() or checksum.exists():
        if not force:
            raise FileExistsError(f"Manifest already exists: {path}; use --force only to deliberately replace it")
        path.unlink(missing_ok=True)
        checksum.unlink(missing_ok=True)
    records, provenance = _stratified_image_records(config, seed) if slug == "crc_val_he_7k" else _breakhis_records(config, seed)
    payload = _manifest_payload(config, records, seed, provenance)
    atomic_json_dump(payload, path)
    digest = _sha256_file(path)
    _write_checksum(path, digest)
    return {
        "dataset": slug,
        "manifest_file": str(path),
        "manifest_sha256": digest,
        "records": len(records),
        "split_counts": payload["split_counts"],
        "selected_class_split_counts": payload["selected_class_split_counts"],
        "split_policy": config.split_policy,
    }


def _validate_manifest_payload(config: ExternalDatasetConfig, document: Mapping[str, Any]) -> None:
    if int(document.get("schema_version", -1)) != _MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported external probe manifest schema")
    if document.get("dataset") != config.slug or Path(document.get("dataset_root", "")) != config.root:
        raise ValueError("External probe manifest dataset identity/root mismatch")
    if tuple(document.get("class_names", ())) != config.class_names:
        raise ValueError("External probe manifest class mapping mismatch")
    expected_map = {name: index for index, name in enumerate(config.class_names)}
    if document.get("class_to_id") != expected_map:
        raise ValueError("External probe manifest class-id mapping mismatch")
    if document.get("split_policy") != config.split_policy:
        raise ValueError("External probe manifest split policy mismatch")
    records = document.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("External probe manifest record count mismatch")
    if int(document.get("source_image_count", -1)) != config.expected_images:
        raise ValueError("External probe manifest source image count mismatch")
    seen_paths: set[str] = set()
    observed_counts = {split: 0 for split in _SPLITS}
    root_resolved = config.root.resolve()
    class_split_counts = {(class_id, split): 0 for class_id in range(len(config.class_names)) for split in _SPLITS}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("External probe manifest contains a non-object record")
        relative = record.get("relative_path")
        class_id = record.get("class_id")
        split = record.get("split")
        if not isinstance(relative, str) or relative in seen_paths:
            raise ValueError("External probe manifest contains duplicate or invalid paths")
        if not isinstance(class_id, int) or not 0 <= class_id < len(config.class_names):
            raise ValueError("External probe manifest contains an invalid class id")
        if record.get("class_name") != config.class_names[class_id] or split not in _SPLITS:
            raise ValueError("External probe manifest class/split mismatch")
        image_path = (config.root / relative).resolve()
        if root_resolved not in image_path.parents or not image_path.is_file():
            raise FileNotFoundError(f"External probe manifest image is absent or escapes its root: {relative}")
        seen_paths.add(relative)
        observed_counts[split] += 1
        class_split_counts[(class_id, split)] += 1
    if observed_counts != document.get("split_counts") or not all(observed_counts.values()):
        raise ValueError("External probe manifest split-count mismatch")
    if any(class_split_counts[(class_id, split)] == 0 for class_id in range(len(config.class_names)) for split in _SPLITS):
        raise ValueError("External probe manifest does not cover every class in every split")
    expected_balanced = document.get("selected_class_split_counts")
    observed_balanced = {
        name: {split: class_split_counts[(class_id, split)] for split in _SPLITS}
        for class_id, name in enumerate(config.class_names)
    }
    if expected_balanced != observed_balanced:
        raise ValueError("External probe manifest selected class/split count mismatch")
    if any(len({observed_balanced[name][split] for name in config.class_names}) != 1 for split in _SPLITS):
        raise ValueError("External probe manifest is not class-balanced within every split")


def load_external_manifest(slug: str, output_root: str | Path) -> ExternalProbeDataset:
    """Load a previously prepared manifest, rejecting a missing or modified one."""
    config = external_dataset_config(slug)
    path = manifest_path(slug, output_root)
    checksum_path = manifest_checksum_path(path)
    if not path.is_file() or not checksum_path.is_file():
        raise FileNotFoundError(
            f"Missing frozen {slug} manifest/checksum. Prepare it explicitly with "
            f"scripts/prepare_external_probe_manifest.py --dataset {slug}"
        )
    expected = checksum_path.read_text(encoding="utf-8").strip().split(maxsplit=1)[0]
    observed = _sha256_file(path)
    if expected != observed:
        raise RuntimeError(f"External probe manifest checksum mismatch: {path}")
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    _validate_manifest_payload(config, document)
    return ExternalProbeDataset(config=config, manifest_path=path, manifest_sha256=observed, manifest=document)


class ExternalProbeImageDataset(Dataset):
    """Read raw RGB images and deterministically map them to a 256px square."""

    def __init__(self, rows: list[Mapping[str, Any]], root: str | Path) -> None:
        self.rows = [dict(row) for row in rows]
        self.root = Path(root)

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _resize_center_crop(image: Image.Image) -> Image.Image:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions: {image.size}")
        scale = 256.0 / min(width, height)
        resized = image.resize(
            (max(256, int(math.floor(width * scale + 0.5))), max(256, int(math.floor(height * scale + 0.5)))),
            resample=Image.Resampling.BICUBIC,
        )
        left = (resized.width - 256) // 2
        top = (resized.height - 256) // 2
        return resized.crop((left, top, left + 256, top + 256))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        row = self.rows[index]
        path = self.root / str(row["relative_path"])
        with Image.open(path) as image:
            processed = self._resize_center_crop(image.convert("RGB"))
            pixels = np.asarray(processed, dtype=np.uint8).copy()
        tensor = torch.from_numpy(pixels).permute(2, 0, 1)
        return tensor, int(row["class_id"]), int(row["record_index"])
