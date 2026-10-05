"""Manifest-backed external pathology classification datasets.

The registry intentionally separates *planned* datasets from datasets whose
final protocol is ready.  A blocked dataset is still visible to tooling, but
manifest preparation/evaluation fails with its recorded blocker rather than
silently inventing a split, label, leakage policy, or preprocessing rule.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import random
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

from pannuke_ssl.config import load_yaml
from pannuke_ssl.utils import atomic_json_dump
from .runtime_paths import expand_runtime_string


PANNUKE_DATASET = "pannuke19"
EXTERNAL_DATASETS = (
    "mhist",
    "crc_val_he_7k",
    "breakhis_8subtype",
    "kather_2016",
    "bach",
    "sicapv2_4class",
    "wsss4luad_3class",
    "oral_oscc",
    "endometrial_4class",
    "osteosarcoma_3class",
    "gashissdb_binary",
    "renalcell_6class",
    "ebhi_seg_6class",
    "lc25000_5class",
    "pcam_binary",
)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _PROJECT_ROOT / "configs/ssl_standard/external_probe_datasets.yaml"
_IMAGE_SUFFIXES = frozenset((".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"))
_SPLITS = ("train", "val", "test")
_MANIFEST_SCHEMA_VERSION = 3


class DatasetProtocolNotReadyError(RuntimeError):
    """Raised when a planned dataset still lacks a defensible frozen protocol."""


@dataclass(frozen=True)
class ExternalDatasetConfig:
    # First five fields deliberately preserve the old positional constructor so
    # existing tests/tools can override a dataset root without code churn.
    slug: str
    root: Path
    class_names: tuple[str, ...]
    split_policy: str
    expected_images: int
    root_spec: str = ""
    builder: str = ""
    protocol_status: str = "ready"
    balance_policy: str = "balanced"
    tier: str = "main"
    blocker: str | None = None

    @property
    def ready(self) -> bool:
        return self.protocol_status == "ready"


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
            "dataset_root_spec": self.config.root_spec,
            "class_names": list(self.class_names),
            "num_classes": self.num_classes,
            "split_policy": self.config.split_policy,
            "balance_policy": self.config.balance_policy,
            "tier": self.config.tier,
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
        raise RuntimeError(
            f"Invalid external probe dataset registry: {_CONFIG_PATH}; "
            f"expected={EXTERNAL_DATASETS}, got={tuple(datasets or ())}"
        )
    result: dict[str, ExternalDatasetConfig] = {}
    for slug in EXTERNAL_DATASETS:
        value = datasets[slug]
        classes = tuple(str(item) for item in value["class_names"])
        if len(classes) < 2 or len(classes) != len(set(classes)):
            raise RuntimeError(f"External dataset {slug} has invalid class names")
        status = str(value.get("protocol_status", "ready"))
        if status not in {"ready", "blocked"}:
            raise RuntimeError(f"External dataset {slug} has invalid protocol_status={status!r}")
        balance = str(value.get("balance_policy", "balanced"))
        if balance not in {"balanced", "natural_imbalance", "group_constrained"}:
            raise RuntimeError(f"External dataset {slug} has invalid balance_policy={balance!r}")
        blocker = value.get("blocker")
        if status == "blocked" and not blocker:
            raise RuntimeError(f"Blocked dataset {slug} must record why its protocol is not ready")
        root_spec = str(value["root"])
        result[slug] = ExternalDatasetConfig(
            slug=slug,
            root=Path(expand_runtime_string(root_spec)),
            class_names=classes,
            split_policy=str(value["split_policy"]),
            expected_images=int(value["expected_images"]),
            root_spec=root_spec,
            builder=str(value.get("builder", slug)),
            protocol_status=status,
            balance_policy=balance,
            tier=str(value.get("tier", "main")),
            blocker=None if blocker is None else str(blocker),
        )
    return result


def external_dataset_config(slug: str, *, dataset_root: Path | None = None) -> ExternalDatasetConfig:
    if slug not in EXTERNAL_DATASETS:
        raise ValueError(f"Unknown optional downstream dataset {slug!r}; expected one of {EXTERNAL_DATASETS}")
    config = _dataset_configs()[slug]
    return config if dataset_root is None else replace(config, root=Path(dataset_root), root_spec=str(dataset_root))


def external_dataset_status() -> dict[str, dict[str, Any]]:
    """Return the planned suite and whether each final protocol is executable."""
    return {
        slug: {
            "status": config.protocol_status,
            "tier": config.tier,
            "builder": config.builder,
            "split_policy": config.split_policy,
            "balance_policy": config.balance_policy,
            "blocker": config.blocker,
        }
        for slug, config in _dataset_configs().items()
    }


def ready_external_datasets(*, tier: str | None = None) -> tuple[str, ...]:
    configs = _dataset_configs()
    return tuple(
        slug
        for slug in EXTERNAL_DATASETS
        if configs[slug].ready and (tier is None or configs[slug].tier == tier)
    )


def assert_dataset_ready(slug: str) -> ExternalDatasetConfig:
    config = external_dataset_config(slug)
    if not config.ready:
        raise DatasetProtocolNotReadyError(f"{slug}: {config.blocker}")
    return config


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


def _record(path: Path, root: Path, class_id: int, class_name: str, split: str) -> dict[str, Any]:
    return {
        "storage": "file",
        "relative_path": path.relative_to(root).as_posix(),
        "class_id": int(class_id),
        "class_name": str(class_name),
        "split": str(split),
    }


def _deterministic_take(paths: Iterable[Path], count: int, seed: int) -> list[Path]:
    values = sorted(paths)
    random.Random(seed).shuffle(values)
    if len(values) < count:
        raise ValueError(f"Requested {count} records from only {len(values)} available")
    return values[:count]


def _split_counts_8_1_1(total: int) -> dict[str, int]:
    """Largest-remainder 8:1:1 split with deterministic train/val/test tie order."""
    if total < 3:
        raise ValueError("Need at least three records for an 8:1:1 split")
    weights = (("train", 0.8), ("val", 0.1), ("test", 0.1))
    exact = {name: total * weight for name, weight in weights}
    counts = {name: int(math.floor(value)) for name, value in exact.items()}
    leftover = total - sum(counts.values())
    order = sorted((name for name, _ in weights), key=lambda name: (-(exact[name] - counts[name]), ("train", "val", "test").index(name)))
    for name in order[:leftover]:
        counts[name] += 1
    if sum(counts.values()) != total or any(value <= 0 for value in counts.values()):
        raise AssertionError(f"Invalid 8:1:1 split for total={total}: {counts}")
    return counts


def _crc_records(config: ExternalDatasetConfig, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scanned = {name: _image_paths(config.root / name) for name in config.class_names}
    if any(not values for values in scanned.values()):
        raise FileNotFoundError(f"CRC class directory missing/empty under {config.root}")
    source_class_counts = {name: len(paths) for name, paths in scanned.items()}
    quota = min(source_class_counts.values())
    if quota != 339:
        raise ValueError(f"CRC registered dataset requires minimum class count 339, found {quota}")
    train_count, val_count, test_count = 271, 34, 34
    records: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(config.class_names):
        selected = _deterministic_take(scanned[class_name], quota, seed + class_id)
        offsets = (("train", 0, train_count), ("val", train_count, train_count + val_count), ("test", train_count + val_count, quota))
        for split, lo, hi in offsets:
            records.extend(_record(path, config.root, class_id, class_name, split) for path in selected[lo:hi])
    return sorted(records, key=lambda x: (x["split"], x["class_id"], x["relative_path"])), {
        "source_image_count": sum(source_class_counts.values()),
        "source_class_counts": source_class_counts,
        "balancing": {
            "policy": "deterministic_equal_class_sample_before_split",
            "seed": seed,
            "per_class_quota": 339,
            "per_class_split_counts": {"train": 271, "val": 34, "test": 34},
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
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            grouped[_parse_breakhis_subtype(config.root, path, config.class_names)].append(path)
    source_class_counts = {name: len(paths) for name, paths in grouped.items()}
    quota = min(source_class_counts.values())
    if quota != 444:
        raise ValueError(f"BreakHis registered dataset requires minimum subtype count 444, found {quota}")
    train_count, val_count, test_count = 356, 44, 44
    records: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(config.class_names):
        selected = _deterministic_take(grouped[class_name], quota, seed + class_id)
        offsets = (("train", 0, train_count), ("val", train_count, train_count + val_count), ("test", train_count + val_count, quota))
        for split, lo, hi in offsets:
            records.extend(_record(path, config.root, class_id, class_name, split) for path in selected[lo:hi])
    return sorted(records, key=lambda x: (x["split"], x["class_id"], x["relative_path"])), {
        "source_image_count": sum(source_class_counts.values()),
        "source_class_counts": source_class_counts,
        "balancing": {
            "policy": "deterministic_equal_subtype_image_sample_before_split",
            "seed": seed,
            "per_class_quota": 444,
            "per_class_split_counts": {"train": 356, "val": 44, "test": 44},
            "split_unit": "image",
            "patient_isolation_enforced": False,
        },
    }


def _mhist_records(config: ExternalDatasetConfig, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    annotations = list(csv.DictReader((config.root / "annotations.csv").open(newline="", encoding="utf-8")))
    source_counts = Counter(str(row["Majority Vote Label"]) for row in annotations)
    if source_counts != Counter({"HP": 2162, "SSA": 990}):
        raise ValueError(f"Unexpected MHIST source counts: {dict(source_counts)}")
    grouped: dict[str, list[Path]] = {name: [] for name in config.class_names}
    for row in annotations:
        class_name = str(row["Majority Vote Label"])
        if class_name not in grouped:
            raise ValueError(f"Unexpected MHIST class {class_name!r}")
        path = config.root / "images" / str(row["Image Name"])
        if not path.is_file():
            raise FileNotFoundError(path)
        grouped[class_name].append(path)
    quota = min(1000, min(len(paths) for paths in grouped.values()))
    if quota != 990:
        raise ValueError(f"MHIST registered dataset requires min(1000, smallest class)=990, found {quota}")
    split_counts = _split_counts_8_1_1(quota)
    if split_counts != {"train": 792, "val": 99, "test": 99}:
        raise AssertionError(f"Unexpected MHIST split counts: {split_counts}")
    records: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(config.class_names):
        selected = _deterministic_take(grouped[class_name], quota, seed + class_id)
        train_end = split_counts["train"]
        val_end = train_end + split_counts["val"]
        slices = {
            "train": selected[:train_end],
            "val": selected[train_end:val_end],
            "test": selected[val_end:],
        }
        for split in _SPLITS:
            records.extend(_record(path, config.root, class_id, class_name, split) for path in slices[split])
    return sorted(records, key=lambda x: (x["split"], x["class_id"], x["relative_path"])), {
        "source_image_count": len(annotations),
        "source_class_counts": {name: int(source_counts[name]) for name in config.class_names},
        "balancing": {
            "policy": "min_1000_smallest_class_then_deterministic_8_1_1",
            "seed": seed,
            "per_class_quota": quota,
            "per_class_split_counts": split_counts,
            "oversampling": False,
        },
    }


def _xlsx_first_sheet_records(path: Path) -> list[dict[str, str]]:
    """Read a simple first-sheet XLSX table using only the Python stdlib.

    SICAPv2 partition workbooks are plain rectangular tables. Avoiding pandas
    Excel engines keeps the standard environment portable across both servers.
    """
    from xml.etree import ElementTree as ET

    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    ns = {"m": main_ns}
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", ns):
                shared.append("".join(node.text or "" for node in item.iter(f"{{{main_ns}}}t")))
        worksheet_names = sorted(
            name for name in archive.namelist()
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        )
        if not worksheet_names:
            raise ValueError(f"XLSX contains no worksheet: {path}")
        sheet = ET.fromstring(archive.read(worksheet_names[0]))

    def column_index(cell_ref: str) -> int:
        letters = "".join(ch for ch in cell_ref if ch.isalpha())
        value = 0
        for ch in letters.upper():
            value = value * 26 + (ord(ch) - ord("A") + 1)
        return value - 1

    raw_rows: list[dict[int, str]] = []
    for row in sheet.findall(".//m:sheetData/m:row", ns):
        values: dict[int, str] = {}
        for cell in row.findall("m:c", ns):
            index = column_index(str(cell.attrib.get("r", "A1")))
            cell_type = cell.attrib.get("t")
            if cell_type == "inlineStr":
                inline = cell.find("m:is", ns)
                value = "" if inline is None else "".join(
                    node.text or "" for node in inline.iter(f"{{{main_ns}}}t")
                )
            else:
                node = cell.find("m:v", ns)
                value = "" if node is None or node.text is None else node.text
                if cell_type == "s" and value:
                    value = shared[int(value)]
            values[index] = value
        raw_rows.append(values)
    if not raw_rows:
        raise ValueError(f"XLSX worksheet is empty: {path}")
    max_column = max(max(row, default=-1) for row in raw_rows)
    headers = [raw_rows[0].get(index, "") for index in range(max_column + 1)]
    return [
        {headers[index]: row.get(index, "") for index in range(len(headers)) if headers[index]}
        for row in raw_rows[1:]
        if any(row.get(index, "") for index in range(len(headers)))
    ]


def _primary_sicap_rows(path: Path, class_names: tuple[str, ...]) -> dict[str, list[str]]:
    grouped = {name: [] for name in class_names}
    for row in _xlsx_first_sheet_records(path):
        active = [name for name in class_names if int(float(row[name])) == 1]
        if len(active) != 1:
            raise ValueError(f"SICAPv2 primary class is not one-hot for {row.get('image_name')}: {active}")
        grouped[active[0]].append(str(row["image_name"]))
    return grouped

def _sicap_records(config: ExternalDatasetConfig, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    partition = config.root / "partition"
    split_tables = {
        "train": partition / "Validation/Val1/Train.xlsx",
        "val": partition / "Validation/Val1/Test.xlsx",
        "test": partition / "Test/Test.xlsx",
    }
    grouped_by_split = {split: _primary_sicap_rows(path, config.class_names) for split, path in split_tables.items()}
    quotas = {split: min(min(len(v) for v in grouped.values()), 1000) for split, grouped in grouped_by_split.items()}
    if quotas != {"train": 518, "val": 198, "test": 232}:
        raise ValueError(f"Unexpected SICAPv2 per-class split quotas: {quotas}")

    # Source counts are based on the non-overlapping official outer train/test.
    outer_train = _primary_sicap_rows(partition / "Test/Train.xlsx", config.class_names)
    outer_test = _primary_sicap_rows(partition / "Test/Test.xlsx", config.class_names)
    source_class_counts = {name: len(outer_train[name]) + len(outer_test[name]) for name in config.class_names}
    source_count = sum(source_class_counts.values())
    records: list[dict[str, Any]] = []
    selected_names: dict[str, set[str]] = {split: set() for split in _SPLITS}
    for split_index, split in enumerate(_SPLITS):
        for class_id, class_name in enumerate(config.class_names):
            names = sorted(grouped_by_split[split][class_name])
            random.Random(seed + split_index * 100 + class_id).shuffle(names)
            for filename in names[: quotas[split]]:
                image = config.root / "images" / filename
                if not image.is_file():
                    raise FileNotFoundError(image)
                if filename in selected_names[split]:
                    raise ValueError(f"Duplicate SICAPv2 selected image in {split}: {filename}")
                selected_names[split].add(filename)
                records.append(_record(image, config.root, class_id, class_name, split))
    if selected_names["train"] & selected_names["val"] or selected_names["train"] & selected_names["test"] or selected_names["val"] & selected_names["test"]:
        raise ValueError("SICAPv2 TRAIN/VAL/TEST image leakage detected")
    return sorted(records, key=lambda x: (x["split"], x["class_id"], x["relative_path"])), {
        "source_image_count": source_count,
        "source_class_counts": source_class_counts,
        "balancing": {
            "policy": "official_outer_test_plus_val1_train_val_balance_within_partition",
            "seed": seed,
            "per_class_split_counts": quotas,
            "per_class_selected_total": sum(quotas.values()),
            "g4c_used_as_separate_class": False,
            "official_test_preserved": True,
        },
    }


from . import protocol_builders as _pb

_BUILDERS = {
    "crc_val_he_7k": _crc_records,
    "breakhis_8subtype": _breakhis_records,
    "mhist_balanced_8_1_1": _mhist_records,
    "kather_2016": _pb.kather_2016,
    "bach": _pb.bach,
    "sicapv2_4class": _pb.sicap,
    "wsss4luad_3class": _pb.wsss,
    "oral_oscc": _pb.oral_oscc,
    "endometrial_4class": _pb.endometrial,
    "osteosarcoma_3class": _pb.osteosarcoma,
    "gashissdb_binary": _pb.gashis,
    "renalcell_6class": _pb.renalcell,
    "ebhi_seg_6class": _pb.ebhi,
    "lc25000_5class": _pb.lc25000,
    "pcam_binary": _pb.pcam,
}


def _manifest_payload(config: ExternalDatasetConfig, records: Iterable[Mapping[str, Any]], seed: int, provenance: Mapping[str, Any]) -> dict[str, Any]:
    rows = [dict(record) for record in records]
    counts = {split: sum(record["split"] == split for record in rows) for split in _SPLITS}
    source_count = int(provenance["source_image_count"])
    if source_count != config.expected_images:
        raise ValueError(
            f"{config.slug} source scan found {source_count} eligible images, expected {config.expected_images}; "
            "refusing to freeze an incomplete/changed manifest"
        )
    class_split_counts = _count_by_class_split(rows, config.class_names)
    if any(class_split_counts[name][split] == 0 for name in config.class_names for split in _SPLITS):
        raise AssertionError("Every class must be represented in every split")
    if config.balance_policy == "balanced" and any(
        len({class_split_counts[name][split] for name in config.class_names}) != 1 for split in _SPLITS
    ):
        raise AssertionError("Balanced dataset protocol requires every split to be class-balanced")
    return {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "dataset": config.slug,
        "dataset_root_spec": config.root_spec,
        "seed": int(seed),
        "class_names": list(config.class_names),
        "class_to_id": {name: index for index, name in enumerate(config.class_names)},
        "split_policy": config.split_policy,
        "balance_policy": config.balance_policy,
        "tier": config.tier,
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
    """Freeze an immutable classification manifest only when its protocol is ready."""
    config = external_dataset_config(slug, dataset_root=dataset_root)
    if not config.ready:
        raise DatasetProtocolNotReadyError(f"{slug}: {config.blocker}")
    if not config.root.is_dir():
        raise FileNotFoundError(f"Optional dataset root does not exist: {config.root}")
    builder_key = config.builder or config.slug  # backward-compatible test/root overrides
    builder = _BUILDERS.get(builder_key)
    if builder is None:
        raise DatasetProtocolNotReadyError(
            f"{slug}: protocol is marked ready but builder {builder_key!r} is not implemented"
        )
    path = manifest_path(slug, output_root)
    checksum = manifest_checksum_path(path)
    if path.exists() or checksum.exists():
        if not force:
            raise FileExistsError(f"Manifest already exists: {path}; use --force only to deliberately replace it")
        path.unlink(missing_ok=True)
        checksum.unlink(missing_ok=True)
    records, provenance = builder(config, seed)
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
        "balance_policy": config.balance_policy,
    }


def _record_identity(record: Mapping[str, Any]) -> str:
    storage = str(record.get("storage", "file"))
    if storage == "file":
        return "file:" + str(record.get("relative_path"))
    if storage == "parquet":
        return "parquet:" + ":".join(str(record.get(key)) for key in ("parquet_file", "row_group", "row_in_group"))
    if storage == "hdf5":
        return "hdf5:" + ":".join(str(record.get(key)) for key in ("hdf5_file", "hdf5_index"))
    raise ValueError(f"Unknown external record storage={storage!r}")


def _validate_manifest_payload(config: ExternalDatasetConfig, document: Mapping[str, Any]) -> None:
    if int(document.get("schema_version", -1)) != _MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported external probe manifest schema")
    if document.get("dataset") != config.slug or document.get("dataset_root_spec", "") != config.root_spec:
        raise ValueError("External probe manifest dataset identity/root-spec mismatch")
    if tuple(document.get("class_names", ())) != config.class_names:
        raise ValueError("External probe manifest class mapping mismatch")
    expected_map = {name: index for index, name in enumerate(config.class_names)}
    if document.get("class_to_id") != expected_map:
        raise ValueError("External probe manifest class-id mapping mismatch")
    if document.get("split_policy") != config.split_policy or document.get("balance_policy") != config.balance_policy:
        raise ValueError("External probe manifest protocol mismatch")
    records = document.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("External probe manifest record count mismatch")
    if int(document.get("source_image_count", -1)) != config.expected_images:
        raise ValueError("External probe manifest source image count mismatch")
    seen: set[str] = set()
    observed_counts = {split: 0 for split in _SPLITS}
    root_resolved = config.root.resolve()
    class_split_counts = {(class_id, split): 0 for class_id in range(len(config.class_names)) for split in _SPLITS}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("External probe manifest contains a non-object record")
        identity = _record_identity(record)
        if identity in seen:
            raise ValueError("External probe manifest contains duplicate records")
        seen.add(identity)
        class_id = record.get("class_id")
        split = record.get("split")
        if not isinstance(class_id, int) or not 0 <= class_id < len(config.class_names):
            raise ValueError("External probe manifest contains an invalid class id")
        if record.get("class_name") != config.class_names[class_id] or split not in _SPLITS:
            raise ValueError("External probe manifest class/split mismatch")
        storage = str(record.get("storage", "file"))
        if storage == "file":
            relative = record.get("relative_path")
            if not isinstance(relative, str):
                raise ValueError("Filesystem record is missing relative_path")
            image_path = (config.root / relative).resolve()
            if root_resolved not in image_path.parents or not image_path.is_file():
                raise FileNotFoundError(f"External probe manifest image is absent or escapes its root: {relative}")
        elif storage == "parquet":
            relative = record.get("parquet_file")
            if not isinstance(relative, str):
                raise ValueError("Parquet record is missing parquet_file")
            parquet_path = (config.root / relative).resolve()
            if root_resolved not in parquet_path.parents or not parquet_path.is_file():
                raise FileNotFoundError(f"External parquet shard is absent or escapes its root: {relative}")
            if int(record.get("row_group", -1)) < 0 or int(record.get("row_in_group", -1)) < 0:
                raise ValueError("Parquet record has invalid row coordinates")
        elif storage == "hdf5":
            relative = record.get("hdf5_file")
            if not isinstance(relative, str):
                raise ValueError("HDF5 record is missing hdf5_file")
            hdf5_path = (config.root / relative).resolve()
            if root_resolved not in hdf5_path.parents or not hdf5_path.is_file():
                raise FileNotFoundError(f"External HDF5 source is absent or escapes its root: {relative}")
            if int(record.get("hdf5_index", -1)) < 0:
                raise ValueError("HDF5 record has invalid index")
        else:
            raise ValueError(f"Unknown record storage={storage!r}")
        observed_counts[str(split)] += 1
        class_split_counts[(class_id, str(split))] += 1
    group_splits: dict[str, set[str]] = {}
    for record in records:
        if record.get("group_id") is not None:
            group_splits.setdefault(str(record["group_id"]), set()).add(str(record["split"]))
    leaking = [group for group, splits in group_splits.items() if len(splits) > 1]
    if leaking:
        raise ValueError(f"External probe manifest group leakage detected: {leaking[:5]}")
    if observed_counts != document.get("split_counts") or not all(observed_counts.values()):
        raise ValueError("External probe manifest split-count mismatch")
    if any(class_split_counts[(class_id, split)] == 0 for class_id in range(len(config.class_names)) for split in _SPLITS):
        raise ValueError("External probe manifest does not cover every class in every split")
    observed = {
        name: {split: class_split_counts[(class_id, split)] for split in _SPLITS}
        for class_id, name in enumerate(config.class_names)
    }
    if document.get("selected_class_split_counts") != observed:
        raise ValueError("External probe manifest selected class/split count mismatch")
    if config.balance_policy == "balanced" and any(
        len({observed[name][split] for name in config.class_names}) != 1 for split in _SPLITS
    ):
        raise ValueError("External probe manifest is not class-balanced within every split")


def load_external_manifest(slug: str, output_root: str | Path) -> ExternalProbeDataset:
    config = assert_dataset_ready(slug)
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
    """Read filesystem or HF-Parquet RGB records and map them to 256x256."""

    def __init__(self, rows: list[Mapping[str, Any]], root: str | Path) -> None:
        self.rows = [dict(row) for row in rows]
        self.root = Path(root)
        self._parquet_handles: dict[Path, pq.ParquetFile] = {}
        self._hdf5_handles: dict[Path, Any] = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_parquet_handles"] = {}
        state["_hdf5_handles"] = {}
        return state

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

    def _open_record(self, row: Mapping[str, Any]) -> Image.Image:
        storage = str(row.get("storage", "file"))
        if storage == "file":
            with Image.open(self.root / str(row["relative_path"])) as image:
                return image.convert("RGB").copy()
        if storage == "parquet":
            path = self.root / str(row["parquet_file"])
            handle = self._parquet_handles.get(path)
            if handle is None:
                handle = pq.ParquetFile(path)
                self._parquet_handles[path] = handle
            column = handle.read_row_group(int(row["row_group"]), columns=["image"])["image"]
            value = column[int(row["row_in_group"])].as_py()
            encoded = value.get("bytes") if isinstance(value, Mapping) else value
            if encoded is None and isinstance(value, Mapping) and value.get("path"):
                with Image.open(value["path"]) as image:
                    return image.convert("RGB").copy()
            if encoded is None:
                raise ValueError(f"HF-Parquet image record has neither bytes nor path: {path}")
            with Image.open(io.BytesIO(encoded)) as image:
                return image.convert("RGB").copy()
        if storage == "hdf5":
            import h5py
            path = self.root / str(row["hdf5_file"])
            handle = self._hdf5_handles.get(path)
            if handle is None:
                handle = h5py.File(path, "r")
                self._hdf5_handles[path] = handle
            key = next(iter(handle.keys()))
            array = np.asarray(handle[key][int(row["hdf5_index"])], dtype=np.uint8)
            return Image.fromarray(array).convert("RGB")
        raise ValueError(f"Unknown external record storage={storage!r}")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        row = self.rows[index]
        image = self._open_record(row)
        processed = self._resize_center_crop(image)
        pixels = np.asarray(processed, dtype=np.uint8).copy()
        tensor = torch.from_numpy(pixels).permute(2, 0, 1)
        return tensor, int(row["class_id"]), int(row["record_index"])
