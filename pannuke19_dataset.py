"""Efficient PyTorch access to the balanced PanNuke Parquet classification set."""

from __future__ import annotations

import csv
import io
import os
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


SAMPLE_ID_PATTERN = re.compile(r"^fold([123])_(\d{4})$")
CSV_FIELDS = ("fold", "sample_index", "tissue_label", "class_id", "split")


@dataclass(frozen=True)
class SourceLocation:
    shard: Path
    row_group: int
    row_in_group: int


@dataclass(frozen=True)
class SourceRecord:
    fold: int
    sample_index: int
    tissue_label: str
    class_id: int
    location: SourceLocation


def _parse_sample_id(sample_id: str) -> tuple[int, int]:
    match = SAMPLE_ID_PATTERN.fullmatch(sample_id)
    if not match:
        raise ValueError(f"Unexpected PanNuke sample_id: {sample_id!r}")
    return int(match.group(1)), int(match.group(2))


def _parquet_files(data_root: Path) -> list[Path]:
    files = sorted(Path(data_root).glob("fold*-of-*.parquet"))
    if len(files) != 6:
        raise FileNotFoundError(f"Expected six PanNuke shards in {data_root}, found {len(files)}")
    return files


def build_source_index(data_root: Path) -> dict[tuple[int, int], SourceRecord]:
    """Index all source rows without reading their image payloads."""
    records: dict[tuple[int, int], SourceRecord] = {}
    class_names: dict[int, str] = {}
    for shard in _parquet_files(data_root):
        parquet = pq.ParquetFile(shard)
        for row_group in range(parquet.num_row_groups):
            table = parquet.read_row_group(
                row_group, columns=["fold", "sample_id", "tissue", "tissue_name"]
            )
            for row_in_group, row in enumerate(table.to_pylist()):
                fold_from_id, sample_index = _parse_sample_id(row["sample_id"])
                fold = int(row["fold"])
                class_id = int(row["tissue"])
                tissue_label = str(row["tissue_name"])
                if fold != fold_from_id:
                    raise ValueError(
                        f"Fold mismatch in {shard.name}: field={fold}, sample_id={row['sample_id']}"
                    )
                if not 0 <= class_id < 19:
                    raise ValueError(f"Out-of-range tissue ID {class_id} in {row['sample_id']}")
                previous_name = class_names.setdefault(class_id, tissue_label)
                if previous_name != tissue_label:
                    raise ValueError(f"Class ID {class_id} has conflicting labels")
                key = (fold, sample_index)
                if key in records:
                    raise ValueError(f"Duplicate source key {key}")
                records[key] = SourceRecord(
                    fold=fold,
                    sample_index=sample_index,
                    tissue_label=tissue_label,
                    class_id=class_id,
                    location=SourceLocation(shard, row_group, row_in_group),
                )
    if len(records) != 7901:
        raise ValueError(f"Expected 7901 source records, found {len(records)}")
    if set(class_names) != set(range(19)):
        raise ValueError(f"Expected 19 tissue classes, found {sorted(class_names)}")
    return records


def read_metadata(metadata_csv: Path) -> list[dict[str, object]]:
    with Path(metadata_csv).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(f"Metadata columns must be exactly {CSV_FIELDS}; got {reader.fieldnames}")
        rows: list[dict[str, object]] = []
        for row in reader:
            rows.append(
                {
                    "fold": int(row["fold"]),
                    "sample_index": int(row["sample_index"]),
                    "tissue_label": row["tissue_label"],
                    "class_id": int(row["class_id"]),
                    "split": row["split"],
                }
            )
    return rows


def verify_records(rows: Iterable[Mapping[str, object]], index: Mapping[tuple[int, int], SourceRecord]) -> None:
    """Verify every metadata row preserves source fold/index/label alignment."""
    seen: set[tuple[int, int]] = set()
    for row in rows:
        key = (int(row["fold"]), int(row["sample_index"]))
        if key in seen:
            raise ValueError(f"Duplicate metadata source key {key}")
        seen.add(key)
        source = index.get(key)
        if source is None:
            raise ValueError(f"Metadata key absent from source: {key}")
        if source.tissue_label != str(row["tissue_label"]) or source.class_id != int(row["class_id"]):
            raise ValueError(f"Source/metadata label mismatch for {key}")


def _image_to_array(value: object) -> np.ndarray:
    if isinstance(value, Mapping):
        encoded = value.get("bytes")
    else:
        encoded = value
    if encoded is None:
        raise ValueError("Parquet image field has no embedded bytes")
    with Image.open(io.BytesIO(encoded)) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if array.shape != (256, 256, 3):
        raise ValueError(f"Expected RGB 256x256 image, got {array.shape}")
    return array


class PanNukeParquetDataset(Dataset[tuple[torch.Tensor, int]]):
    """Classification dataset keyed by original PanNuke ``(fold, sample_index)``."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, object]],
        source_index: Mapping[tuple[int, int], SourceRecord],
        transform: Callable[[np.ndarray], object] | None = None,
        image_cache: Mapping[tuple[int, int], np.ndarray] | None = None,
        decode_cache_size: int = 256,
    ) -> None:
        verify_records(rows, source_index)
        self.rows = [dict(row) for row in rows]
        self.source_index = dict(source_index)
        self.transform = transform
        self.image_cache = image_cache
        self.decode_cache_size = decode_cache_size
        self._parquet_handles: dict[Path, pq.ParquetFile] = {}
        self._decoded_lru: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.rows)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_parquet_handles"] = {}
        state["_decoded_lru"] = OrderedDict()
        return state

    def _handle(self, path: Path) -> pq.ParquetFile:
        handle = self._parquet_handles.get(path)
        if handle is None:
            handle = pq.ParquetFile(path)
            self._parquet_handles[path] = handle
        return handle

    def _read_uncached_image(self, key: tuple[int, int]) -> np.ndarray:
        cached = self._decoded_lru.get(key)
        if cached is not None:
            self._decoded_lru.move_to_end(key)
            return cached
        location = self.source_index[key].location
        table = self._handle(location.shard).read_row_group(location.row_group, columns=["image"])
        image = _image_to_array(table.column("image")[location.row_in_group].as_py())
        self._decoded_lru[key] = image
        if len(self._decoded_lru) > self.decode_cache_size:
            self._decoded_lru.popitem(last=False)
        return image

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        row = self.rows[item]
        key = (int(row["fold"]), int(row["sample_index"]))
        image = self.image_cache[key] if self.image_cache is not None else self._read_uncached_image(key)
        if self.transform is not None:
            transformed = self.transform(image.copy())
            if isinstance(transformed, torch.Tensor):
                tensor = transformed
            else:
                tensor = torch.as_tensor(np.asarray(transformed)).permute(2, 0, 1).contiguous()
        else:
            tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).contiguous()
        return tensor, int(row["class_id"])


def preload_selected_images(
    rows: Sequence[Mapping[str, object]], source_index: Mapping[tuple[int, int], SourceRecord]
) -> dict[tuple[int, int], np.ndarray]:
    """Decode only selected images into RAM; no image data is persisted."""
    by_group: dict[tuple[Path, int], list[tuple[tuple[int, int], int]]] = defaultdict(list)
    for row in rows:
        key = (int(row["fold"]), int(row["sample_index"]))
        location = source_index[key].location
        by_group[(location.shard, location.row_group)].append((key, location.row_in_group))

    cache: dict[tuple[int, int], np.ndarray] = {}
    for (shard, row_group), needed in by_group.items():
        table = pq.ParquetFile(shard).read_row_group(row_group, columns=["image"])
        image_column = table.column("image")
        for key, row_in_group in needed:
            image = _image_to_array(image_column[row_in_group].as_py())
            image.setflags(write=False)
            cache[key] = image
    if len(cache) != len(rows):
        raise RuntimeError("RAM cache does not contain every selected image")
    return cache


def build_dataloaders(
    metadata_csv: Path,
    data_root: Path,
    *,
    batch_size: int = 64,
    num_workers: int | None = None,
    cache_in_ram: bool = True,
    train_transform: Callable[[np.ndarray], object] | None = None,
    eval_transform: Callable[[np.ndarray], object] | None = None,
) -> dict[str, DataLoader]:
    """Build train/validation/test loaders with GPU-friendly defaults."""
    rows = read_metadata(metadata_csv)
    index = build_source_index(data_root)
    verify_records(rows, index)
    split_rows = {split: [row for row in rows if row["split"] == split] for split in ("train", "val", "test")}
    if {split: len(part) for split, part in split_rows.items()} != {"train": 2052, "val": 247, "test": 247}:
        raise ValueError("Metadata does not have the required fixed split sizes")

    image_cache = preload_selected_images(rows, index) if cache_in_ram else None
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    datasets = {
        "train": PanNukeParquetDataset(split_rows["train"], index, train_transform, image_cache),
        "val": PanNukeParquetDataset(split_rows["val"], index, eval_transform, image_cache),
        "test": PanNukeParquetDataset(split_rows["test"], index, eval_transform, image_cache),
    }
    common: dict[str, object] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        common["prefetch_factor"] = 2
    return {
        split: DataLoader(dataset, shuffle=(split == "train"), **common)
        for split, dataset in datasets.items()
    }
