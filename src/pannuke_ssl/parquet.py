from __future__ import annotations

import csv
import io
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset


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


def build_source_index(data_root: Path) -> dict[tuple[int, int], SourceRecord]:
    shards = sorted(Path(data_root).glob("fold*-of-*.parquet"))
    if len(shards) != 6:
        raise FileNotFoundError(f"Expected six PanNuke shards in {data_root}, found {len(shards)}")
    records: dict[tuple[int, int], SourceRecord] = {}
    class_names: dict[int, str] = {}
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        for row_group in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group, columns=["fold", "sample_id", "tissue", "tissue_name"])
            for row_in_group, row in enumerate(table.to_pylist()):
                parsed_fold, sample_index = _parse_sample_id(row["sample_id"])
                fold = int(row["fold"])
                class_id = int(row["tissue"])
                tissue_label = str(row["tissue_name"])
                if fold != parsed_fold:
                    raise ValueError(f"Fold mismatch for {row['sample_id']}")
                if not 0 <= class_id < 19:
                    raise ValueError(f"Invalid tissue ID {class_id}")
                if class_names.setdefault(class_id, tissue_label) != tissue_label:
                    raise ValueError(f"Conflicting label for tissue ID {class_id}")
                key = (fold, sample_index)
                if key in records:
                    raise ValueError(f"Duplicate source key {key}")
                records[key] = SourceRecord(
                    fold, sample_index, tissue_label, class_id,
                    SourceLocation(shard, row_group, row_in_group),
                )
    if len(records) != 7901 or set(class_names) != set(range(19)):
        raise ValueError("PanNuke source must contain 7,901 records and 19 tissues")
    return records


def read_metadata(path: Path) -> list[dict[str, object]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError(f"Metadata columns must be exactly {CSV_FIELDS}")
        return [
            {
                "fold": int(row["fold"]),
                "sample_index": int(row["sample_index"]),
                "tissue_label": row["tissue_label"],
                "class_id": int(row["class_id"]),
                "split": row["split"],
            }
            for row in reader
        ]


def verify_records(rows: Iterable[Mapping[str, object]], index: Mapping[tuple[int, int], SourceRecord]) -> None:
    seen: set[tuple[int, int]] = set()
    for row in rows:
        key = (int(row["fold"]), int(row["sample_index"]))
        if key in seen:
            raise ValueError(f"Duplicate metadata key {key}")
        seen.add(key)
        source = index.get(key)
        if source is None or source.class_id != int(row["class_id"]) or source.tissue_label != str(row["tissue_label"]):
            raise ValueError(f"Source/metadata mismatch for {key}")


def decode_image(value: object) -> np.ndarray:
    encoded = value.get("bytes") if isinstance(value, Mapping) else value
    if encoded is None:
        raise ValueError("Parquet image has no embedded bytes")
    with Image.open(io.BytesIO(encoded)) as image:
        result = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if result.shape != (256, 256, 3):
        raise ValueError(f"Expected RGB 256x256 image, got {result.shape}")
    return result


def preload_images(
    rows: Sequence[Mapping[str, object]], index: Mapping[tuple[int, int], SourceRecord]
) -> dict[tuple[int, int], np.ndarray]:
    grouped: dict[tuple[Path, int], list[tuple[tuple[int, int], int]]] = defaultdict(list)
    for row in rows:
        key = (int(row["fold"]), int(row["sample_index"]))
        location = index[key].location
        grouped[(location.shard, location.row_group)].append((key, location.row_in_group))
    cache: dict[tuple[int, int], np.ndarray] = {}
    for (shard, row_group), needed in grouped.items():
        column = pq.ParquetFile(shard).read_row_group(row_group, columns=["image"]).column("image")
        for key, row_in_group in needed:
            cache[key] = decode_image(column[row_in_group].as_py())
    if len(cache) != len(rows):
        raise RuntimeError("RAM cache is incomplete")
    return cache


class UncachedParquetDataset(Dataset):
    def __init__(self, rows: Sequence[Mapping[str, object]], index: Mapping[tuple[int, int], SourceRecord], cache_size: int = 256) -> None:
        self.rows = list(rows)
        self.index = index
        self.cache_size = cache_size
        self.handles: dict[Path, pq.ParquetFile] = {}
        self.cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["handles"] = {}
        state["cache"] = OrderedDict()
        return state

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        row = self.rows[item]
        key = (int(row["fold"]), int(row["sample_index"]))
        image = self.cache.get(key)
        if image is None:
            location = self.index[key].location
            handle = self.handles.get(location.shard)
            if handle is None:
                handle = pq.ParquetFile(location.shard)
                self.handles[location.shard] = handle
            column = handle.read_row_group(location.row_group, columns=["image"]).column("image")
            image = decode_image(column[location.row_in_group].as_py())
            self.cache[key] = image
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
        tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)
        return tensor, int(row["class_id"])
