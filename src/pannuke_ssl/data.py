from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .parquet import (
    SourceRecord,
    UncachedParquetDataset,
    build_source_index,
    preload_images,
    read_metadata,
    verify_records,
)


def all_source_rows(source_index: Mapping[tuple[int, int], SourceRecord]) -> list[dict[str, object]]:
    return [
        {
            "fold": record.fold,
            "sample_index": record.sample_index,
            "tissue_label": record.tissue_label,
            "class_id": record.class_id,
        }
        for _, record in sorted(source_index.items())
    ]


class PanNukeImageDataset(Dataset):
    """Image access keyed by the immutable source (fold, sample_index)."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, object]],
        source_index: Mapping[tuple[int, int], SourceRecord],
        image_cache: Mapping[tuple[int, int], np.ndarray] | None,
        *,
        include_label: bool,
        include_key: bool = False,
    ) -> None:
        verify_records(rows, source_index)
        self.rows = [dict(row) for row in rows]
        self.image_cache = image_cache
        self.source_index = source_index
        self._uncached = (
            None
            if image_cache is not None
            else UncachedParquetDataset(rows, source_index, cache_size=256)
        )
        self.include_label = include_label
        self.include_key = include_key

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        key = (int(row["fold"]), int(row["sample_index"]))
        if self.image_cache is not None:
            # The copy is ephemeral batch memory; no duplicate image is persisted to disk.
            image = torch.from_numpy(np.array(self.image_cache[key], copy=True)).permute(2, 0, 1)
        else:
            assert self._uncached is not None
            image, _ = self._uncached[index]
        result: list[object] = [image]
        if self.include_label:
            result.append(int(row["class_id"]))
        if self.include_key:
            result.extend(key)
        return tuple(result) if len(result) > 1 else result[0]


def loader_kwargs(batch_size: int, num_workers: int, *, shuffle: bool) -> dict[str, object]:
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    result: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "drop_last": False,
    }
    if num_workers > 0:
        result["prefetch_factor"] = 2
    return result


def build_ssl_loader(
    data_root: str | Path,
    *,
    batch_size: int,
    num_workers: int | None = None,
    shuffle: bool = True,
    cache_in_ram: bool = True,
) -> tuple[DataLoader, dict[tuple[int, int], SourceRecord]]:
    source_index = build_source_index(Path(data_root))
    rows = all_source_rows(source_index)
    if len(rows) != 7901 or len({(r["fold"], r["sample_index"]) for r in rows}) != 7901:
        raise ValueError("SSL input must contain exactly 7,901 unique source records")
    cache = preload_images(rows, source_index) if cache_in_ram else None
    dataset = PanNukeImageDataset(rows, source_index, cache, include_label=False)
    workers = min(8, os.cpu_count() or 1) if num_workers is None else num_workers
    return DataLoader(dataset, **loader_kwargs(batch_size, workers, shuffle=shuffle)), source_index


def build_balanced_loaders(
    data_root: str | Path,
    metadata_csv: str | Path,
    *,
    batch_size: int,
    num_workers: int | None = None,
    include_key: bool = False,
    cache_in_ram: bool = True,
) -> tuple[dict[str, DataLoader], dict[tuple[int, int], SourceRecord]]:
    source_index = build_source_index(Path(data_root))
    rows = read_metadata(Path(metadata_csv))
    verify_records(rows, source_index)
    expected = {"train": 2052, "val": 247, "test": 247}
    split_rows = {name: [row for row in rows if row["split"] == name] for name in expected}
    actual = {name: len(values) for name, values in split_rows.items()}
    if actual != expected:
        raise ValueError(f"Unexpected balanced split sizes: {actual}")
    counts: dict[tuple[str, int], int] = {}
    for row in rows:
        key = (str(row["split"]), int(row["class_id"]))
        counts[key] = counts.get(key, 0) + 1
    for class_id in range(19):
        if [counts.get((s, class_id), 0) for s in expected] != [108, 13, 13]:
            raise ValueError(f"Class {class_id} does not have the required 108/13/13 split")
    cache = preload_images(rows, source_index) if cache_in_ram else None
    workers = min(8, os.cpu_count() or 1) if num_workers is None else num_workers
    loaders = {
        split: DataLoader(
            PanNukeImageDataset(part, source_index, cache, include_label=True, include_key=include_key),
            **loader_kwargs(batch_size, workers, shuffle=(split == "train")),
        )
        for split, part in split_rows.items()
    }
    return loaders, source_index
