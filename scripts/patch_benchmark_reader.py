"""Faster exact-equivalent benchmark image reader for HF-Parquet datasets.

Caches at most two row groups per DataLoader worker; otherwise uses precisely
the standard ExternalProbeImageDataset decoding/cropping implementation.
"""
from __future__ import annotations

import io
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import pyarrow.parquet as pq
from PIL import Image
import torch
import numpy as np
from torch.utils.data import DataLoader

from pannuke_ssl.data import loader_kwargs
from pannuke_ssl.ssl_framework.external_datasets import ExternalProbeImageDataset


class CachedBenchmarkImageDataset(ExternalProbeImageDataset):
    def __init__(self, rows, root):
        super().__init__(rows, root)
        self._rg_cache = OrderedDict()
        self._rg_limit = 2

    def __getstate__(self):
        state = super().__getstate__()
        state["_rg_cache"] = OrderedDict()
        return state

    def _open_record(self, row: Mapping):
        if row.get("storage", "file") != "parquet":
            return super()._open_record(row)
        path = self.root / str(row["parquet_file"])
        rg = int(row["row_group"])
        key = (str(path), rg)
        array = self._rg_cache.get(key)
        if array is None:
            handle = self._parquet_handles.get(path)
            if handle is None:
                handle = pq.ParquetFile(path)
                self._parquet_handles[path] = handle
            # Exactly the existing source column and bytes; only the repeated
            # per-image read_row_group I/O is eliminated.
            array = handle.read_row_group(rg, columns=["image"])["image"]
            self._rg_cache[key] = array
            if len(self._rg_cache) > self._rg_limit:
                self._rg_cache.popitem(last=False)
        else:
            self._rg_cache.move_to_end(key)
        value = array[int(row["row_in_group"])].as_py()
        encoded = value.get("bytes") if isinstance(value, Mapping) else value
        if encoded is None and isinstance(value, Mapping) and value.get("path"):
            with Image.open(value["path"]) as image:
                return image.convert("RGB").copy()
        if encoded is None:
            raise ValueError(f"Parquet record has no image bytes or path: {path}")
        with Image.open(io.BytesIO(encoded)) as image:
            return image.convert("RGB").copy()


@torch.inference_mode()
def extract_frozen_features(encoder, rows, config, device, external_dataset):
    """Match ssl_framework.downstream._extract's exact tensors/readout/order."""
    batch_size = int(config["downstream"]["batch_size"])
    workers = int(config["downstream"]["num_workers"])
    original_indices = [int(r["record_index"]) for r in rows]
    if all(r.get("storage") == "parquet" for r in rows):
        # Preserve output/label/sample order after extraction while reading
        # large Parquet rowgroups together rather than repeatedly by class.
        ordered_rows = sorted(rows, key=lambda r: (
            r["parquet_file"], int(r["row_group"]), int(r["row_in_group"])
        ))
    else:
        ordered_rows = rows
    loader = DataLoader(
        CachedBenchmarkImageDataset(ordered_rows, external_dataset.root),
        **loader_kwargs(batch_size, workers, shuffle=False),
    )
    xs, ys, indices = [], [], []
    for images, labels, record_indices in loader:
        x = images.to(device, dtype=torch.float32, non_blocking=True).div_(255.0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features = encoder(x)
        if features.ndim == 3:
            features = features.mean(1)
        expected = config["downstream"].get("feature_dim")
        if features.ndim != 2 or (
            expected is not None and features.shape[1] != int(expected)
        ):
            raise RuntimeError(f"Invalid encoder output {tuple(features.shape)}")
        xs.append(features.float().cpu())
        ys.append(labels.long().cpu())
        indices.extend(record_indices.tolist())
    features = torch.cat(xs)
    targets = torch.cat(ys)
    indices = np.asarray(indices, dtype=np.int64)
    pos = {int(idx): i for i, idx in enumerate(indices)}
    if len(pos) != len(indices) or set(pos) != set(original_indices):
        raise RuntimeError("Parquet throughput optimisation changed sample identities")
    perm = [pos[idx] for idx in original_indices]
    return features[perm], targets[perm], indices[perm]
