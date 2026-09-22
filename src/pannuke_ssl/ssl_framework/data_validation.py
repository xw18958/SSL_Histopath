from __future__ import annotations
import hashlib, random
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np, torch
from torch.utils.data import DataLoader
from pannuke_ssl.data import PanNukeImageDataset, loader_kwargs
from pannuke_ssl.monitor import _classification_metrics, _fixed_linear_predictions, feature_diagnostics, weighted_knn_predictions
from pannuke_ssl.parquet import build_source_index, preload_images, read_metadata, verify_records
from pannuke_ssl.utils import write_csv

NUM_CLASSES = 19


def module_sha(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        t = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update((t.view(torch.uint16) if t.dtype == torch.bfloat16 else t).numpy().tobytes())
    return h.hexdigest()


def validated_pannuke_split(c: dict[str, Any]):
    """Load and strictly validate the fixed 6305/798/798 PanNuke partition."""
    index = build_source_index(Path(c["data"]["root"]))
    rows = read_metadata(Path(c["data"]["metadata_csv"]))
    verify_records(rows, index)

    expected = {
        "train": int(c["downstream"]["train_count"]),
        "val": int(c["downstream"]["validation_count"]),
        "test": int(c["downstream"]["test_count"]),
    }
    observed = Counter(str(row["split"]) for row in rows)
    if observed != Counter(expected):
        raise ValueError(f"PanNuke split sizes changed: observed={dict(observed)}, expected={expected}")

    keys = [(int(row["fold"]), int(row["sample_index"])) for row in rows]
    if len(keys) != len(index) or len(set(keys)) != len(index) or set(keys) != set(index):
        raise ValueError("PanNuke metadata must partition all 7901 source images exactly once")

    per = Counter((str(row["split"]), int(row["class_id"])) for row in rows)
    for split in ("val", "test"):
        total = expected[split]
        if total % NUM_CLASSES:
            raise ValueError(f"Balanced {split} size must be divisible by {NUM_CLASSES}")
        per_class = total // NUM_CLASSES
        if [per[(split, class_id)] for class_id in range(NUM_CLASSES)] != [per_class] * NUM_CLASSES:
            raise ValueError(f"PanNuke {split} split must contain exactly {per_class} images per class")

    if len([row for row in rows if row["split"] == c["data"]["ssl_split"]]) != int(c["data"]["expected_ssl_images"]):
        raise ValueError("SSL split size does not match expected_ssl_images")
    return rows, index


def _base_dataset(c):
    rows, index = validated_pannuke_split(c)
    split = str(c["data"]["ssl_split"])
    ssl_rows = [row for row in rows if row["split"] == split]
    expected = int(c["data"]["expected_ssl_images"])
    if len(ssl_rows) != expected:
        raise ValueError(f"Need exactly {expected} SSL train sources, found {len(ssl_rows)}")
    cache = preload_images(ssl_rows, index) if c["data"]["cache_in_ram"] else None
    return PanNukeImageDataset(ssl_rows, index, cache, include_label=False, include_key=True)


def build_ssl_loader(c, method, *, batch_size=None, workers=None):
    def seed_worker(_):
        s = torch.initial_seed() % 2**32
        random.seed(s)
        np.random.seed(s)

    dataset = method.wrap_dataset(_base_dataset(c))
    w = int(c["training"]["num_workers"] if workers is None else workers)
    kw = dict(
        batch_size=int(batch_size or c["training"]["batch_size"]),
        shuffle=True,
        num_workers=w,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=w > 0,
        drop_last=False,
        generator=torch.Generator().manual_seed(int(c["seed"])),
        worker_init_fn=seed_worker,
    )
    if w > 0:
        kw["prefetch_factor"] = 2
    return DataLoader(dataset, **kw)


class Validator:
    """Train/validation-only frozen-feature monitor; test images are never decoded."""

    def __init__(self, c: dict[str, Any], out: Path, device: torch.device):
        self.c, self.out, self.device, self.history = c, Path(out), device, []
        rows, index = validated_pannuke_split(c)
        selected = [row for row in rows if row["split"] in ("train", "val")]
        cache = preload_images(selected, index)
        self.loaders = {
            split: DataLoader(
                PanNukeImageDataset(
                    [row for row in selected if row["split"] == split],
                    index,
                    cache,
                    include_label=True,
                ),
                **loader_kwargs(
                    int(c["validation"]["batch_size"]),
                    int(c["validation"]["num_workers"]),
                    shuffle=False,
                ),
            )
            for split in ("train", "val")
        }

    @torch.inference_mode()
    def features(self, encoder, split):
        xs, ys = [], []
        for images, y in self.loaders[split]:
            x = images.to(self.device, dtype=torch.float32, non_blocking=True).div_(255.0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = encoder(x)
            if tokens.ndim != 3:
                raise RuntimeError("Encoder must return [B,T,D]")
            xs.append(tokens.float().mean(1).cpu())
            ys.append(y.long().cpu())
        return torch.cat(xs), torch.cat(ys)

    def evaluate(self, encoder, epoch):
        was = encoder.training
        encoder.eval()
        tx, ty = self.features(encoder, "train")
        vx, vy = self.features(encoder, "val")
        vc = self.c["validation"]
        knn = weighted_knn_predictions(
            tx,
            ty,
            vx,
            classes=NUM_CLASSES,
            k=int(vc["knn_k"]),
            temperature=float(vc["knn_temperature"]),
        )
        lin = _fixed_linear_predictions(
            tx,
            ty,
            vx,
            classes=NUM_CLASSES,
            seed=int(self.c["seed"]),
            max_iter=int(vc["linear_lbfgs_max_iter"]),
            device=self.device,
        )
        row = {"epoch": float(epoch)}
        row.update({f"knn_val_{k}": v for k, v in _classification_metrics(knn, vy).items()})
        row.update({f"linear_val_{k}": v for k, v in _classification_metrics(lin, vy).items()})
        row.update(
            feature_diagnostics(
                torch.cat((tx, vx)).to(self.device),
                near_constant_std_threshold=float(vc["near_constant_std_threshold"]),
            )
        )
        self.history.append(row)
        write_csv(self.history, self.out / "validation_metrics.csv")
        if was:
            encoder.train()
        return row
