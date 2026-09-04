from __future__ import annotations

import csv
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_json_dump(value: Any, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=destination.parent, delete=False, encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def write_csv(rows: Iterable[Mapping[str, Any]], destination: str | Path) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Training rows gain a few fields as optional monitoring is enabled.  Keep a
    # stable union of all fields instead of silently dropping fields which do
    # not happen to be present in the first row.
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def plot_history(history: list[dict[str, float]], fields: list[str], destination: str | Path, title: str) -> None:
    if not history:
        return
    fig, axis = plt.subplots(figsize=(8, 5))
    x = [row.get("epoch", index + 1) for index, row in enumerate(history)]
    for field in fields:
        if field in history[0]:
            axis.plot(x, [row[field] for row in history], marker="o", label=field)
    axis.set_xlabel("epoch")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)
