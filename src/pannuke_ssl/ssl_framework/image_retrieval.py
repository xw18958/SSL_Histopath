"""Frozen-encoder image-image retrieval on a frozen external TEST manifest.

Protocol: the TEST set is both query and gallery; the query image itself is
removed from its gallery. Relevance is equality of the pathology/class label.
No SSL or probe parameters are trained during retrieval evaluation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from pannuke_ssl.utils import atomic_json_dump, seed_everything, write_csv
from .downstream import _extract
from .external_datasets import load_external_manifest
from .trainer import load_checkpoint


def _validate_ks(ks: Iterable[int], gallery_size: int) -> tuple[int, ...]:
    values = tuple(sorted(set(int(k) for k in ks)))
    if not values or any(k < 1 for k in values):
        raise ValueError(f"Retrieval K values must be positive, got {values}")
    if max(values) >= gallery_size:
        raise ValueError(f"Largest K={max(values)} must be smaller than TEST size={gallery_size}")
    return values


def class_retrieval_metrics(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    ks: Iterable[int] = (1, 5, 10),
    chunk_size: int = 256,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Compute class-relevance retrieval metrics, excluding self matches."""
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError(f"Invalid retrieval tensors: features={tuple(features.shape)}, labels={tuple(labels.shape)}")
    n = int(labels.numel())
    if n < 3:
        raise ValueError("Image retrieval requires at least three TEST images")
    ks = _validate_ks(ks, n)
    counts = torch.bincount(labels.long())
    if any(int(counts[int(label)]) < 2 for label in labels):
        raise ValueError("Every retrieval class needs at least two TEST images after self-exclusion")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gallery = torch.nn.functional.normalize(features.float(), dim=1).to(device)
    gallery_labels = labels.long().to(device)
    totals = {f"precision@{k}": 0.0 for k in ks}
    totals.update({f"recall@{k}": 0.0 for k in ks})
    totals.update({f"hit_rate@{k}": 0.0 for k in ks})
    total_ap = 0.0
    per_query: list[dict[str, Any]] = []

    for start in range(0, n, int(chunk_size)):
        stop = min(n, start + int(chunk_size))
        similarity = gallery[start:stop] @ gallery.T
        local = torch.arange(stop - start, device=device)
        similarity[local, torch.arange(start, stop, device=device)] = -torch.inf
        ranking = torch.argsort(similarity, dim=1, descending=True)
        ranked_labels = gallery_labels[ranking]
        query_labels = gallery_labels[start:stop, None]
        query_indices = torch.arange(start, stop, device=device)[:, None]
        relevant = ranked_labels.eq(query_labels) & ranking.ne(query_indices)
        for local_index in range(stop - start):
            global_index = start + local_index
            total_relevant = int((gallery_labels == gallery_labels[global_index]).sum().item()) - 1
            rel = relevant[local_index]
            cumulative = rel.cumsum(0).float()
            relevant_positions = torch.nonzero(rel, as_tuple=False).flatten()
            precision_at_relevant = cumulative[relevant_positions] / (relevant_positions.float() + 1.0)
            ap = float(precision_at_relevant.sum().item() / total_relevant)
            row: dict[str, Any] = {
                "query_index": global_index,
                "class_id": int(gallery_labels[global_index].item()),
                "relevant_gallery": total_relevant,
                "average_precision": ap,
            }
            total_ap += ap
            for k in ks:
                found = int(rel[:k].sum().item())
                precision = found / k
                recall = found / total_relevant
                hit = float(found > 0)
                row[f"precision@{k}"] = precision
                row[f"recall@{k}"] = recall
                row[f"hit_rate@{k}"] = hit
                totals[f"precision@{k}"] += precision
                totals[f"recall@{k}"] += recall
                totals[f"hit_rate@{k}"] += hit
            per_query.append(row)

    metrics = {key: value / n for key, value in totals.items()}
    metrics["mAP"] = total_ap / n
    return metrics, per_query


def run_image_retrieval(
    c: dict[str, Any],
    checkpoint: Path,
    out: Path,
    *,
    dataset: str,
    ks: Iterable[int] = (1, 5, 10),
) -> dict[str, Any]:
    """Evaluate one frozen SSL checkpoint on one external TEST retrieval set."""
    from pannuke_ssl.ssl_methods.registry import build_method

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for frozen feature extraction")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "retrieval_metrics.json").exists():
        raise FileExistsError(f"Refuse overwrite: {out}")
    seed_everything(int(c["seed"]))
    external = load_external_manifest(dataset, Path(c["manifests"]["root"]))
    device = torch.device("cuda")
    method = build_method(c, device)
    checkpoint_state = load_checkpoint(method, Path(checkpoint), device)
    encoder = method.encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    rows = external.split_rows["test"]
    features, labels, keys = _extract(encoder, rows, c, device, external)
    metrics, query_rows = class_retrieval_metrics(features, labels, ks=ks)
    key_lookup = {int(row["record_index"]): row for row in rows}
    for row, record_index in zip(query_rows, keys.tolist()):
        source = key_lookup[int(record_index)]
        row["record_index"] = int(record_index)
        row["source"] = str(source.get("relative_path", source.get("parquet_file", "")))
    write_csv(query_rows, out / "per_query_metrics.csv")
    result = {
        "method": c["method"]["name"],
        "dataset": dataset,
        "encoder_epoch": int(checkpoint_state["epoch"]),
        "feature_dim": int(features.shape[1]),
        "queries": int(labels.numel()),
        "gallery": int(labels.numel()),
        "self_match_excluded": True,
        "relevance": "same_class_label",
        "encoder_frozen": True,
        "ks": sorted(set(int(k) for k in ks)),
        "metrics": metrics,
        "manifest_sha256": external.manifest_sha256,
        "test_only_retrieval": True,
    }
    atomic_json_dump(result, out / "retrieval_metrics.json")
    return result
