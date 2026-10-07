from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pannuke_ssl.utils import atomic_json_dump
from .runtime_paths import runtime_identity

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _manifest_hash_catalog(c: dict[str, Any]) -> dict[str, str]:
    root = Path(c["manifests"]["root"])
    catalog: dict[str, str] = {}
    if not root.is_dir():
        return catalog
    for checksum in sorted(root.rglob("*.json.sha256")):
        value = checksum.read_text(encoding="utf-8").strip().split(maxsplit=1)[0]
        catalog[checksum.relative_to(root).as_posix().removesuffix(".sha256")] = value
    return catalog


def default_run_id(c: dict[str, Any]) -> str:
    method = str(c["method"]["name"])
    seed = int(c["seed"])
    if method == "simplex_sigreg_lejepa":
        k = int(c["method"]["objective"]["simplex_components"])
        return f"ssl-simplex-k{k}-s{seed}"
    if method == "ppc_lejepa":
        return f"ssl-ppc-lejepa-s{seed}"
    return f"ssl-{method.replace('_', '-')}-s{seed}"


def sanitize_run_id(run_id: str) -> str:
    clean = _SAFE.sub("-", run_id).strip("-")
    if not clean or clean != run_id:
        raise ValueError(f"Run id must already be filesystem-safe [A-Za-z0-9._-]: {run_id!r}")
    return clean


def experiment_root(c: dict[str, Any], run_id: str | None = None) -> Path:
    rid = sanitize_run_id(run_id or default_run_id(c))
    return Path(c["output"]["root"]) / "runs" / rid


def attach_run_context(c: dict[str, Any], run_id: str | None = None) -> tuple[dict[str, Any], str, Path]:
    rid = sanitize_run_id(run_id or default_run_id(c))
    root = experiment_root(c, rid)
    c.setdefault("runtime", {})
    c["runtime"].update({"run_id": rid, "run_root": str(root)})
    return c, rid, root


def build_run_metadata(c: dict[str, Any], *, run_id: str, action: str, dataset: str | None = None) -> dict[str, Any]:
    device = None
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        device = {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
        }
    return {
        "run_id": run_id,
        "action": action,
        "dataset": dataset,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": c["method"]["name"],
        "seed": int(c["seed"]),
        "simplex_components": c["method"].get("objective", {}).get("simplex_components"),
        "simplex_sigma": c["method"].get("objective", {}).get("simplex_sigma"),
        "ppc_lambda": c["method"].get("projector_plasticity", {}).get("lambda"),
        "ppc_epsilon": c["method"].get("projector_plasticity", {}).get("epsilon"),
        "peak_lr": float(c["method"]["optimizer"]["peak_lr"]),
        "checkpoint_epochs": [int(x) for x in c["training"]["checkpoint_epochs"]],
        "manifest_hashes": _manifest_hash_catalog(c),
        "runtime": runtime_identity(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_device": device,
    }


def write_run_metadata(c: dict[str, Any], root: Path, *, run_id: str, action: str, dataset: str | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "run_metadata.json"
    metadata = build_run_metadata(c, run_id=run_id, action=action, dataset=dataset)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        immutable = (
            "run_id", "method", "seed", "simplex_components", "simplex_sigma",
            "ppc_lambda", "ppc_epsilon", "peak_lr", "checkpoint_epochs", "manifest_hashes",
        )
        mismatched = [key for key in immutable if existing.get(key) != metadata.get(key)]
        existing_commit = existing.get("runtime", {}).get("git_commit")
        current_commit = metadata.get("runtime", {}).get("git_commit")
        if existing_commit != current_commit and action == "pretrain":
            mismatched.append("git_commit")
        if mismatched:
            raise RuntimeError(f"Run metadata collision at {path}; mismatched fields={sorted(set(mismatched))}")
    else:
        atomic_json_dump(metadata, path)

    # Every invocation gets its own provenance record. This is essential when
    # pretraining and downstream jobs for one run execute on different servers.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    host = str(metadata["runtime"]["hostname"]).replace("/", "-")
    action_slug = action.replace("/", "-")
    dataset_slug = "" if dataset is None else "_" + str(dataset).replace("/", "-")
    execution = root / "executions" / f"{stamp}_{host}_{action_slug}{dataset_slug}.json"
    atomic_json_dump(metadata, execution)
    return path
