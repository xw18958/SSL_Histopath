"""B0-Delta adapter for the completed audited B0 final-probe pipeline.

The actual feature extraction, train-only standardization, probe grid, validation
selection, and one-time test functions are imported directly from
``b0_duration_final_probe.py``. This adapter only redirects the selected encoder
and output paths and replaces B0's hard-coded epoch-10 lock with the
validation-selected B0-Delta epoch recorded by its duration run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import b0_duration_final_probe as b0_probe
from pannuke_ssl.config import load_yaml
from pannuke_ssl.utils import atomic_json_dump

ROOT = Path("/raid1/xwan0900/SSL_proj")
ENCODER = ROOT / "outputs/b0_delta_duration_pilot/checkpoints/best.pt"
OUT = ROOT / "outputs/b0_delta_duration_pilot_final_probe"
DURATION_SELECTION = ROOT / "outputs/b0_delta_duration_pilot/duration_selection.json"

# Redirect the existing audited B0 pipeline rather than reimplement it.
b0_probe.ENCODER = ENCODER
b0_probe.OUT = OUT


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(values):
    return b0_probe.tensor_sha(values)


def _load_duration_selection():
    if not DURATION_SELECTION.is_file():
        raise FileNotFoundError(
            "Run B0-Delta duration pretraining before the final probe: "
            f"{DURATION_SELECTION}"
        )
    selection = json.loads(DURATION_SELECTION.read_text())
    required = {
        "selected_epoch",
        "selection_metric",
        "selection_score",
        "horizon_epochs",
        "minimum_delta_macro_f1",
        "test_split_used",
    }
    missing = required.difference(selection)
    if missing:
        raise ValueError(f"Invalid B0-Delta duration selection; missing {sorted(missing)}")
    if selection["selection_metric"] != "validation_linear_macro_f1":
        raise ValueError("B0-Delta duration selection must use validation linear macro-F1")
    if int(selection["horizon_epochs"]) != 300:
        raise ValueError("B0-Delta duration selection must come from the 300-epoch horizon")
    if float(selection["minimum_delta_macro_f1"]) != 0.005:
        raise ValueError("B0-Delta checkpoint replacement must use minimum_delta=0.005")
    if bool(selection["test_split_used"]):
        raise ValueError("Test data must not be used for SSL duration selection")
    return selection


def lock_encoder(c):
    """B0 lock logic with only the method-specific selected epoch made dynamic."""
    OUT.mkdir(exist_ok=False)
    duration = _load_duration_selection()
    checkpoint = torch.load(ENCODER, map_location="cpu", weights_only=False)
    selected_epoch = int(duration["selected_epoch"])
    if int(checkpoint["epoch"]) != selected_epoch:
        raise RuntimeError(
            f"Checkpoint epoch {checkpoint['epoch']} != duration-selected epoch {selected_epoch}"
        )
    if checkpoint["config"]["train"]["epochs"] != 300:
        raise RuntimeError("B0-Delta checkpoint is not from the matched 300-epoch schedule")
    if checkpoint["config"]["train"]["warmup_fraction"] != 0.1:
        raise RuntimeError("B0-Delta checkpoint does not use the matched 30-epoch warmup")
    if checkpoint["config"]["output_dir"] != str(ROOT / "outputs/b0_delta_duration_pilot"):
        raise RuntimeError("Unexpected B0-Delta checkpoint output_dir")
    if "student" not in checkpoint or "teacher" in checkpoint:
        raise RuntimeError("Final B0-Delta checkpoint must contain the student only")

    state = {k.removeprefix("_orig_mod."): v for k, v in checkpoint["student"].items()}
    lock = {
        "checkpoint_path": str(ENCODER),
        "checkpoint_sha256": sha(ENCODER),
        "student_state_sha256": tensor_sha(state),
        "epoch": selected_epoch,
        "schedule_epochs": 300,
        "warmup_epochs": 30,
        "wording": (
            f"epoch {selected_epoch} of the 300-epoch schedule with 30-epoch warmup"
        ),
        "metadata_sha256": sha(c["metadata_csv"]),
        "old_b0_final_manifest": b0_probe.old_manifest(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "ssl_seed": checkpoint["config"]["seed"],
        "duration_selection_path": str(DURATION_SELECTION),
        "duration_selection_sha256": sha(DURATION_SELECTION),
        "duration_selection": duration,
        "objective": "EMA_teacher(less-degraded) - EMA_teacher(more-degraded)",
    }
    b0_probe.metadata(c)
    atomic_json_dump(c, OUT / "resolved_config.json")
    atomic_json_dump(lock, OUT / "encoder_lock.json")
    return lock


# Base pipeline functions resolve this global at runtime, so patching it makes
# extract/smoke/tune/test all use the dynamic B0-Delta lock.
b0_probe.lock_encoder = lock_encoder
_original_verify_lock = b0_probe.verify_lock


def verify_lock(c):
    lock = _original_verify_lock(c)
    if not DURATION_SELECTION.is_file():
        raise RuntimeError("B0-Delta duration selection disappeared after locking")
    if sha(DURATION_SELECTION) != lock["duration_selection_sha256"]:
        raise RuntimeError("B0-Delta duration selection changed after encoder locking")
    return lock


b0_probe.verify_lock = verify_lock


def test_once(c):
    """Run the base one-time test exactly once, then only relabel saved reporting."""
    b0_probe.test_once(c)

    summary_path = OUT / "linear_probe_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["evaluation_protocol"] = "single-seed, transductive B0-Delta linear-probe result"
    summary["objective"] = "EMA_teacher(less-degraded) - EMA_teacher(more-degraded)"
    atomic_json_dump(summary, summary_path)

    # The base test has already persisted the one test evaluation. Re-render only
    # its saved confusion matrix with the correct method title; no model/test call.
    matrix = np.loadtxt(OUT / "test_confusion_matrix.csv", delimiter=",", dtype=int)
    names = {int(r["class_id"]): r["tissue_label"] for r in b0_probe.metadata(c)}
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks(range(19), [names[i] for i in range(19)], rotation=65, ha="right")
    ax.set_yticks(range(19), [names[i] for i in range(19)])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("B0-Delta duration-pilot encoder: single-seed transductive linear probe")
    for i in range(19):
        for j in range(19):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[i, j] > matrix.max() / 2 else "black",
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT / "test_confusion_matrix.png", dpi=180)
    plt.close(fig)


def main(stage):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/b0_delta_duration_final_probe.yaml"),
    )
    args = parser.parse_args()
    c = load_yaml(args.config)
    b0_probe.setup(c)
    stages = {
        "extract": b0_probe.extract_trainval,
        "smoke": b0_probe.smoke,
        "tune": b0_probe.tune,
        "test": test_once,
    }
    stages[stage](c)
