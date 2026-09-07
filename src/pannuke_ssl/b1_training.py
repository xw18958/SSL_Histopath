"""Isolated validation-selected training for the B1 discrete-velocity ablation."""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .b1 import build_b1_models, residual_endpoint
from .b1_losses import b1_loss
from .data import PanNukeImageDataset, build_ssl_loader, loader_kwargs
from .degradations import DegradationEndpoints, degrade_batch, sample_adjacent_transitions
from .models import update_ema
from .monitor import SSLRepresentationMonitor, improves_macro_f1
from .parquet import build_source_index, preload_images, read_metadata, verify_records
from .training import _cosine, _learning_rate, _make_optimizer, _save_final_student, _weight_decay
from .utils import atomic_json_dump, plot_history, seed_everything, write_csv


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protected_manifest(output_dir: str | Path) -> dict[str, Any]:
    path = Path(output_dir) / "protected_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"B1 requires the pre-implementation manifest: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not isinstance(value.get("protected_paths"), list):
        raise ValueError("Invalid B1 protected manifest")
    return value


def verify_protected_manifest(output_dir: str | Path) -> dict[str, Any]:
    """Fail closed if a B0/I-JEPA path recorded before B1 changed."""
    output = Path(output_dir)
    root = output.parents[1]
    manifest = load_protected_manifest(output)
    for entry in manifest["protected_paths"]:
        path = root / entry["path"]
        if not path.is_file():
            raise RuntimeError(f"Protected reference artifact is missing: {path}")
        if path.stat().st_size != int(entry["size"]) or sha256_path(path) != entry["sha256"]:
            raise RuntimeError(f"Protected B0/I-JEPA artifact changed: {path}")
    return manifest


def _endpoints(path: str | Path) -> DegradationEndpoints:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    return DegradationEndpoints(
        defocus_radius=float(values["defocus"]["selected_radius"]),
        resolution_factor=float(values["resolution"]["selected_factor"]),
    )


def verified_metadata(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_metadata(Path(config["monitor"]["metadata_csv"]))
    expected = {"train": 2052, "val": 247, "test": 247}
    if Counter(str(row["split"]) for row in rows) != expected:
        raise ValueError("Balanced metadata does not have the expected 2052/247/247 splits")
    if len({(int(row["fold"]), int(row["sample_index"])) for row in rows}) != 2546:
        raise ValueError("Balanced metadata has duplicate source keys")
    counts = Counter((str(row["split"]), int(row["class_id"])) for row in rows)
    for split, count in (("train", 108), ("val", 13), ("test", 13)):
        if [counts[split, class_id] for class_id in range(19)] != [count] * 19:
            raise ValueError("Balanced metadata per-class counts are not 108/13/13")
    return rows


class StrictB1Monitor(SSLRepresentationMonitor):
    """Monitor that never constructs a loader/cache for the test split."""

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = dict(config["monitor"])
        self.output_dir = Path(config["output_dir"])
        self.device, self.seed, self.history = device, int(config["seed"]), []
        rows = verified_metadata(config)
        selected = [row for row in rows if row["split"] in ("train", "val")]
        source_index = build_source_index(Path(config["data_root"]))
        verify_records(selected, source_index)
        cache = preload_images(selected, source_index)
        expected_keys = {(int(row["fold"]), int(row["sample_index"])) for row in selected}
        if set(cache) != expected_keys:
            raise RuntimeError("Validation monitor cache contains unexpected records")
        self.loaders = {
            split: DataLoader(
                PanNukeImageDataset(
                    [row for row in selected if row["split"] == split],
                    source_index,
                    cache,
                    include_label=True,
                ),
                **loader_kwargs(
                    int(self.config["batch_size"]),
                    int(self.config["num_workers"]),
                    shuffle=(split == "train"),
                ),
            )
            for split in ("train", "val")
        }


def validate_config(config: dict[str, Any]) -> None:
    train = config["train"]
    expected_model = {
        "image_size": 256,
        "patch_size": 32,
        "hidden_size": 768,
        "predictor_dim": 384,
        "predictor_depth": 2,
        "predictor_heads": 6,
        "predictor_mlp_ratio": 4,
        "dropout": 0.0,
    }
    if config["seed"] != 20260903 or config["model"] != expected_model:
        raise ValueError("B1 must use the fixed matched ViT-B/32 protocol")
    required_train = {
        "epochs": 300,
        "batch_size": 128,
        "effective_batch_size": 128,
        "num_workers": 8,
        "cache_in_ram": True,
        "learning_rate": 1e-4,
        "minimum_learning_rate": 1e-6,
        "weight_decay": 0.04,
        "final_weight_decay": 0.40,
        "warmup_fraction": 0.10,
        "lambda_reg": 0.10,
        "covariance_weight": 0.04,
        "variance_target": 1.0,
        "ema_start": 0.996,
        "gradient_clip_norm": 5.0,
        "bf16": True,
        "compile": False,
        "resume": None,
    }
    if any(train.get(key) != value for key, value in required_train.items()):
        raise ValueError("B1 train settings must exactly match the locked duration protocol")
    monitor = config["monitor"]
    if not bool(monitor.get("enabled")) or monitor.get("evaluation_interval_epochs") != 10:
        raise ValueError("B1 requires validation-only monitoring every 10 epochs")
    expected_monitor = {
        "batch_size": 256,
        "num_workers": 8,
        "cache_in_ram": True,
        "knn_k": 20,
        "knn_temperature": 0.07,
        "linear_lbfgs_max_iter": 100,
        "near_constant_std_threshold": 0.05,
    }
    if any(monitor.get(key) != value for key, value in expected_monitor.items()):
        raise ValueError("B1 monitoring settings must exactly match the duration protocol")
    if monitor.get("minimum_delta_macro_f1") != 0.005:
        raise ValueError("B1 checkpoint replacement requires a strict 0.005 macro-F1 improvement")
    if Path(config["output_dir"]).name != "b1_duration_pilot":
        raise ValueError("B1 output_dir must be the isolated b1_duration_pilot directory")


def _assert_adjacent(source: torch.Tensor, target: torch.Tensor, delta: torch.Tensor) -> None:
    expected = torch.full_like(source, 0.25)
    if not torch.equal(source - target, expected) or not torch.equal(delta, -expected):
        raise RuntimeError("B1 accepts only adjacent reverse transitions with delta=-0.25")


def run_b1_pretraining(config: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("B1 pretraining requires CUDA")
    validate_config(config)
    output = Path(config["output_dir"])
    if not output.is_dir():
        raise FileNotFoundError(f"B1 output directory must be prepared exclusively: {output}")
    if (output / "pretrain_metrics.csv").exists():
        raise FileExistsError("Refusing to overwrite or restart an existing B1 duration run")
    smoke_path = output / "smoke_test.json"
    if not smoke_path.exists() or not json.loads(smoke_path.read_text())["passed"]:
        raise RuntimeError("B1 requires a passing comprehensive smoke test before substantial training")
    manifest = verify_protected_manifest(output)
    seed_everything(int(config["seed"]))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device, train = torch.device("cuda"), config["train"]
    atomic_json_dump(config, output / "resolved_config.json")
    endpoints = _endpoints(config["endpoints_json"])
    loader, _ = build_ssl_loader(
        config["data_root"],
        batch_size=int(train["batch_size"]),
        num_workers=int(train["num_workers"]),
        cache_in_ram=bool(train["cache_in_ram"]),
    )
    if len(loader.dataset) != 7901 or loader.dataset.include_label:
        raise RuntimeError("B1 SSL must contain exactly 7,901 unlabeled source images")
    student, teacher, predictor = build_b1_models(config, device)
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("B1 teacher must be frozen")
    trainable = list(student.parameters()) + list(predictor.parameters())
    optimizer = _make_optimizer(trainable, float(train["learning_rate"]), float(train["weight_decay"]))
    total_steps = int(train["epochs"]) * len(loader)
    monitor = StrictB1Monitor(config, device)
    history: list[dict[str, float]] = []
    best_score, best_epoch = float("-inf"), 0
    started = time.perf_counter()

    def evaluate(epoch: int) -> None:
        nonlocal best_score, best_epoch
        row = monitor.evaluate(student, epoch)
        candidate = float(row["linear_val_macro_f1"])
        if improves_macro_f1(candidate, best_score, float(config["monitor"]["minimum_delta_macro_f1"])):
            best_score, best_epoch = candidate, epoch
            _save_final_student(
                output / "checkpoints" / "best.pt",
                student=student,
                config=config,
                epoch=epoch,
                history=history,
                selection={
                    "metric": "validation_linear_macro_f1",
                    "value": candidate,
                    "minimum_delta": 0.005,
                    "selection_split": "validation",
                    "test_split_used_for_selection": False,
                    "transductive_ssl": True,
                },
            )
        print(json.dumps({"monitor": row, "best_epoch": best_epoch, "best_score": best_score}, sort_keys=True), flush=True)

    evaluate(0)
    global_step = 0
    for epoch in range(1, 301):
        student.train()
        predictor.train()
        teacher.eval()
        torch.cuda.reset_peak_memory_stats()
        epoch_started = time.perf_counter()
        totals = {name: 0.0 for name in ("total", "prediction", "regularizer", "variance", "covariance", "embedding_std", "velocity_rms")}
        seen = 0
        transition_totals = {
            f"prediction_{family}_{source:03d}_to_{target:03d}": [0.0, 0]
            for family in ("defocus", "resolution")
            for source, target in ((100, 75), (75, 50), (50, 25), (25, 0))
        }
        for batch_index, uint8_images in enumerate(loader):
            images = uint8_images.to(device, dtype=torch.float32, non_blocking=True).div_(255.0)
            actions, source, target, delta = sample_adjacent_transitions(images.shape[0], device)
            _assert_adjacent(source, target, delta)
            source_images = degrade_batch(images, actions, source, endpoints)
            target_images = degrade_batch(images, actions, target, endpoints)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(train["bf16"])):
                source_tokens = student(source_images)
                velocity = predictor(source_tokens, source, actions, delta)
                predicted_target = residual_endpoint(source_tokens, velocity, delta)
                with torch.no_grad():
                    teacher_target = teacher(target_images)
            losses = b1_loss(
                predicted_target,
                teacher_target.detach(),
                source_tokens,
                lambda_reg=float(train["lambda_reg"]),
                covariance_weight=float(train["covariance_weight"]),
                variance_target=float(train["variance_target"]),
            )
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"Non-finite B1 loss at epoch={epoch}, batch={batch_index}")
            losses["total"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, float(train["gradient_clip_norm"]))
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite B1 gradient norm")
            for group in optimizer.param_groups:
                group["lr"] = _learning_rate(train, global_step, total_steps)
                group["weight_decay"] = _weight_decay(train, global_step, total_steps)
            optimizer.step()
            momentum = _cosine(float(train["ema_start"]), 1.0, global_step, total_steps)
            update_ema(student, teacher, momentum)
            global_step += 1
            count = images.shape[0]
            seen += count
            losses["velocity_rms"] = velocity.detach().float().square().mean().sqrt()
            for name in totals:
                totals[name] += float(losses[name].detach()) * count
            with torch.no_grad():
                per_example = F.smooth_l1_loss(
                    predicted_target.float(), teacher_target.detach().float(), reduction="none"
                ).mean(dim=(1, 2))
            for action, family in ((0, "defocus"), (1, "resolution")):
                action_mask = actions == action
                for source_anchor in (1.0, 0.75, 0.5, 0.25):
                    mask = action_mask & torch.isclose(source, torch.tensor(source_anchor, device=device))
                    if torch.any(mask):
                        key = f"prediction_{family}_{int(source_anchor * 100):03d}_to_{int((source_anchor - .25) * 100):03d}"
                        transition_totals[key][0] += float(per_example[mask].sum())
                        transition_totals[key][1] += int(mask.sum())
        seconds = time.perf_counter() - epoch_started
        if seen != 7901:
            raise RuntimeError(f"B1 expected 7,901 samples per epoch, saw {seen}")
        row: dict[str, float] = {"epoch": float(epoch), **{name: value / seen for name, value in totals.items()}}
        row.update({
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
            "ema_momentum": float(momentum),
            "seconds": seconds,
            "samples_per_second": seen / seconds,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        })
        for key, (value, count) in transition_totals.items():
            row[key] = value / count if count else float("nan")
        history.append(row)
        write_csv(history, output / "pretrain_metrics.csv")
        print(json.dumps(row, sort_keys=True), flush=True)
        if epoch % 10 == 0:
            evaluate(epoch)
    plot_history(history, ["prediction", "regularizer", "velocity_rms"], output / "pretrain_losses.png", "B1 endpoint and regularization losses")
    plot_history(monitor.history, ["linear_val_macro_f1", "knn_val_macro_f1"], output / "representation_validation_curves.png", "B1 validation-only monitoring")
    plot_history(monitor.history, ["feature_embedding_std", "feature_effective_rank_fraction", "feature_mean_pairwise_cosine", "feature_near_constant_fraction"], output / "representation_health.png", "B1 representation health")
    raw_best = max(monitor.history, key=lambda value: (value["linear_val_macro_f1"], -value["epoch"]))
    checkpoint = output / "checkpoints" / "best.pt"
    selection = {
        "selected_epoch": best_epoch,
        "selection_score": best_score,
        "horizon_epochs": 300,
        "warmup_epochs": 30,
        "minimum_delta_macro_f1": 0.005,
        "selection_split": "validation",
        "test_split_used_for_selection": False,
        "raw_best_monitor_epoch": raw_best["epoch"],
        "raw_best_monitor_score": raw_best["linear_val_macro_f1"],
        "transductive_ssl": True,
        "checkpoint_sha256": sha256_path(checkpoint),
    }
    atomic_json_dump(selection, output / "duration_selection.json")
    verify_protected_manifest(output)
    result = {
        "epochs": 300,
        "best_epoch": best_epoch,
        "best_validation_linear_macro_f1": best_score,
        "total_seconds": time.perf_counter() - started,
        "final_metrics": history[-1],
        "protected_b0_ijepa_unchanged": True,
        "protected_manifest_entries": len(manifest["protected_paths"]),
        "selection": selection,
    }
    atomic_json_dump(result, output / "pretrain_summary.json")
    return result
