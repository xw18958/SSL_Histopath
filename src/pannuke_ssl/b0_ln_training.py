"""Isolated B0-LN ablation training.

This module retains the matched B0 protocol exactly, except that the
stop-gradient EMA teacher endpoint is stateless FP32 LayerNorm-normalized as
in the implemented I-JEPA objective.  It intentionally owns its monitor and
manifest checks so B0/B1/I-JEPA code and artifacts are never modified.
"""
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
from torch import nn
from torch.utils.data import DataLoader

from .data import PanNukeImageDataset, build_ssl_loader, loader_kwargs
from .degradations import DegradationEndpoints, degrade_batch, sample_adjacent_transitions
from .losses import b0_loss
from .models import update_ema
from .monitor import SSLRepresentationMonitor, improves_macro_f1
from .parquet import build_source_index, preload_images, read_metadata, verify_records
from .training import _cosine, _endpoints, _learning_rate, _make_optimizer, _save_final_student, _weight_decay, build_b0_models
from .utils import atomic_json_dump, plot_history, seed_everything, write_csv


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(output_dir: str | Path, name: str, purpose: str) -> dict[str, Any]:
    path = Path(output_dir) / name
    if not path.is_file():
        raise FileNotFoundError(f"B0-LN requires {purpose}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not isinstance(value.get("protected_paths"), list):
        raise ValueError(f"Invalid B0-LN {purpose}: {path}")
    return value


def _verify_manifest(output_dir: str | Path, name: str, purpose: str) -> dict[str, Any]:
    output = Path(output_dir)
    root = output.parents[1]
    manifest = _load_manifest(output, name, purpose)
    for entry in manifest["protected_paths"]:
        path = root / entry["path"]
        if not path.is_file():
            raise RuntimeError(f"Protected B0-LN reference is missing: {path}")
        if path.stat().st_size != int(entry["size"]) or sha256_path(path) != entry["sha256"]:
            raise RuntimeError(f"Protected B0-LN reference changed: {path}")
    return manifest


def verify_protected_manifest(output_dir: str | Path) -> dict[str, Any]:
    """Fail closed if a pre-existing B0/B1/I-JEPA reference changed."""
    return _verify_manifest(output_dir, "protected_manifest.json", "pre-implementation reference manifest")


def verify_implementation_manifest(output_dir: str | Path) -> dict[str, Any]:
    """Fail closed if a frozen B0-LN implementation surface changed."""
    return _verify_manifest(output_dir, "implementation_manifest.json", "frozen implementation manifest")


def layer_norm_teacher_target(teacher: nn.Module, target_images: torch.Tensor) -> torch.Tensor:
    """I-JEPA-style stateless FP32 normalization of *only* the EMA target."""
    with torch.no_grad():
        raw_teacher_tokens = teacher(target_images)
        if raw_teacher_tokens.ndim != 3 or raw_teacher_tokens.shape[1:] != (64, 768):
            raise RuntimeError(f"Unexpected B0-LN teacher token shape: {tuple(raw_teacher_tokens.shape)}")
        teacher_tokens = F.layer_norm(
            raw_teacher_tokens.float(),
            (raw_teacher_tokens.shape[-1],),
        )
    return teacher_tokens


def verified_metadata(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_metadata(Path(config["monitor"]["metadata_csv"]))
    if Counter(str(row["split"]) for row in rows) != {"train": 2052, "val": 247, "test": 247}:
        raise ValueError("Balanced metadata does not have the expected 2052/247/247 splits")
    if len({(int(row["fold"]), int(row["sample_index"])) for row in rows}) != 2546:
        raise ValueError("Balanced metadata contains duplicate source keys")
    counts = Counter((str(row["split"]), int(row["class_id"])) for row in rows)
    for split, expected in (("train", 108), ("val", 13), ("test", 13)):
        if [counts[split, class_id] for class_id in range(19)] != [expected] * 19:
            raise ValueError("Balanced metadata per-class counts are not 108/13/13")
    return rows


class StrictB0LNMonitor(SSLRepresentationMonitor):
    """A monitor that constructs source records/loaders for train and val only."""

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = dict(config["monitor"])
        self.output_dir = Path(config["output_dir"])
        self.device = device
        self.seed = int(config["seed"])
        self.history: list[dict[str, float]] = []
        rows = verified_metadata(config)
        selected = [row for row in rows if row["split"] in ("train", "val")]
        index = build_source_index(Path(config["data_root"]))
        verify_records(selected, index)
        cache = preload_images(selected, index)
        expected_keys = {(int(row["fold"]), int(row["sample_index"])) for row in selected}
        if set(cache) != expected_keys:
            raise RuntimeError("B0-LN monitor cache contains records outside train/validation")
        self.loaders = {
            split: DataLoader(
                PanNukeImageDataset(
                    [row for row in selected if row["split"] == split],
                    index,
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
    expected_train = {
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
    expected_monitor = {
        "enabled": True,
        "batch_size": 256,
        "num_workers": 8,
        "cache_in_ram": True,
        "evaluation_interval_epochs": 10,
        "knn_k": 20,
        "knn_temperature": 0.07,
        "linear_lbfgs_max_iter": 100,
        "near_constant_std_threshold": 0.05,
        "minimum_delta_macro_f1": 0.005,
    }
    adaptive = config.get("adaptive_duration", {})
    if config.get("seed") != 20260903 or config.get("model") != expected_model:
        raise ValueError("B0-LN must use the locked matched ViT-B/32 protocol")
    if any(config["train"].get(key) != value for key, value in expected_train.items()):
        raise ValueError("B0-LN train settings must exactly match B0's 300-epoch protocol")
    if any(config["monitor"].get(key) != value for key, value in expected_monitor.items()):
        raise ValueError("B0-LN monitor settings must exactly match B0's validation monitor")
    if adaptive != {"minimum_epochs": 100, "consecutive_non_improving_monitors": 5}:
        raise ValueError("B0-LN adaptive duration must be exactly the approved 100-epoch/5-monitor policy")
    if Path(config["output_dir"]).name != "b0_ln_duration_pilot":
        raise ValueError("B0-LN output_dir must be the isolated b0_ln_duration_pilot directory")


def adaptive_failure_count(epoch: int, improved: bool, current_failures: int) -> int:
    """Apply the approved stopping counter without changing checkpoint selection."""
    if improved:
        return 0
    if epoch >= 110:
        return current_failures + 1
    return current_failures


def _assert_adjacent(source: torch.Tensor, target: torch.Tensor, delta: torch.Tensor) -> None:
    expected = torch.full_like(source, 0.25)
    if not torch.equal(source - target, expected) or not torch.equal(delta, -expected):
        raise RuntimeError("B0-LN accepts only adjacent reverse transitions with delta=-0.25")


def run_b0_ln_pretraining(config: dict[str, Any]) -> dict[str, Any]:
    """Run B0-LN with validation-only selection and the approved adaptive stop."""
    if not torch.cuda.is_available():
        raise RuntimeError("B0-LN pretraining requires CUDA")
    validate_config(config)
    output = Path(config["output_dir"])
    if not output.is_dir():
        raise FileNotFoundError(f"B0-LN output directory must be prepared exclusively: {output}")
    if (output / "pretrain_metrics.csv").exists():
        raise FileExistsError("Refusing to overwrite or restart an existing B0-LN duration run")
    if (output / "test_started.json").exists():
        raise RuntimeError("B0-LN pretraining refuses to run after a test marker exists")
    smoke = output / "smoke_test.json"
    if not smoke.is_file() or not json.loads(smoke.read_text(encoding="utf-8")).get("passed"):
        raise RuntimeError("B0-LN requires a passing comprehensive smoke test before substantial training")
    protected = verify_protected_manifest(output)
    implementation = verify_implementation_manifest(output)
    seed_everything(int(config["seed"]))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    train = config["train"]
    atomic_json_dump(config, output / "resolved_config.json")
    endpoints: DegradationEndpoints = _endpoints(config["endpoints_json"])
    loader, _ = build_ssl_loader(
        config["data_root"],
        batch_size=int(train["batch_size"]),
        num_workers=int(train["num_workers"]),
        cache_in_ram=bool(train["cache_in_ram"]),
    )
    if len(loader.dataset) != 7901 or loader.dataset.include_label:
        raise RuntimeError("B0-LN SSL must contain exactly 7,901 unlabeled source images")
    student, teacher, predictor = build_b0_models(config, device)
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("B0-LN EMA teacher must remain frozen")
    trainable = list(student.parameters()) + list(predictor.parameters())
    optimizer = _make_optimizer(trainable, float(train["learning_rate"]), float(train["weight_decay"]))
    batch_size = int(train["batch_size"])
    effective_batch = int(train["effective_batch_size"])
    if effective_batch < batch_size or effective_batch % batch_size:
        raise ValueError("B0-LN effective_batch_size must be a positive batch-size multiple")
    accumulation_steps = effective_batch // batch_size
    optimizer_steps_per_epoch = math.ceil(len(loader) / accumulation_steps)
    total_steps = int(train["epochs"]) * optimizer_steps_per_epoch
    monitor = StrictB0LNMonitor(config, device)
    history: list[dict[str, float]] = []
    best_score, best_epoch = float("-inf"), 0
    global_step = 0
    failures = 0
    adaptive_history: list[dict[str, Any]] = []
    early_stopped = False
    stopping_epoch = int(train["epochs"])
    started = time.perf_counter()

    def evaluate(epoch: int) -> None:
        nonlocal best_score, best_epoch, failures
        row = monitor.evaluate(student, epoch)
        candidate = float(row["linear_val_macro_f1"])
        incumbent = best_score
        improved = improves_macro_f1(candidate, incumbent, float(config["monitor"]["minimum_delta_macro_f1"]))
        counted_failure = False
        if improved:
            best_score, best_epoch = candidate, epoch
            failures = adaptive_failure_count(epoch, True, failures)
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
        else:
            updated_failures = adaptive_failure_count(epoch, False, failures)
            counted_failure = updated_failures != failures
            failures = updated_failures
        adaptive_history.append(
            {
                "epoch": epoch,
                "candidate_validation_linear_macro_f1": candidate,
                "incumbent_before": incumbent,
                "strict_improvement": improved,
                "counted_failure": counted_failure,
                "consecutive_failures": failures,
            }
        )
        print(json.dumps({"monitor": row, "best_epoch": best_epoch, "best_score": best_score,
                          "adaptive_failures": failures}, sort_keys=True), flush=True)

    evaluate(0)
    for epoch in range(1, int(train["epochs"]) + 1):
        student.train()
        predictor.train()
        teacher.eval()
        torch.cuda.reset_peak_memory_stats()
        epoch_started = time.perf_counter()
        totals = {name: 0.0 for name in ("total", "prediction", "regularizer", "variance", "covariance", "embedding_std")}
        transition_totals = {
            f"prediction_{family}_{source:03d}_to_{target:03d}": [0.0, 0]
            for family in ("defocus", "resolution")
            for source, target in ((100, 75), (75, 50), (50, 25), (25, 0))
        }
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, uint8_images in enumerate(loader):
            images = uint8_images.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
            actions, source_severity, target_severity, delta = sample_adjacent_transitions(images.shape[0], device)
            _assert_adjacent(source_severity, target_severity, delta)
            source_images = degrade_batch(images, actions, source_severity, endpoints)
            target_images = degrade_batch(images, actions, target_severity, endpoints)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(train["bf16"])):
                student_tokens = student(source_images)
                prediction = predictor(student_tokens, source_severity, actions, delta)
                teacher_tokens = layer_norm_teacher_target(teacher, target_images)
            if tuple(student_tokens.shape[1:]) != (64, 768) or prediction.shape != student_tokens.shape:
                raise RuntimeError("B0-LN did not traverse the matched 64x768 predictor path")
            if teacher_tokens.dtype != torch.float32 or teacher_tokens.shape != student_tokens.shape:
                raise RuntimeError("B0-LN teacher target must be FP32 and shape-matched")
            losses = b0_loss(
                prediction,
                teacher_tokens.detach(),
                student_tokens,
                lambda_reg=float(train["lambda_reg"]),
                covariance_weight=float(train["covariance_weight"]),
                variance_target=float(train["variance_target"]),
            )
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"Non-finite B0-LN loss at epoch={epoch}, batch={batch_index}")
            (losses["total"] / accumulation_steps).backward()
            is_update = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader)
            if is_update:
                gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, float(train["gradient_clip_norm"]))
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("Non-finite B0-LN gradient norm")
                for group in optimizer.param_groups:
                    group["lr"] = _learning_rate(train, global_step, total_steps)
                    group["weight_decay"] = _weight_decay(train, global_step, total_steps)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update_ema(student, teacher, _cosine(float(train["ema_start"]), 1.0, global_step, total_steps))
                global_step += 1
            count = images.shape[0]
            seen += count
            for name in totals:
                totals[name] += float(losses[name].detach()) * count
            with torch.no_grad():
                per_example = F.smooth_l1_loss(
                    prediction.float(), teacher_tokens.detach().float(), reduction="none"
                ).mean(dim=(1, 2))
            for action, family in ((0, "defocus"), (1, "resolution")):
                action_mask = actions == action
                for source_anchor in (1.0, 0.75, 0.5, 0.25):
                    mask = action_mask & torch.isclose(source_severity, torch.tensor(source_anchor, device=device))
                    if torch.any(mask):
                        key = f"prediction_{family}_{int(source_anchor * 100):03d}_to_{int((source_anchor - .25) * 100):03d}"
                        transition_totals[key][0] += float(per_example[mask].sum())
                        transition_totals[key][1] += int(mask.sum())
        if seen != 7901:
            raise RuntimeError(f"B0-LN expected 7,901 samples per epoch, saw {seen}")
        seconds = time.perf_counter() - epoch_started
        row: dict[str, float] = {"epoch": float(epoch), **{name: value / seen for name, value in totals.items()}}
        row.update(
            {
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
                "ema_momentum": float(_cosine(float(train["ema_start"]), 1.0, max(0, global_step - 1), total_steps)),
                "seconds": seconds,
                "samples_per_second": seen / seconds,
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
        )
        for key, (value, count) in transition_totals.items():
            row[key] = value / count if count else float("nan")
        history.append(row)
        write_csv(history, output / "pretrain_metrics.csv")
        print(json.dumps(row, sort_keys=True), flush=True)
        if epoch % int(config["monitor"]["evaluation_interval_epochs"]) == 0:
            evaluate(epoch)
            if epoch >= int(config["adaptive_duration"]["minimum_epochs"]) and failures >= int(config["adaptive_duration"]["consecutive_non_improving_monitors"]):
                early_stopped = True
                stopping_epoch = epoch
                break

    monitor.plot()
    plot_history(history, ["total", "prediction", "regularizer", "variance", "covariance"], output / "pretrain_losses.png", "B0-LN pretraining losses")
    plot_history(history, ["embedding_std"], output / "embedding_std.png", "B0-LN student embedding standard deviation")
    raw_best = max(monitor.history, key=lambda value: (float(value["linear_val_macro_f1"]), -int(value["epoch"])))
    checkpoint = output / "checkpoints" / "best.pt"
    selection = {
        "selected_epoch": best_epoch,
        "selection_metric": "validation_linear_macro_f1",
        "selection_score": best_score,
        "raw_best_monitor_epoch": int(raw_best["epoch"]),
        "raw_best_monitor_score": float(raw_best["linear_val_macro_f1"]),
        "horizon_epochs": 300,
        "warmup_epochs": 30,
        "minimum_delta_macro_f1": 0.005,
        "selection_split": "validation",
        "test_split_used_for_selection": False,
        "transductive_ssl": True,
        "early_stopped": early_stopped,
        "stopping_epoch": stopping_epoch,
        "adaptive_stopping": {
            "minimum_epochs": 100,
            "first_counted_monitor_epoch": 110,
            "consecutive_non_improving_monitors_to_stop": 5,
            "history": adaptive_history,
        },
        "checkpoint_sha256": sha256_path(checkpoint),
    }
    atomic_json_dump(selection, output / "duration_selection.json")
    verify_protected_manifest(output)
    verify_implementation_manifest(output)
    result = {
        "epochs_completed": stopping_epoch,
        "configured_horizon_epochs": 300,
        "best_epoch": best_epoch,
        "best_validation_linear_macro_f1": best_score,
        "total_seconds": time.perf_counter() - started,
        "final_metrics": history[-1],
        "protected_references_unchanged": True,
        "protected_manifest_entries": len(protected["protected_paths"]),
        "implementation_manifest_entries": len(implementation["protected_paths"]),
        "selection": selection,
    }
    atomic_json_dump(result, output / "pretrain_summary.json")
    return result
