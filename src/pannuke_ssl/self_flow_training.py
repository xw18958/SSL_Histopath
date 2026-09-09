"""Training loop for the capacity-matched pixel-space Self-Flow baseline."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from .data import build_ssl_loader
from .models import update_ema
from .monitor import SSLRepresentationMonitor, improves_macro_f1
from .self_flow import PixelSelfFlowB32, make_self_flow_teacher, self_flow_objective
from .training import _cosine, _learning_rate, _make_optimizer, _save_final_student, _weight_decay
from .utils import atomic_json_dump, plot_history, seed_everything, write_csv


def build_self_flow_models(config: dict, device: torch.device) -> tuple[PixelSelfFlowB32, PixelSelfFlowB32]:
    model = config["model"]
    student = PixelSelfFlowB32(
        image_size=int(model["image_size"]),
        patch_size=int(model["patch_size"]),
        hidden_size=int(model["hidden_size"]),
        depth=int(model["depth"]),
        num_heads=int(model["num_heads"]),
        mlp_ratio=float(model["mlp_ratio"]),
        student_rep_layer=int(model["student_rep_layer"]),
        teacher_rep_layer=int(model["teacher_rep_layer"]),
    ).to(device)
    teacher = make_self_flow_teacher(student).to(device)
    return student, teacher


def _validate_config(config: dict) -> None:
    model = config["model"]
    train = config["train"]
    if int(config["seed"]) != 20260903:
        raise ValueError("Fair benchmark seed is fixed to 20260903")
    expected_model = {
        "image_size": 256,
        "patch_size": 32,
        "hidden_size": 768,
        "depth": 8,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "student_rep_layer": 2,
        "teacher_rep_layer": 6,
        "mask_ratio": 0.25,
        "representation_weight": 0.5,
        "pixel_normalization": "minus_one_to_one",
        "timestep_distribution": "uniform",
        "class_conditioning": False,
        "position_embedding": "fixed_2d_sincos",
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise ValueError(f"Self-Flow benchmark fixes model.{key}={expected!r}; got {model.get(key)!r}")
    fixed_train = {
        "epochs": 300,
        "batch_size": 128,
        "effective_batch_size": 128,
        "learning_rate": 1e-4,
        "minimum_learning_rate": 1e-6,
        "weight_decay": 0.04,
        "final_weight_decay": 0.40,
        "warmup_fraction": 0.10,
        "ema_start": 0.996,
        "gradient_clip_norm": 1.0,
        "bf16": True,
    }
    for key, expected in fixed_train.items():
        if train.get(key) != expected:
            raise ValueError(f"Self-Flow benchmark fixes train.{key}={expected!r}; got {train.get(key)!r}")
    if train.get("resume") not in (None, ""):
        raise ValueError("Duration-selection Self-Flow run is intentionally non-resumable")
    monitor = config.get("monitor", {})
    if not bool(monitor.get("enabled", True)):
        raise ValueError("Validation-only representation monitoring must stay enabled")
    if int(monitor.get("evaluation_interval_epochs", 10)) != 10:
        raise ValueError("Checkpoint monitoring interval is fixed to 10 epochs")


def run_self_flow_pretraining(config: dict) -> dict[str, Any]:
    """Run the fair PanNuke Self-Flow comparison without touching B0/I-JEPA files."""
    if not torch.cuda.is_available():
        raise RuntimeError("Self-Flow pretraining requires CUDA")
    _validate_config(config)
    seed_everything(int(config["seed"]))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda")
    train = config["train"]
    model_config = config["model"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke_path = output_dir / "smoke_test.json"
    if not smoke_path.exists():
        raise FileNotFoundError(f"Run the Self-Flow smoke test first: {smoke_path}")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if not bool(smoke.get("passed", False)):
        raise RuntimeError("Self-Flow smoke test did not pass")
    if (output_dir / "pretrain_metrics.csv").exists():
        raise FileExistsError(f"Refusing to overwrite an existing duration run: {output_dir}")
    atomic_json_dump(config, output_dir / "resolved_config.json")

    loader, _ = build_ssl_loader(
        config["data_root"],
        batch_size=int(train["batch_size"]),
        num_workers=int(train["num_workers"]),
        cache_in_ram=bool(train.get("cache_in_ram", True)),
    )
    if len(loader.dataset) != 7901 or getattr(loader.dataset, "include_label", True):
        raise RuntimeError("Self-Flow SSL must see exactly 7,901 PanNuke images and no labels")

    student, teacher = build_self_flow_models(config, device)
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    trainable_parameters = sum(parameter.numel() for parameter in trainable)
    optimizer = _make_optimizer(trainable, float(train["learning_rate"]), float(train["weight_decay"]))

    batch_size = int(train["batch_size"])
    effective_batch = int(train.get("effective_batch_size", batch_size))
    if effective_batch < batch_size or effective_batch % batch_size:
        raise ValueError("effective_batch_size must be a positive multiple of batch_size")
    accumulation_steps = effective_batch // batch_size
    epochs = int(train["epochs"])
    optimizer_steps_per_epoch = math.ceil(len(loader) / accumulation_steps)
    total_steps = epochs * optimizer_steps_per_epoch
    global_step = 0

    monitor_config = config["monitor"]
    monitor = SSLRepresentationMonitor(
        monitor_config,
        data_root=config["data_root"],
        output_dir=output_dir,
        seed=int(config["seed"]),
        device=device,
    )
    history: list[dict[str, float]] = []
    best_epoch = 0
    baseline = monitor.evaluate(student, epoch=0)
    best_monitor_score = float(baseline["linear_val_macro_f1"])
    _save_final_student(
        output_dir / "checkpoints" / "best.pt",
        student=student,
        config=config,
        epoch=0,
        history=history,
        selection={
            "metric": "validation_linear_macro_f1",
            "value": best_monitor_score,
            "minimum_delta": float(monitor_config.get("minimum_delta_macro_f1", 0.005)),
            "selection_split": "validation",
            "representation": "clean_t0_final_layer_mean_pool",
            "test_split_used_for_selection": False,
            "transductive_ssl": True,
        },
    )
    print(json.dumps({"monitor": baseline, "trainable_parameters": trainable_parameters}, sort_keys=True), flush=True)

    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    use_bf16 = bool(train["bf16"])

    metric_names = (
        "total",
        "flow",
        "representation",
        "mean_t",
        "mean_s",
        "mean_student_tau",
        "mean_teacher_tau",
        "harder_token_fraction",
        "mask_fraction",
        "velocity_target_rms",
    )

    for epoch in range(1, epochs + 1):
        student.train()
        teacher.eval()
        torch.cuda.reset_peak_memory_stats()
        totals = {name: 0.0 for name in metric_names}
        seen = 0
        epoch_started = time.perf_counter()

        for batch_index, uint8_images in enumerate(loader):
            images = uint8_images.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                total_loss, metrics = self_flow_objective(
                    student,
                    teacher,
                    images,
                    mask_ratio=float(model_config["mask_ratio"]),
                    representation_weight=float(model_config["representation_weight"]),
                    student_rep_layer=int(model_config["student_rep_layer"]),
                    teacher_rep_layer=int(model_config["teacher_rep_layer"]),
                )
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"Non-finite Self-Flow loss at epoch={epoch}, batch={batch_index}")
            (total_loss / accumulation_steps).backward()

            is_update = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader)
            if is_update:
                gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, float(train["gradient_clip_norm"]))
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("Non-finite Self-Flow gradient norm")
                learning_rate = _learning_rate(train, global_step, total_steps)
                weight_decay = _weight_decay(train, global_step, total_steps)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                    group["weight_decay"] = weight_decay
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                momentum = _cosine(float(train["ema_start"]), 1.0, global_step, total_steps)
                update_ema(student, teacher, momentum)
                global_step += 1

            count = images.shape[0]
            seen += count
            for name in metric_names:
                totals[name] += float(metrics[name].detach()) * count

        if seen != 7901:
            raise RuntimeError(f"Expected 7,901 samples per epoch, saw {seen}")
        seconds = time.perf_counter() - epoch_started
        row = {"epoch": epoch, **{name: value / seen for name, value in totals.items()}}
        row.update(
            {
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
                "ema_momentum": _cosine(float(train["ema_start"]), 1.0, max(0, global_step - 1), total_steps),
                "seconds": seconds,
                "samples": seen,
                "samples_per_second": seen / seconds,
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                "trainable_parameters": trainable_parameters,
            }
        )
        history.append(row)
        write_csv(history, output_dir / "pretrain_metrics.csv")
        print(json.dumps(row, sort_keys=True), flush=True)

        if epoch % int(monitor_config["evaluation_interval_epochs"]) == 0:
            monitor_row = monitor.evaluate(student, epoch=epoch)
            candidate = float(monitor_row["linear_val_macro_f1"])
            if improves_macro_f1(
                candidate,
                best_monitor_score,
                float(monitor_config.get("minimum_delta_macro_f1", 0.005)),
            ):
                best_monitor_score = candidate
                best_epoch = epoch
                _save_final_student(
                    output_dir / "checkpoints" / "best.pt",
                    student=student,
                    config=config,
                    epoch=best_epoch,
                    history=history,
                    selection={
                        "metric": "validation_linear_macro_f1",
                        "value": best_monitor_score,
                        "minimum_delta": float(monitor_config.get("minimum_delta_macro_f1", 0.005)),
                        "selection_split": "validation",
                        "representation": "clean_t0_final_layer_mean_pool",
                        "test_split_used_for_selection": False,
                        "transductive_ssl": True,
                    },
                )
            print(
                json.dumps(
                    {"monitor": monitor_row, "best_epoch": best_epoch, "best_score": best_monitor_score},
                    sort_keys=True,
                ),
                flush=True,
            )

    plot_history(
        history,
        ["total", "flow", "representation"],
        output_dir / "pretrain_losses.png",
        "Self-Flow pixel-space pretraining losses",
    )
    plot_history(
        monitor.history,
        ["linear_val_macro_f1", "knn_val_macro_f1"],
        output_dir / "representation_validation_curves.png",
        "Self-Flow validation-only representation monitoring",
    )
    plot_history(
        monitor.history,
        [
            "feature_embedding_std",
            "feature_effective_rank_fraction",
            "feature_mean_pairwise_cosine",
            "feature_near_constant_fraction",
        ],
        output_dir / "representation_health.png",
        "Self-Flow representation health",
    )

    raw_best = max(monitor.history, key=lambda row: (float(row["linear_val_macro_f1"]), -float(row["epoch"])))
    selection = {
        "selected_epoch": best_epoch,
        "selection_score": best_monitor_score,
        "raw_best_monitor_epoch": int(raw_best["epoch"]),
        "raw_best_monitor_score": float(raw_best["linear_val_macro_f1"]),
        "horizon_epochs": epochs,
        "minimum_delta_macro_f1": float(monitor_config.get("minimum_delta_macro_f1", 0.005)),
        "selection_split": "validation",
        "test_split_used_for_selection": False,
        "representation": "clean_t0_final_layer_mean_pool",
        "transductive_ssl": True,
    }
    atomic_json_dump(selection, output_dir / "duration_selection.json")
    result = {
        "method": "Self-Flow-Pixel-B32",
        "adaptation": "raw RGB pixel patches; no pretrained VAE; no class conditioning",
        "trainable_parameters": trainable_parameters,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_validation_linear_macro_f1": best_monitor_score,
        "total_seconds": time.perf_counter() - started,
        "final_metrics": history[-1],
        "selection": selection,
    }
    atomic_json_dump(result, output_dir / "pretrain_summary.json")
    return result


def load_self_flow_checkpoint(checkpoint_path: str | Path, device: torch.device) -> PixelSelfFlowB32:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    student, _ = build_self_flow_models(config, device)
    student.load_state_dict(checkpoint["student"], strict=True)
    student.eval()
    return student
