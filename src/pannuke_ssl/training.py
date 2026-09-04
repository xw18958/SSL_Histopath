from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import build_ssl_loader
from .degradations import DegradationEndpoints, degrade_batch, sample_adjacent_transitions
from .losses import b0_loss
from .monitor import SSLRepresentationMonitor, improves_macro_f1
from .models import B0Predictor, FreshPLIPVisionEncoder, make_teacher, update_ema
from .utils import atomic_json_dump, plot_history, seed_everything, write_csv


def _endpoints(path: str | Path) -> DegradationEndpoints:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    return DegradationEndpoints(
        defocus_radius=float(value["defocus"]["selected_radius"]),
        resolution_factor=float(value["resolution"]["selected_factor"]),
    )


def _cosine(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return end
    progress = min(1.0, step / (total_steps - 1))
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


def _learning_rate(config: dict, step: int, total_steps: int) -> float:
    peak = float(config["learning_rate"])
    minimum = float(config["minimum_learning_rate"])
    warmup = max(1, int(total_steps * float(config["warmup_fraction"])))
    if step < warmup:
        return peak * (step + 1) / warmup
    return _cosine(peak, minimum, step - warmup, max(1, total_steps - warmup))


def _make_optimizer(parameters, learning_rate: float, weight_decay: float) -> torch.optim.Optimizer:
    kwargs: dict[str, Any] = {"lr": learning_rate, "weight_decay": weight_decay}
    if torch.cuda.is_available():
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(parameters, **kwargs)
    except (RuntimeError, TypeError):
        kwargs.pop("fused", None)
        return torch.optim.AdamW(parameters, **kwargs)


def _weight_decay(config: dict, step: int, total_steps: int) -> float:
    """Cosine weight-decay schedule; default preserves the original constant WD."""
    start = float(config["weight_decay"])
    end = float(config.get("final_weight_decay", start))
    return _cosine(start, end, step, total_steps)


def _model_state(module: nn.Module) -> dict[str, torch.Tensor]:
    original = getattr(module, "_orig_mod", module)
    return original.state_dict()


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    global_step: int,
    student: nn.Module,
    teacher: nn.Module,
    predictor: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict,
    history: list[dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "student": _model_state(student),
        "teacher": _model_state(teacher),
        "predictor": _model_state(predictor),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "history": history,
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _save_final_student(
    path: Path,
    *,
    student: nn.Module,
    config: dict,
    epoch: int,
    history: list[dict[str, float]],
    selection: dict[str, Any] | None = None,
) -> None:
    """Persist only the downstream artifact; predictor and teacher are discarded."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    state: dict[str, Any] = {"student": _model_state(student), "config": config, "epoch": epoch, "history": history}
    if selection is not None:
        state["selection"] = selection
    torch.save(state, temporary)
    temporary.replace(path)


def _restore_rng(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def build_b0_models(config: dict, device: torch.device) -> tuple[nn.Module, nn.Module, nn.Module]:
    model = config["model"]
    if int(model["image_size"]) != 256 or int(model["patch_size"]) != 32:
        raise ValueError("B0 is fixed to native 256x256 PanNuke inputs and 32x32 patches")
    student = FreshPLIPVisionEncoder(config["plip_config_dir"], image_size=256).to(device)
    if student.hidden_size != 768 or student.num_patches != 64:
        raise ValueError("The local PLIP config is not the expected ViT-B/32 architecture")
    teacher = make_teacher(student).to(device)
    predictor = B0Predictor(
        input_dim=768,
        predictor_dim=int(model["predictor_dim"]),
        depth=int(model["predictor_depth"]),
        heads=int(model["predictor_heads"]),
        mlp_ratio=int(model["predictor_mlp_ratio"]),
        dropout=float(model["dropout"]),
    ).to(device)
    return student, teacher, predictor


def run_pretraining(config: dict) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("B0 pretraining requires CUDA")
    seed_everything(int(config["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    train = config["train"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(config, output_dir / "resolved_config.json")
    endpoints = _endpoints(config["endpoints_json"])
    loader, _ = build_ssl_loader(
        config["data_root"],
        batch_size=int(train["batch_size"]),
        num_workers=int(train["num_workers"]),
        cache_in_ram=bool(train.get("cache_in_ram", True)),
    )
    student, teacher, predictor = build_b0_models(config, device)
    trainable = list(student.parameters()) + list(predictor.parameters())
    optimizer = _make_optimizer(trainable, float(train["learning_rate"]), float(train["weight_decay"]))
    effective_batch = int(train.get("effective_batch_size", train["batch_size"]))
    batch_size = int(train["batch_size"])
    if effective_batch < batch_size or effective_batch % batch_size:
        raise ValueError("effective_batch_size must be a positive multiple of batch_size")
    accumulation_steps = effective_batch // batch_size
    epochs = int(train["epochs"])
    optimizer_steps_per_epoch = math.ceil(len(loader) / accumulation_steps)
    total_steps = epochs * optimizer_steps_per_epoch
    start_epoch = 0
    global_step = 0
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_epoch = 0
    monitor_config = config.get("monitor", {})
    monitor_enabled = bool(monitor_config.get("enabled", False))
    monitor: SSLRepresentationMonitor | None = None
    best_monitor_score = float("-inf")
    resume = train.get("resume")
    if monitor_enabled and resume:
        raise ValueError("Duration monitoring stores a student-only best checkpoint and cannot resume optimizer state")
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        student.load_state_dict(checkpoint["student"])
        teacher.load_state_dict(checkpoint["teacher"])
        predictor.load_state_dict(checkpoint["predictor"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        history = list(checkpoint["history"])
        best_loss = min(float(row["total"]) for row in history)
        best_epoch = min(history, key=lambda row: float(row["total"]))["epoch"]
        _restore_rng(checkpoint["rng"])
    if bool(train.get("compile", False)):
        student = torch.compile(student)
        predictor = torch.compile(predictor)
    optimizer.zero_grad(set_to_none=True)
    run_started = time.perf_counter()
    if monitor_enabled:
        monitor = SSLRepresentationMonitor(
            monitor_config,
            data_root=config["data_root"],
            output_dir=output_dir,
            seed=int(config["seed"]),
            device=device,
        )
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
                "transductive_ssl": True,
            },
        )
        print(json.dumps({"monitor": baseline}, sort_keys=True), flush=True)
    for epoch in range(start_epoch, epochs):
        student.train()
        predictor.train()
        torch.cuda.reset_peak_memory_stats()
        totals = {name: 0.0 for name in ("total", "prediction", "regularizer", "variance", "covariance", "embedding_std")}
        transition_totals = {
            f"prediction_{family}_{source:03d}_to_{target:03d}": [0.0, 0]
            for family in ("defocus", "resolution")
            for source, target in ((100, 75), (75, 50), (50, 25), (25, 0))
        }
        seen = 0
        epoch_started = time.perf_counter()
        for batch_index, uint8_images in enumerate(loader):
            images = uint8_images.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
            actions, source_severity, target_severity, delta = sample_adjacent_transitions(images.shape[0], device)
            source_images = degrade_batch(images, actions, source_severity, endpoints)
            target_images = degrade_batch(images, actions, target_severity, endpoints)
            use_bf16 = bool(train.get("bf16", True))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                student_tokens = student(source_images)
                prediction = predictor(student_tokens, source_severity, actions, delta)
                with torch.no_grad():
                    teacher_tokens = teacher(target_images)
            losses = b0_loss(
                prediction,
                teacher_tokens.detach(),
                student_tokens,
                lambda_reg=float(train["lambda_reg"]),
                covariance_weight=float(train["covariance_weight"]),
                variance_target=float(train["variance_target"]),
            )
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"Non-finite B0 loss at epoch={epoch + 1}, batch={batch_index}")
            (losses["total"] / accumulation_steps).backward()
            is_update = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader)
            if is_update:
                gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, float(train["gradient_clip_norm"]))
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("Non-finite gradient norm")
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
            for name in totals:
                totals[name] += float(losses[name].detach()) * count
            with torch.no_grad():
                per_example_prediction = F.smooth_l1_loss(
                    prediction.float(), teacher_tokens.detach().float(), reduction="none"
                ).mean(dim=(1, 2))
            for action, family in ((0, "defocus"), (1, "resolution")):
                mask = actions == action
                if not torch.any(mask):
                    continue
                source_values = source_severity[mask]
                for source_anchor in (1.0, 0.75, 0.5, 0.25):
                    anchor_mask = mask.clone()
                    anchor_mask[mask] = torch.isclose(source_values, torch.tensor(source_anchor, device=device))
                    if not torch.any(anchor_mask):
                        continue
                    target_anchor = source_anchor - 0.25
                    key = f"prediction_{family}_{int(round(source_anchor * 100)):03d}_to_{int(round(target_anchor * 100)):03d}"
                    value = per_example_prediction[anchor_mask]
                    transition_totals[key][0] += float(value.sum().detach())
                    transition_totals[key][1] += int(value.numel())
        seconds = time.perf_counter() - epoch_started
        row = {"epoch": epoch + 1, **{name: value / seen for name, value in totals.items()}}
        row.update(
            {
                "learning_rate": optimizer.param_groups[0]["lr"],
                "weight_decay": optimizer.param_groups[0]["weight_decay"],
                "ema_momentum": _cosine(float(train["ema_start"]), 1.0, max(0, global_step - 1), total_steps),
                "seconds": seconds,
                "samples_per_second": seen / seconds,
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
        )
        for key, (value, count) in transition_totals.items():
            row[key] = value / count if count else float("nan")
        history.append(row)
        write_csv(history, output_dir / "pretrain_metrics.csv")
        print(json.dumps(row, sort_keys=True), flush=True)
        if not monitor_enabled and row["total"] < best_loss:
            best_loss = row["total"]
            best_epoch = epoch + 1
            _save_checkpoint(
                output_dir / "checkpoints" / "best.pt",
                epoch=epoch + 1,
                global_step=global_step,
                student=student,
                teacher=teacher,
                predictor=predictor,
                optimizer=optimizer,
                config=config,
                history=history,
            )
        if monitor is not None and (epoch + 1) % int(monitor_config.get("evaluation_interval_epochs", 10)) == 0:
            monitor_row = monitor.evaluate(student, epoch=epoch + 1)
            candidate = float(monitor_row["linear_val_macro_f1"])
            if improves_macro_f1(candidate, best_monitor_score, float(monitor_config.get("minimum_delta_macro_f1", 0.005))):
                best_monitor_score = candidate
                best_epoch = epoch + 1
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
                        "transductive_ssl": True,
                    },
                )
            print(json.dumps({"monitor": monitor_row, "best_epoch": best_epoch, "best_score": best_monitor_score}, sort_keys=True), flush=True)
    final_path = output_dir / "checkpoints" / "best.pt"
    if monitor is None:
        best_checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
        best_epoch = int(best_checkpoint["epoch"])
        temporary = final_path.with_suffix(".student.tmp")
        torch.save(
            {
                "student": best_checkpoint["student"],
                "config": config,
                "epoch": best_epoch,
                "history": history,
                "selection": {"metric": "lowest_epoch_total_loss", "value": best_loss},
            },
            temporary,
        )
        temporary.replace(final_path)
    else:
        monitor.plot()
    plot_history(history, ["total", "prediction", "regularizer", "variance", "covariance"], output_dir / "pretrain_losses.png", "B0 pretraining losses")
    plot_history(history, ["embedding_std"], output_dir / "embedding_std.png", "Student embedding standard deviation")
    result = {
        "checkpoint": str(final_path),
        "epochs": epochs,
        "best_epoch": int(best_epoch),
        "total_seconds": time.perf_counter() - run_started,
        "final_metrics": history[-1],
    }
    if monitor is None:
        result["best_total_loss"] = best_loss
    else:
        result.update(
            {
                "best_validation_linear_macro_f1": best_monitor_score,
                "selection_split": "validation",
                "transductive_ssl": True,
                "monitor_epochs": [int(row["epoch"]) for row in monitor.history],
            }
        )
        raw_best = max(monitor.history, key=lambda row: (float(row["linear_val_macro_f1"]), -int(row["epoch"])))
        final_monitor = monitor.history[-1]
        earlier_scores = [float(row["linear_val_macro_f1"]) for row in monitor.history[:-1]]
        minimum_delta = float(monitor_config.get("minimum_delta_macro_f1", 0.005))
        requires_extension = bool(
            int(final_monitor["epoch"]) == epochs
            and earlier_scores
            and float(final_monitor["linear_val_macro_f1"]) > max(earlier_scores) + minimum_delta
        )
        duration_selection = {
            "selected_epoch": int(best_epoch),
            "selection_metric": "validation_linear_macro_f1",
            "selection_score": best_monitor_score,
            "raw_best_monitor_epoch": int(raw_best["epoch"]),
            "raw_best_monitor_score": float(raw_best["linear_val_macro_f1"]),
            "horizon_epochs": epochs,
            "requires_extension": requires_extension,
            "extension_rule": "extend only when the final monitored score exceeds every earlier score by minimum_delta",
            "minimum_delta_macro_f1": minimum_delta,
            "test_split_used": False,
            "transductive_ssl": True,
        }
        atomic_json_dump(duration_selection, output_dir / "duration_selection.json")
        result["duration_selection"] = duration_selection
    atomic_json_dump(result, output_dir / "pretrain_summary.json")
    return result


def load_student_checkpoint(config_dir: str | Path, checkpoint_path: str | Path, device: torch.device) -> FreshPLIPVisionEncoder:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder = FreshPLIPVisionEncoder(config_dir, image_size=256).to(device)
    state = checkpoint["student"]
    if any(key.startswith("_orig_mod.") for key in state):
        state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    encoder.load_state_dict(state)
    encoder.eval().requires_grad_(False)
    return encoder
