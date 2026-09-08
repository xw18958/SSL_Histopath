"""Change-JEPA: degradation-dynamics joint-embedding predictive SSL.

The controlled first experiment keeps the B0 data, degradation calibration,
fresh PLIP ViT-B/32 student, EMA teacher, optimizer/schedules, validation
monitor, checkpoint criterion, and downstream clean-image linear-probe protocol.

The SSL task is changed to:

    source x_s = the more-degraded endpoint of a B0 adjacent transition
    target x_t = the adjacent less-degraded endpoint
    change image x_delta = ((x_t - x_s) + 1) / 2

    P(S(x_s), source_state, action, delta_s, patch_position)
        -> T_ema(x_delta)

The predictor uses one JEPA-style degradation query per spatial patch. The first
controlled run deliberately retains B0's adjacent transitions, so delta_s is
always -0.25; this isolates the new target/query mechanism before any multi-step
transition experiment.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .change_jepa import build_models, objective
from .data import build_ssl_loader
from .degradations import degrade_batch, sample_adjacent_transitions
from .models import update_ema
from .monitor import SSLRepresentationMonitor, improves_macro_f1
from .training import _cosine, _endpoints, _learning_rate, _make_optimizer, _save_final_student, _weight_decay
from .utils import atomic_json_dump, plot_history, seed_everything, write_csv


EXPECTED_MODEL = {
    "image_size": 256,
    "patch_size": 32,
    "hidden_size": 768,
    "predictor_dim": 384,
    "predictor_depth": 2,
    "predictor_heads": 6,
    "predictor_mlp_ratio": 4,
    "dropout": 0.0,
}

EXPECTED_OBJECTIVE = {
    "transition_sampling": "b0_adjacent_more_to_less_degraded",
    "change_target": "mapped_signed_pixel_difference",
    "change_mapping": "((target-source)+1)/2",
    "teacher_target_layer_norm": True,
    "loss": "smooth_l1",
    "vicreg": False,
    "query_conditioning": "position+action+source_state+delta",
}

EXPECTED_TRAIN = {
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
    "ema_start": 0.996,
    "gradient_clip_norm": 5.0,
    "bf16": True,
    "compile": False,
    "resume": None,
}

EXPECTED_MONITOR = {
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
    "early_stopping_patience_evaluations": 3,
}


def validate_config(config: dict[str, Any]) -> None:
    if int(config["seed"]) != 20260903:
        raise ValueError("Change-JEPA must use the matched seed 20260903")
    if config["model"] != EXPECTED_MODEL:
        raise ValueError(f"Change-JEPA model settings must equal {EXPECTED_MODEL}")
    if config["objective"] != EXPECTED_OBJECTIVE:
        raise ValueError(f"Change-JEPA objective settings must equal {EXPECTED_OBJECTIVE}")

    train = config["train"]
    mismatched_train = {k: (train.get(k), v) for k, v in EXPECTED_TRAIN.items() if train.get(k) != v}
    if mismatched_train:
        raise ValueError(f"Change-JEPA train settings drifted from the locked protocol: {mismatched_train}")

    monitor = config["monitor"]
    mismatched_monitor = {
        k: (monitor.get(k), v) for k, v in EXPECTED_MONITOR.items() if monitor.get(k) != v
    }
    if mismatched_monitor:
        raise ValueError(f"Change-JEPA monitor settings drifted from the locked protocol: {mismatched_monitor}")
    if int(monitor["evaluation_interval_epochs"]) * int(monitor["early_stopping_patience_evaluations"]) != 30:
        raise ValueError("Change-JEPA early-stopping patience must equal three 10-epoch monitors = 30 epochs")
    if Path(config["output_dir"]).name != "change_jepa_duration_pilot":
        raise ValueError("output_dir must end in change_jepa_duration_pilot")


def should_early_stop(non_improving_evaluations: int, patience_evaluations: int) -> bool:
    if non_improving_evaluations < 0:
        raise ValueError("non_improving_evaluations must be non-negative")
    if patience_evaluations <= 0:
        raise ValueError("patience_evaluations must be positive")
    return non_improving_evaluations >= patience_evaluations


@torch.no_grad()
def _batch_diagnostics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    source_images: torch.Tensor,
    target_images: torch.Tensor,
    student_tokens: torch.Tensor,
) -> dict[str, torch.Tensor]:
    pred = prediction.float()
    tgt = target.float()
    return {
        "prediction_cosine": F.cosine_similarity(pred.flatten(1), tgt.flatten(1), dim=1),
        "target_squared_sum": tgt.square().sum(),
        "target_elements": torch.tensor(tgt.numel(), device=tgt.device, dtype=torch.long),
        "signed_change_abs_sum": (target_images.float() - source_images.float()).abs().sum(),
        "change_elements": torch.tensor(target_images.numel(), device=tgt.device, dtype=torch.long),
        "embedding_std": student_tokens.float().mean(dim=1).std(dim=0, unbiased=True).mean(),
    }


def run_change_jepa_pretraining(config: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Change-JEPA pretraining requires CUDA")
    validate_config(config)

    seed_everything(int(config["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    train = config["train"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "pretrain_metrics.csv").exists():
        raise FileExistsError(f"Refusing to overwrite existing Change-JEPA run: {output_dir}")
    atomic_json_dump(config, output_dir / "resolved_config.json")

    endpoints = _endpoints(config["endpoints_json"])
    loader, _ = build_ssl_loader(
        config["data_root"],
        batch_size=int(train["batch_size"]),
        num_workers=int(train["num_workers"]),
        cache_in_ram=bool(train["cache_in_ram"]),
    )
    if len(loader.dataset) != 7901:
        raise RuntimeError(f"Expected 7,901 unlabeled SSL images, got {len(loader.dataset)}")

    student, teacher, predictor = build_models(config, device)
    trainable = list(student.parameters()) + list(predictor.parameters())
    optimizer = _make_optimizer(
        trainable,
        float(train["learning_rate"]),
        float(train["weight_decay"]),
    )

    epochs = int(train["epochs"])
    total_steps = epochs * len(loader)  # fixed 300-epoch schedule even when early stopping fires
    global_step = 0
    history: list[dict[str, float]] = []
    run_started = time.perf_counter()

    monitor_config = config["monitor"]
    evaluation_interval = int(monitor_config["evaluation_interval_epochs"])
    patience_evaluations = int(monitor_config["early_stopping_patience_evaluations"])
    monitor = SSLRepresentationMonitor(
        monitor_config,
        data_root=config["data_root"],
        output_dir=output_dir,
        seed=int(config["seed"]),
        device=device,
    )

    baseline = monitor.evaluate(student, epoch=0)
    best_monitor_score = float(baseline["linear_val_macro_f1"])
    best_epoch = 0
    non_improving_evaluations = 0
    stopped_early = False
    early_stop_epoch: int | None = None
    minimum_delta = float(monitor_config["minimum_delta_macro_f1"])

    _save_final_student(
        output_dir / "checkpoints" / "best.pt",
        student=student,
        config=config,
        epoch=0,
        history=history,
        selection={
            "metric": "validation_linear_macro_f1",
            "value": best_monitor_score,
            "minimum_delta": minimum_delta,
            "selection_split": "validation",
            "transductive_ssl": True,
        },
    )
    print(
        json.dumps(
            {
                "monitor": baseline,
                "best_epoch": best_epoch,
                "best_score": best_monitor_score,
                "early_stopping_patience_evaluations": patience_evaluations,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        student.train()
        predictor.train()
        teacher.eval()
        torch.cuda.reset_peak_memory_stats()
        epoch_started = time.perf_counter()

        prediction_sum = 0.0
        cosine_sum = 0.0
        target_squared_sum = 0.0
        target_elements = 0
        signed_change_abs_sum = 0.0
        change_elements = 0
        embedding_std_sum = 0.0
        seen = 0
        transition_totals = {
            f"prediction_{family}_{source:03d}_to_{target:03d}": [0.0, 0]
            for family in ("defocus", "resolution")
            for source, target in ((100, 75), (75, 50), (50, 25), (25, 0))
        }

        for batch_index, uint8_images in enumerate(loader):
            images = uint8_images.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            ).div_(255.0)
            actions, source_severity, target_severity, delta = sample_adjacent_transitions(
                images.shape[0], device
            )
            expected_step = torch.full_like(source_severity, 0.25)
            if not torch.equal(source_severity - target_severity, expected_step):
                raise RuntimeError("Change-JEPA requires B0 adjacent source-target transitions")
            if not torch.equal(delta, -expected_step):
                raise RuntimeError("Change-JEPA controlled run requires delta_s=-0.25")

            source_images = degrade_batch(images, actions, source_severity, endpoints)
            target_images = degrade_batch(images, actions, target_severity, endpoints)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=bool(train["bf16"]),
            ):
                loss, prediction, target, student_tokens, _ = objective(
                    student,
                    teacher,
                    predictor,
                    source_images,
                    target_images,
                    source_severity,
                    actions,
                    delta,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite Change-JEPA loss at epoch={epoch}, batch={batch_index}"
                )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                float(train["gradient_clip_norm"]),
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite Change-JEPA gradient norm")

            learning_rate = _learning_rate(train, global_step, total_steps)
            weight_decay = _weight_decay(train, global_step, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
                group["weight_decay"] = weight_decay
            optimizer.step()

            momentum = _cosine(
                float(train["ema_start"]),
                1.0,
                global_step,
                total_steps,
            )
            update_ema(student, teacher, momentum)
            global_step += 1

            diagnostics = _batch_diagnostics(
                prediction,
                target,
                source_images,
                target_images,
                student_tokens,
            )
            count = images.shape[0]
            seen += count
            prediction_sum += float(loss.detach()) * count
            cosine_sum += float(diagnostics["prediction_cosine"].sum())
            target_squared_sum += float(diagnostics["target_squared_sum"])
            target_elements += int(diagnostics["target_elements"])
            signed_change_abs_sum += float(diagnostics["signed_change_abs_sum"])
            change_elements += int(diagnostics["change_elements"])
            embedding_std_sum += float(diagnostics["embedding_std"]) * count

            with torch.no_grad():
                per_example_prediction = F.smooth_l1_loss(
                    prediction.float(), target.float(), reduction="none"
                ).mean(dim=(1, 2))
            for action_value, family in ((0, "defocus"), (1, "resolution")):
                family_mask = actions == action_value
                if not torch.any(family_mask):
                    continue
                for source_anchor in (1.0, 0.75, 0.5, 0.25):
                    anchor_mask = family_mask & torch.isclose(
                        source_severity,
                        torch.tensor(source_anchor, device=device),
                    )
                    if not torch.any(anchor_mask):
                        continue
                    target_anchor = source_anchor - 0.25
                    key = (
                        f"prediction_{family}_{int(round(source_anchor * 100)):03d}"
                        f"_to_{int(round(target_anchor * 100)):03d}"
                    )
                    values = per_example_prediction[anchor_mask]
                    transition_totals[key][0] += float(values.sum())
                    transition_totals[key][1] += int(values.numel())

        seconds = time.perf_counter() - epoch_started
        if seen != 7901:
            raise RuntimeError(f"Expected to see 7,901 SSL images, saw {seen}")
        row: dict[str, float] = {
            "epoch": float(epoch),
            "prediction": prediction_sum / seen,
            "prediction_cosine": cosine_sum / seen,
            "target_rms": (target_squared_sum / max(1, target_elements)) ** 0.5,
            "signed_change_abs_mean": signed_change_abs_sum / max(1, change_elements),
            "embedding_std": embedding_std_sum / seen,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
            "ema_momentum": float(momentum),
            "seconds": seconds,
            "samples_per_second": seen / seconds,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        for key, (value, count) in transition_totals.items():
            row[key] = value / count if count else float("nan")
        history.append(row)
        write_csv(history, output_dir / "pretrain_metrics.csv")
        print(json.dumps(row, sort_keys=True), flush=True)

        if epoch % evaluation_interval == 0:
            monitor_row = monitor.evaluate(student, epoch=epoch)
            candidate = float(monitor_row["linear_val_macro_f1"])
            improved = improves_macro_f1(candidate, best_monitor_score, minimum_delta)
            if improved:
                best_monitor_score = candidate
                best_epoch = epoch
                non_improving_evaluations = 0
                _save_final_student(
                    output_dir / "checkpoints" / "best.pt",
                    student=student,
                    config=config,
                    epoch=best_epoch,
                    history=history,
                    selection={
                        "metric": "validation_linear_macro_f1",
                        "value": best_monitor_score,
                        "minimum_delta": minimum_delta,
                        "selection_split": "validation",
                        "transductive_ssl": True,
                    },
                )
            else:
                non_improving_evaluations += 1

            print(
                json.dumps(
                    {
                        "monitor": monitor_row,
                        "best_epoch": best_epoch,
                        "best_score": best_monitor_score,
                        "checkpoint_qualifying_improvement": improved,
                        "non_improving_evaluations": non_improving_evaluations,
                        "early_stopping_patience_evaluations": patience_evaluations,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

            if should_early_stop(non_improving_evaluations, patience_evaluations):
                stopped_early = True
                early_stop_epoch = epoch
                break

    plot_history(
        monitor.history,
        [
            "knn_val_macro_f1",
            "linear_val_macro_f1",
            "knn_val_balanced_accuracy",
            "linear_val_balanced_accuracy",
        ],
        output_dir / "representation_validation_curves.png",
        "Change-JEPA validation-only representation monitoring",
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
        "Change-JEPA representation health diagnostics",
    )
    plot_history(
        history,
        ["prediction", "prediction_cosine", "target_rms", "signed_change_abs_mean"],
        output_dir / "pretrain_diagnostics.png",
        "Change-JEPA pretraining diagnostics",
    )
    plot_history(
        history,
        ["embedding_std"],
        output_dir / "embedding_std.png",
        "Change-JEPA student embedding standard deviation",
    )

    final_path = output_dir / "checkpoints" / "best.pt"
    raw_best = max(
        monitor.history,
        key=lambda row: (float(row["linear_val_macro_f1"]), -int(row["epoch"])),
    )
    epochs_run = int(history[-1]["epoch"]) if history else 0
    duration_selection = {
        "selected_epoch": int(best_epoch),
        "selection_metric": "validation_linear_macro_f1",
        "selection_score": best_monitor_score,
        "raw_best_monitor_epoch": int(raw_best["epoch"]),
        "raw_best_monitor_score": float(raw_best["linear_val_macro_f1"]),
        "horizon_epochs": epochs,
        "epochs_run": epochs_run,
        "stopped_early": stopped_early,
        "early_stop_epoch": early_stop_epoch,
        "early_stopping_patience_evaluations": patience_evaluations,
        "early_stopping_patience_epochs": evaluation_interval * patience_evaluations,
        "early_stopping_rule": (
            "stop after three consecutive 10-epoch validation monitors without "
            ">0.005 macro-F1 improvement over the selected checkpoint"
        ),
        "minimum_delta_macro_f1": minimum_delta,
        "test_split_used": False,
        "transductive_ssl": True,
        "objective": "EMA representation of mapped signed pixel-change image",
        "transition_sampling": "B0 adjacent more-degraded to less-degraded",
    }
    atomic_json_dump(duration_selection, output_dir / "duration_selection.json")

    result = {
        "checkpoint": str(final_path),
        "best_epoch": int(best_epoch),
        "best_validation_linear_macro_f1": best_monitor_score,
        "epochs": epochs,
        "epochs_run": epochs_run,
        "stopped_early": stopped_early,
        "early_stop_epoch": early_stop_epoch,
        "monitor_epochs": [int(row["epoch"]) for row in monitor.history],
        "final_metrics": history[-1] if history else None,
        "total_seconds": time.perf_counter() - run_started,
        "duration_selection": duration_selection,
    }
    atomic_json_dump(result, output_dir / "pretrain_summary.json")
    return result
