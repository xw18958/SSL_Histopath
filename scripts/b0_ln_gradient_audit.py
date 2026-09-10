#!/usr/bin/env python
"""Diagnostic-only encoder-gradient audit for the existing B0-LN run.

The selected B0-LN artifact contains only the epoch-20 student.  This script
replays the original B0-LN optimizer trajectory in a separate process through
epoch 50, verifies the replayed epoch-20 student byte-for-byte against the
saved artifact, and only then measures gradients at epochs 1/5/10/20/50.

The replay is deliberately kept separate from the B0-LN training output.  It
does not write checkpoints, update the saved optimizer/EMA, construct a
balanced downstream loader, or access the held-out test split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from pannuke_ssl.b0_ln_training import (
    layer_norm_teacher_target,
    verify_protected_manifest,
)
from pannuke_ssl.config import load_yaml
from pannuke_ssl.data import all_source_rows, build_ssl_loader
from pannuke_ssl.parquet import build_source_index, preload_images
from pannuke_ssl.degradations import degrade_batch, sample_adjacent_transitions
from pannuke_ssl.losses import b0_loss, variance_covariance_loss
from pannuke_ssl.training import (
    _cosine,
    _endpoints,
    _learning_rate,
    _make_optimizer,
    _weight_decay,
    build_b0_models,
)
from pannuke_ssl.utils import atomic_json_dump, seed_everything


AUDIT_EPOCHS = (1, 5, 10, 20, 50)
EXPECTED_SSL_SAMPLES = 7901
EXPECTED_TOKEN_SHAPE = (64, 768)


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_state_hash(state: dict[str, torch.Tensor]) -> str:
    """Hash an ordered state dict, including key/dtype/shape metadata."""
    digest = hashlib.sha256()
    for name, value in state.items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError("Cannot write an empty audit CSV")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _assert_config(reference: dict[str, Any]) -> None:
    """Fail closed if the audit is pointed at a different protocol."""
    train = reference["train"]
    model = reference["model"]
    if int(reference["seed"]) != 20260903:
        raise RuntimeError("Gradient audit requires seed 20260903")
    if (int(model["image_size"]), int(model["patch_size"]), int(model["hidden_size"])) != (256, 32, 768):
        raise RuntimeError("Gradient audit requires the matched ViT-B/32 encoder")
    expected_train = {
        "epochs": 300,
        "batch_size": 128,
        "effective_batch_size": 128,
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
    for key, expected in expected_train.items():
        if train.get(key) != expected:
            raise RuntimeError(f"B0-LN training control changed: {key}={train.get(key)!r}, expected {expected!r}")


def _emulate_monitor_rng_reset(seed: int, device: torch.device) -> None:
    """Consume the exact GPU RNG effect of the monitor's fixed LBFGS probe.

    The real monitor resets torch's generators and constructs one Linear(768,
    19) on CUDA.  The monitor has no other stochastic CUDA operation.  Doing
    this small equivalent keeps the replay's transition sampling aligned
    without constructing a downstream train/validation loader.
    """
    torch.manual_seed(seed)
    probe = torch.nn.Linear(768, 19, device=device)
    del probe


def _fixed_diagnostic_batch(
    config: dict[str, Any],
    device: torch.device,
    *,
    batch_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Load eight immutable SSL images and a fixed four-by-two transition map."""
    source_index = build_source_index(Path(config["data_root"]))
    rows = all_source_rows(source_index)
    if len(rows) != EXPECTED_SSL_SAMPLES:
        raise RuntimeError(f"Expected {EXPECTED_SSL_SAMPLES} SSL source records, found {len(rows)}")
    selected = rows[:batch_size]
    cache = preload_images(selected, source_index)
    # Read the fixed records directly.  Constructing/iterating even a
    # shuffle=False DataLoader advances the CPU RNG's worker base seed and
    # would shift the original SSL loader's shuffle stream during replay.
    uint8_images = torch.stack(
        [
            torch.from_numpy(
                np.array(cache[(int(row["fold"]), int(row["sample_index"]))], copy=True)
            ).permute(2, 0, 1)
            for row in selected
        ]
    )
    folds = torch.tensor([int(row["fold"]) for row in selected], dtype=torch.int64)
    sample_indices = torch.tensor([int(row["sample_index"]) for row in selected], dtype=torch.int64)
    images = uint8_images.to(device=device, dtype=torch.float32).div_(255.0)

    # Use a private generator so constructing the diagnostic batch can never
    # perturb replay transition sampling or model initialization.
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    actions = torch.randint(0, 2, (batch_size,), device=device, generator=generator)
    target_anchor = torch.randint(0, 4, (batch_size,), device=device, generator=generator)
    anchors = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device)
    target = anchors[target_anchor]
    source = anchors[target_anchor + 1]
    delta = target - source
    records = []
    for index in range(batch_size):
        records.append(
            {
                "batch_index": index,
                "fold": int(folds[index]),
                "sample_index": int(sample_indices[index]),
                "action": int(actions[index]),
                "action_name": "defocus" if int(actions[index]) == 0 else "resolution",
                "source_severity": float(source[index]),
                "target_severity": float(target[index]),
                "delta": float(delta[index]),
            }
        )
    return images, actions, source, target, delta, records


def _flatten_grads(grads: Iterable[torch.Tensor | None], parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for grad, parameter in zip(grads, parameters, strict=True):
        if grad is None:
            pieces.append(torch.zeros_like(parameter, dtype=torch.float32).reshape(-1))
        else:
            pieces.append(grad.detach().float().reshape(-1))
    if not pieces:
        raise RuntimeError("Student encoder has no parameters")
    return torch.cat(pieces)


def _gradient_audit(
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    predictor: torch.nn.Module,
    images: torch.Tensor,
    actions: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    delta: torch.Tensor,
    endpoints: Any,
    train_config: dict[str, Any],
    epoch: int,
) -> dict[str, Any]:
    """Measure separate FP32 student-parameter gradients without updates."""
    student.eval()
    predictor.eval()
    teacher.eval()
    source_images = degrade_batch(images, actions, source, endpoints)
    target_images = degrade_batch(images, actions, target, endpoints)
    parameters = list(student.parameters())
    with torch.enable_grad():
        # The diagnostic is intentionally FP32.  The target normalization is
        # still exactly the B0-LN implementation, and no student normalization
        # is introduced.
        student_tokens = student(source_images)
        prediction = predictor(student_tokens, source, actions, delta)
        teacher_tokens = layer_norm_teacher_target(teacher, target_images)
        if tuple(student_tokens.shape[1:]) != EXPECTED_TOKEN_SHAPE:
            raise RuntimeError(f"Unexpected student shape: {tuple(student_tokens.shape)}")
        if tuple(teacher_tokens.shape) != tuple(student_tokens.shape):
            raise RuntimeError("Teacher/student token shapes do not match")
        if teacher_tokens.dtype != torch.float32 or teacher_tokens.requires_grad:
            raise RuntimeError("Teacher target must be detached FP32")

        values = b0_loss(
            prediction,
            teacher_tokens.detach(),
            student_tokens,
            lambda_reg=float(train_config["lambda_reg"]),
            covariance_weight=float(train_config["covariance_weight"]),
            variance_target=float(train_config["variance_target"]),
        )
        g_pred = torch.autograd.grad(
            values["prediction"],
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        g_reg = torch.autograd.grad(
            values["regularizer"] * float(train_config["lambda_reg"]),
            parameters,
            retain_graph=False,
            allow_unused=True,
        )
        pred_vector = _flatten_grads(g_pred, parameters)
        reg_vector = _flatten_grads(g_reg, parameters)
        pred_norm = torch.linalg.vector_norm(pred_vector)
        reg_norm = torch.linalg.vector_norm(reg_vector)
        denominator = pred_norm * reg_norm
        cosine = torch.dot(pred_vector, reg_vector) / denominator if float(denominator) > 0 else torch.tensor(float("nan"))
        raw_reg = values["regularizer"].detach().float()
        weighted_reg = raw_reg * float(train_config["lambda_reg"])
        scalars = {
            "prediction_loss": values["prediction"].detach().float(),
            "regularizer_loss": raw_reg,
            "weighted_regularizer_loss": weighted_reg,
            "total_loss": values["total"].detach().float(),
        }
        all_values = [pred_norm, reg_norm, cosine, *scalars.values()]
        if not all(torch.isfinite(value) for value in all_values):
            raise FloatingPointError(f"Non-finite gradient audit value at epoch {epoch}")

        # LayerNorm has per-token unit population variance (up to epsilon and
        # floating-point error).  These diagnostics are reported as evidence
        # that the intended target, not the student, was normalized.
        target_mean = teacher_tokens.float().mean(dim=-1).abs().max()
        target_var = teacher_tokens.float().var(dim=-1, unbiased=False).mean()
        return {
            "epoch": epoch,
            **{name: float(value) for name, value in scalars.items()},
            "lambda_reg": float(train_config["lambda_reg"]),
            "weighted_regularizer_loss": float(weighted_reg),
            "g_pred_norm": float(pred_norm),
            "g_reg_norm": float(reg_norm),
            "reg_pred_grad_ratio": float(reg_norm / pred_norm) if float(pred_norm) > 0 else float("inf"),
            "gradient_cosine": float(cosine),
            "teacher_target_dtype": str(teacher_tokens.dtype),
            "teacher_target_mean_abs_max": float(target_mean),
            "teacher_target_population_variance_mean": float(target_var),
            "student_tokens_dtype": str(student_tokens.dtype),
            "analysis_dtype": "torch.float32",
        }


def _compare_epoch20(student: torch.nn.Module, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected = checkpoint.get("student")
    if not isinstance(expected, dict):
        raise RuntimeError("Saved B0-LN artifact has no student state")
    actual = {name: value.detach().cpu() for name, value in student.state_dict().items()}
    if list(expected) != list(actual):
        return {"exact": False, "reason": "state_dict_keys_differ"}
    max_abs = 0.0
    max_rel = 0.0
    different = 0
    for name in expected:
        left = expected[name].detach().cpu().float()
        right = actual[name].float()
        diff = (left - right).abs()
        max_abs = max(max_abs, float(diff.max()))
        max_rel = max(max_rel, float((diff / left.abs().clamp_min(1e-12)).max()))
        different += int(torch.count_nonzero(diff).item())
    expected_hash = tensor_state_hash(expected)
    actual_hash = tensor_state_hash(actual)
    return {
        "exact": expected_hash == actual_hash,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "expected_hash": expected_hash,
        "replayed_hash": actual_hash,
        "max_abs_difference": max_abs,
        "max_relative_difference": max_rel,
        "different_tensor_values": different,
    }


def _replay_and_audit(reference: dict[str, Any], output: Path, audit_epochs: tuple[int, ...]) -> dict[str, Any]:
    _assert_config(reference)
    if not torch.cuda.is_available():
        raise RuntimeError("B0-LN gradient audit requires CUDA")
    seed = int(reference["seed"])
    train_config = reference["train"]
    device = torch.device("cuda")
    seed_everything(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    endpoints = _endpoints(reference["endpoints_json"])

    # Keep initialization and loader construction in the same order as the
    # original B0-LN run.  This replay is read-only with respect to its source.
    loader, _ = build_ssl_loader(
        reference["data_root"],
        batch_size=int(train_config["batch_size"]),
        num_workers=int(train_config["num_workers"]),
        cache_in_ram=bool(train_config["cache_in_ram"]),
    )
    if len(loader.dataset) != EXPECTED_SSL_SAMPLES or loader.dataset.include_label:
        raise RuntimeError("Replay loader is not the 7,901-image unlabeled SSL loader")
    student, teacher, predictor = build_b0_models(reference, device)
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("Replay teacher unexpectedly has trainable parameters")
    parameters = list(student.parameters()) + list(predictor.parameters())
    optimizer = _make_optimizer(parameters, float(train_config["learning_rate"]), float(train_config["weight_decay"]))
    effective_batch = int(train_config["effective_batch_size"])
    batch_size = int(train_config["batch_size"])
    accumulation_steps = effective_batch // batch_size
    steps_per_epoch = math.ceil(len(loader) / accumulation_steps)
    total_steps = int(train_config["epochs"]) * steps_per_epoch
    global_step = 0

    # This is the only RNG effect of the validation monitor relevant to the
    # CUDA transition stream.  No downstream metadata/loader is constructed.
    _emulate_monitor_rng_reset(seed, device)

    reference_metrics: dict[int, dict[str, float]] = {}
    metrics_path = Path(reference["output_dir"]) / "pretrain_metrics.csv"
    if metrics_path.is_file():
        with metrics_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                epoch = int(float(row["epoch"]))
                reference_metrics[epoch] = {key: float(row[key]) for key in ("total", "prediction", "regularizer")}

    images, diag_actions, diag_source, diag_target, diag_delta, batch_records = _fixed_diagnostic_batch(
        reference,
        device,
        batch_size=8,
        seed=int(reference["seed"]) + 7301,
    )
    # _fixed_diagnostic_batch uses a private generator and therefore has no
    # effect on the replay stream.  It is loaded before epoch 1 for audit use.
    audit_rows: list[dict[str, Any]] = []
    replay_metrics: list[dict[str, Any]] = []
    state_match: dict[str, Any] | None = None
    started = time.perf_counter()
    for epoch in range(1, max(audit_epochs) + 1):
        student.train()
        predictor.train()
        teacher.eval()
        optimizer.zero_grad(set_to_none=True)
        totals = {name: 0.0 for name in ("total", "prediction", "regularizer")}
        seen = 0
        for batch_index, uint8_images in enumerate(loader):
            batch_images = uint8_images.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
            # The original transition sampler is used exactly; its RNG stream
            # is kept aligned by the monitor reset above and at each monitor.
            actions, source, target, delta = sample_adjacent_transitions(batch_images.shape[0], device)
            source_images = degrade_batch(batch_images, actions, source, endpoints)
            target_images = degrade_batch(batch_images, actions, target, endpoints)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(train_config["bf16"])):
                student_tokens = student(source_images)
                prediction = predictor(student_tokens, source, actions, delta)
                teacher_tokens = layer_norm_teacher_target(teacher, target_images)
            losses = b0_loss(
                prediction,
                teacher_tokens.detach(),
                student_tokens,
                lambda_reg=float(train_config["lambda_reg"]),
                covariance_weight=float(train_config["covariance_weight"]),
                variance_target=float(train_config["variance_target"]),
            )
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError(f"Non-finite replay loss at epoch {epoch}, batch {batch_index}")
            (losses["total"] / accumulation_steps).backward()
            is_update = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader)
            if is_update:
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float(train_config["gradient_clip_norm"]))
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("Non-finite replay gradient norm")
                for group in optimizer.param_groups:
                    group["lr"] = _learning_rate(train_config, global_step, total_steps)
                    group["weight_decay"] = _weight_decay(train_config, global_step, total_steps)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                from pannuke_ssl.models import update_ema

                update_ema(student, teacher, _cosine(float(train_config["ema_start"]), 1.0, global_step, total_steps))
                global_step += 1
            count = int(batch_images.shape[0])
            seen += count
            for name in totals:
                totals[name] += float(losses[name].detach()) * count
        if seen != EXPECTED_SSL_SAMPLES:
            raise RuntimeError(f"Replay expected {EXPECTED_SSL_SAMPLES} samples, saw {seen}")
        replay_row = {"epoch": epoch, **{name: value / seen for name, value in totals.items()}}
        replay_metrics.append(replay_row)
        if epoch in audit_epochs:
            audit_rows.append(
                _gradient_audit(
                    student,
                    teacher,
                    predictor,
                    images,
                    actions=diag_actions,
                    source=diag_source,
                    target=diag_target,
                    delta=diag_delta,
                    endpoints=endpoints,
                    train_config=train_config,
                    epoch=epoch,
                )
            )
        if epoch == 20:
            state_match = _compare_epoch20(student, Path(reference["output_dir"]) / "checkpoints/best.pt")
            if not state_match.get("exact", False):
                break
        # The original B0-LN monitor runs at epoch 0,10,20,... and its only
        # relevant CUDA RNG effect is the seeded Linear classifier init.
        if epoch % 10 == 0:
            _emulate_monitor_rng_reset(seed, device)

    replay_comparison = []
    for row in replay_metrics:
        epoch = int(row["epoch"])
        if epoch not in reference_metrics:
            continue
        replay_comparison.append(
            {
                "epoch": epoch,
                "total_abs_diff": abs(row["total"] - reference_metrics[epoch]["total"]),
                "prediction_abs_diff": abs(row["prediction"] - reference_metrics[epoch]["prediction"]),
                "regularizer_abs_diff": abs(row["regularizer"] - reference_metrics[epoch]["regularizer"]),
            }
        )
    return {
        "status": "passed" if state_match and state_match.get("exact") else "blocked_replay_mismatch",
        "replay_seconds": time.perf_counter() - started,
        "audit_epochs_requested": list(audit_epochs),
        "audit_epochs_measured": [int(row["epoch"]) for row in audit_rows],
        "epoch20_state_match": state_match,
        "replay_metrics": replay_metrics,
        "replay_vs_saved_training_metrics": replay_comparison,
        "gradient_rows": audit_rows,
        "diagnostic_batch": batch_records,
        "diagnostic_batch_size": len(batch_records),
        "diagnostic_transition_seed": int(reference["seed"]) + 7301,
        "ssl_source_count": EXPECTED_SSL_SAMPLES,
        "balanced_downstream_loader_constructed": False,
        "test_loader_constructed": False,
        "test_marker_touched": False,
        "test_predictions_touched": False,
    }


def _category(value: float) -> str:
    if value < 0.25:
        return "weak"
    if value < 0.75:
        return "moderate"
    if value <= 1.5:
        return "comparable"
    return "regularizer-dominant"


def _cosine_category(value: float) -> str:
    if value > 0.3:
        return "meaningfully aligned"
    if value < -0.3:
        return "meaningfully conflicting"
    return "mostly orthogonal / weak relation"


def _render_report(result: dict[str, Any], path: Path, protected_before: dict[str, Any], protected_after: dict[str, Any], reference: dict[str, Any]) -> None:
    lines = [
        "# B0-LN encoder-gradient audit",
        "",
        "Diagnostic-only, single-batch, FP32 gradient measurement for the existing B0-LN objective. No parameters, optimizer state, or EMA state were updated during measurement.",
        "",
        "## Exact scope",
        "",
        "- The prediction gradient is `∇θ Lpred`; the regularizer gradient is `∇θ (lambda_reg * Lreg)` for student encoder parameters only.",
        "- The B0-LN teacher target is the existing stateless FP32 final-dimension LayerNorm; student tokens are not normalized.",
        "- The replay used the original 300-epoch schedule controls through epoch 50, then verified the replayed epoch-20 student state against the saved B0-LN epoch-20 checkpoint before accepting missing epoch states.",
        "- The fixed diagnostic batch has 8 unlabeled SSL images and a private transition seed; the balanced downstream/test loader was never constructed.",
        "",
        "## Results",
        "",
        "| Epoch | Pred loss | Weighted reg loss | ||g_pred|| | ||g_reg|| | Reg/Pred grad ratio | Cosine | Ratio class | Cosine class |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in result.get("gradient_rows", []):
        ratio = float(row["reg_pred_grad_ratio"])
        cosine = float(row["gradient_cosine"])
        lines.append(
            f"| {int(row['epoch'])} | {row['prediction_loss']:.6g} | {row['weighted_regularizer_loss']:.6g} | "
            f"{row['g_pred_norm']:.6g} | {row['g_reg_norm']:.6g} | {ratio:.6g} | {cosine:.6g} | {_category(ratio)} | {_cosine_category(cosine)} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
    ]
    if result.get("status") != "passed":
        lines += [
            "The diagnostic is blocked: the isolated replay did not reproduce the saved epoch-20 student state exactly. No unverified missing-epoch gradient rows are reported.",
            f"Replay status: `{result.get('status')}`.",
        ]
    else:
        rows = result.get("gradient_rows", [])
        ratios = [float(row["reg_pred_grad_ratio"]) for row in rows]
        cosines = [float(row["gradient_cosine"]) for row in rows]
        dominant = [int(row["epoch"]) for row in rows if _category(float(row["reg_pred_grad_ratio"])) == "regularizer-dominant"]
        conflicting = [int(row["epoch"]) for row in rows if _cosine_category(float(row["gradient_cosine"])) == "meaningfully conflicting"]
        if dominant:
            answer1 = f"VICReg is gradient-dominant at epochs {dominant}; elsewhere its student-encoder gradient is at most comparable/moderate by the requested descriptive thresholds."
        else:
            answer1 = "VICReg does not dominate the student-encoder gradient at any audited epoch by the requested >1.5 ratio threshold."
        if conflicting:
            answer2 = f"The two gradients are meaningfully conflicting at epochs {conflicting}; cosine values should be read as a fixed-batch diagnostic, not a statistical test."
        else:
            answer2 = "No audited epoch has a meaningfully conflicting cosine (< -0.3); the gradients are otherwise aligned or weakly related as shown in the table."
        answer3 = "A lambda_reg ablation is justified only if the dominant/conflicting pattern is persistent and coincides with poor representation dynamics; this audit alone does not authorize running it."
        lines += [answer1, answer2, answer3]
        if ratios:
            lines += [
                "",
                f"Across the audited fixed batch, the ratio range is `{min(ratios):.6g}`–`{max(ratios):.6g}` and cosine range is `{min(cosines):.6g}`–`{max(cosines):.6g}`.",
            ]
    lines += [
        "",
        "## Integrity and no-test proof",
        "",
        f"- Epoch-20 replay verification: `{json.dumps(result.get('epoch20_state_match'), sort_keys=True)}`.",
        f"- Protected B0/B1/I-JEPA reference entries before/after: `{len(protected_before['protected_paths'])}` / `{len(protected_after['protected_paths'])}`; verification passed.",
        f"- Saved B0-LN checkpoint SHA-256 before/after: `{sha256_path(Path(reference['output_dir']) / 'checkpoints/best.pt')}` (unchanged).",
        "- The script constructed only the unlabeled 7,901-image SSL loader and the fixed 8-image diagnostic batch. It did not read balanced downstream metadata, construct a test loader, touch a test marker, or read test predictions.",
        "",
        "The result is descriptive and single-seed. It does not establish statistical significance or by itself prove that changing `lambda_reg` will improve B0-LN.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an isolated B0-LN encoder-gradient audit")
    parser.add_argument("--config", default="configs/b0_ln_gradient_audit.yaml")
    args = parser.parse_args()
    audit_config = load_yaml(args.config)
    reference_config_path = Path(audit_config["reference_config"])
    reference = load_yaml(reference_config_path)
    output = Path(audit_config["output_dir"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit output: {output}")
    output.mkdir(parents=True)
    audit_epochs = tuple(int(epoch) for epoch in audit_config.get("audit_epochs", AUDIT_EPOCHS))
    if audit_epochs != AUDIT_EPOCHS:
        raise ValueError(f"Audit epochs are locked to {AUDIT_EPOCHS}")
    if int(audit_config.get("diagnostic_batch_size", 8)) != 8:
        raise ValueError("Diagnostic batch size is locked to 8")

    reference_output = Path(reference["output_dir"])
    protected_before = verify_protected_manifest(reference_output)
    checkpoint = reference_output / "checkpoints/best.pt"
    checkpoint_before = sha256_path(checkpoint)
    implementation_files = [Path(__file__).resolve(), reference_config_path.resolve(), Path(args.config).resolve()]
    implementation_manifest = {
        "schema_version": 1,
        "purpose": "isolated B0-LN diagnostic audit implementation",
        "files": [{"path": str(path), "sha256": sha256_path(path), "size": path.stat().st_size} for path in implementation_files],
    }
    atomic_json_dump(implementation_manifest, output / "implementation_manifest.json")
    atomic_json_dump(
        {
            "schema_version": 1,
            "reference_output": str(reference_output),
            "protected_manifest_entries": len(protected_before["protected_paths"]),
            "protected_manifest_sha256": sha256_path(reference_output / "protected_manifest.json"),
            "b0_ln_checkpoint_sha256_before": checkpoint_before,
            "test_loader_constructed": False,
            "test_marker_touched": False,
            "test_predictions_touched": False,
        },
        output / "integrity_before.json",
    )
    try:
        result = _replay_and_audit(reference, output, audit_epochs)
    except Exception as error:
        result = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "audit_epochs_requested": list(audit_epochs),
            "gradient_rows": [],
            "balanced_downstream_loader_constructed": False,
            "test_loader_constructed": False,
            "test_marker_touched": False,
            "test_predictions_touched": False,
        }
    protected_after = verify_protected_manifest(reference_output)
    checkpoint_after = sha256_path(checkpoint)
    implementation_after = [{"path": str(path), "sha256": sha256_path(path), "size": path.stat().st_size} for path in implementation_files]
    if implementation_after != implementation_manifest["files"]:
        raise RuntimeError("Audit implementation/config changed during execution")
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("Existing B0-LN checkpoint changed during audit")
    result["protected_references_unchanged"] = True
    result["b0_ln_checkpoint_sha256_before"] = checkpoint_before
    result["b0_ln_checkpoint_sha256_after"] = checkpoint_after
    result["protected_manifest_entries"] = len(protected_after["protected_paths"])
    result["completed_at_unix"] = time.time()
    atomic_json_dump(result, output / "gradient_audit.json")
    rows = result.get("gradient_rows", [])
    if rows:
        _write_csv(rows, output / "gradient_audit.csv")
    else:
        _write_csv([{"status": result.get("status", "unknown")}], output / "gradient_audit.csv")
    _render_report(result, output / "REPORT.md", protected_before, protected_after, reference)
    atomic_json_dump(
        {
            "protected_manifest_entries_before": len(protected_before["protected_paths"]),
            "protected_manifest_entries_after": len(protected_after["protected_paths"]),
            "protected_references_unchanged": True,
            "b0_ln_checkpoint_sha256_before": checkpoint_before,
            "b0_ln_checkpoint_sha256_after": checkpoint_after,
            "test_loader_constructed": False,
            "test_marker_touched": False,
            "test_predictions_touched": False,
        },
        output / "integrity_after.json",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("status") != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
