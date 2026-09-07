#!/usr/bin/env python
"""Fast GPU smoke test for the B0-Delta objective before the full run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from pannuke_ssl.b0_delta_training import teacher_delta_target, validate_config
from pannuke_ssl.config import load_yaml
from pannuke_ssl.data import PanNukeImageDataset
from pannuke_ssl.degradations import degrade_batch, sample_adjacent_transitions
from pannuke_ssl.losses import b0_loss
from pannuke_ssl.models import update_ema
from pannuke_ssl.parquet import build_source_index, preload_images
from pannuke_ssl.training import _endpoints, build_b0_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test B0-Delta on real PanNuke images")
    parser.add_argument("--config", default="configs/b0_delta_duration_pilot.yaml")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    config = load_yaml(args.config)
    validate_config(config)
    if args.batch_size < 2:
        raise ValueError("Smoke test batch size must be at least 2 for the VICReg control")

    device = torch.device("cuda")
    source_index = build_source_index(Path(config["data_root"]))
    selected = [source_index[key] for key in sorted(source_index)[: args.batch_size]]
    rows = [
        {
            "fold": row.fold,
            "sample_index": row.sample_index,
            "tissue_label": row.tissue_label,
            "class_id": row.class_id,
        }
        for row in selected
    ]
    cache = preload_images(rows, source_index)
    dataset = PanNukeImageDataset(rows, source_index, cache, include_label=True)
    images = torch.stack([dataset[i][0] for i in range(args.batch_size)]).to(
        device=device, dtype=torch.float32
    ).div_(255.0)

    student, teacher, predictor = build_b0_models(config, device)
    trainable = list(student.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=float(config["train"]["learning_rate"]))
    endpoints = _endpoints(config["endpoints_json"])

    actions, source_severity, target_severity, delta = sample_adjacent_transitions(
        args.batch_size, device
    )
    expected = torch.full_like(source_severity, 0.25)
    assert torch.equal(source_severity - target_severity, expected)
    assert torch.equal(delta, -expected)

    source_images = degrade_batch(images, actions, source_severity, endpoints)
    target_images = degrade_batch(images, actions, target_severity, endpoints)

    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        student_tokens = student(source_images)
        prediction = predictor(student_tokens, source_severity, actions, delta)
        with torch.no_grad():
            teacher_source = teacher(source_images)
            teacher_target = teacher(target_images)

    target_delta = teacher_delta_target(teacher_source, teacher_target)
    if target_delta.dtype != torch.float32:
        raise RuntimeError("Teacher residual target must be constructed in FP32")
    if target_delta.shape != prediction.shape:
        raise RuntimeError("B0-Delta prediction and target shapes do not match")

    losses = b0_loss(
        prediction,
        target_delta,
        student_tokens,
        lambda_reg=float(config["train"]["lambda_reg"]),
        covariance_weight=float(config["train"]["covariance_weight"]),
        variance_target=float(config["train"]["variance_target"]),
    )
    if not all(torch.isfinite(value) for value in losses.values()):
        raise FloatingPointError("B0-Delta smoke loss is non-finite")

    optimizer.zero_grad(set_to_none=True)
    losses["total"].backward()
    student_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in student.parameters()
        if parameter.grad is not None
    )
    predictor_grad = sum(
        float(parameter.grad.abs().sum())
        for parameter in predictor.parameters()
        if parameter.grad is not None
    )
    if student_grad <= 0 or predictor_grad <= 0:
        raise RuntimeError("Student and predictor must both receive gradients")
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("EMA teacher must remain gradient-free")

    optimizer.step()
    update_ema(student, teacher, float(config["train"]["ema_start"]))

    with torch.no_grad():
        zero_loss = F.smooth_l1_loss(torch.zeros_like(target_delta), target_delta)
        cosine = F.cosine_similarity(
            prediction.float().flatten(1), target_delta.flatten(1), dim=1
        ).mean()
        target_rms = target_delta.square().mean().sqrt()
        residual_skill = 1.0 - float(losses["prediction"].detach()) / max(
            float(zero_loss), 1e-12
        )

    summary = {
        "passed": True,
        "batch_size": args.batch_size,
        "image_shape": list(images.shape),
        "student_token_shape": list(student_tokens.shape),
        "prediction_shape": list(prediction.shape),
        "target_delta_shape": list(target_delta.shape),
        "target_delta_dtype": str(target_delta.dtype),
        "prediction_loss": float(losses["prediction"].detach()),
        "zero_prediction_loss": float(zero_loss),
        "residual_skill": residual_skill,
        "residual_cosine": float(cosine),
        "target_delta_rms": float(target_rms),
        "student_gradient_sum": student_grad,
        "predictor_gradient_sum": predictor_grad,
        "teacher_has_gradients": False,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "objective": "EMA_teacher(less-degraded) - EMA_teacher(more-degraded)",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
