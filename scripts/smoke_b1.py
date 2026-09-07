#!/usr/bin/env python
"""Comprehensive real-image GPU smoke for B1; it never builds a test loader."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from transformers import CLIPVisionModel

from pannuke_ssl.b1 import build_b1_models, residual_endpoint
from pannuke_ssl.b1_losses import b1_loss
from pannuke_ssl.b1_training import _endpoints, validate_config, verify_protected_manifest
from pannuke_ssl.config import load_yaml
from pannuke_ssl.degradations import degrade_batch
from pannuke_ssl.models import update_ema
from pannuke_ssl.parquet import build_source_index, preload_images
from pannuke_ssl.training import _make_optimizer
from pannuke_ssl.utils import atomic_json_dump, seed_everything


def _smoke_transitions(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exercise both actions and every permitted adjacent transition per step."""
    actions = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], device=device)
    source = torch.tensor([1.0, 1.0, 0.75, 0.75, 0.50, 0.50, 0.25, 0.25], device=device)
    target = source - 0.25
    delta = target - source
    return actions, source, target, delta


def _parameter_sha(module: torch.nn.Module) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B1's comprehensive GPU smoke")
    parser.add_argument("--config", default="configs/b1_duration_pilot.yaml")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("B1 smoke requires CUDA")
    config = load_yaml(args.config)
    validate_config(config)
    output = Path(config["output_dir"])
    if (output / "pretrain_metrics.csv").exists():
        raise RuntimeError("Refusing to smoke after a B1 duration run has started")
    if (output / "smoke_test.json").exists():
        raise FileExistsError("Refusing to overwrite an existing B1 smoke result")
    verify_protected_manifest(output)
    started = time.perf_counter()
    seed_everything(int(config["seed"]))
    torch.set_num_threads(4)
    device = torch.device("cuda")
    source_index = build_source_index(Path(config["data_root"]))
    if len(source_index) != 7901:
        raise RuntimeError("B1 smoke requires exactly 7,901 unlabeled source records")
    # These are unlabeled SSL records only; no downstream metadata, validation,
    # or test loader is opened by this smoke.
    records = [source_index[key] for key in sorted(source_index)[:8]]
    rows = [
        {"fold": record.fold, "sample_index": record.sample_index, "tissue_label": record.tissue_label, "class_id": record.class_id}
        for record in records
    ]
    cache = preload_images(rows, source_index)
    images = torch.stack(
        [torch.from_numpy(np.array(cache[int(row["fold"]), int(row["sample_index"])], copy=True)).permute(2, 0, 1) for row in rows]
    ).to(device=device, dtype=torch.float32).div_(255.0)
    if tuple(images.shape) != (8, 3, 256, 256):
        raise RuntimeError(f"Unexpected B1 smoke image shape: {tuple(images.shape)}")
    # The architecture config may be read, but pretrained weights may not.
    with patch.object(CLIPVisionModel, "from_pretrained", side_effect=AssertionError("Pretrained PLIP weights are forbidden")):
        student, teacher, predictor = build_b1_models(config, device)
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("B1 teacher has trainable parameters")
    if any(not torch.equal(a, b) for a, b in zip(student.parameters(), teacher.parameters(), strict=True)):
        raise RuntimeError("B1 teacher must start as an EMA copy of the student")
    optimizer = _make_optimizer(list(student.parameters()) + list(predictor.parameters()), 1e-4, 0.04)
    endpoints = _endpoints(config["endpoints_json"])
    actions, source, target, delta = _smoke_transitions(device)
    if not torch.equal(source - target, torch.full_like(source, 0.25)) or not torch.equal(delta, torch.full_like(delta, -0.25)):
        raise RuntimeError("B1 smoke transition setup is not adjacent reverse conditioning")
    torch.cuda.reset_peak_memory_stats()
    losses, condition_gradients, endpoint_errors = [], {}, []
    initial_student = _parameter_sha(student)
    for step in range(3):
        student.train()
        predictor.train()
        optimizer.zero_grad(set_to_none=True)
        source_images = degrade_batch(images, actions, source, endpoints)
        target_images = degrade_batch(images, actions, target, endpoints)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            source_tokens = student(source_images)
            velocity = predictor(source_tokens, source, actions, delta)
            prediction = residual_endpoint(source_tokens, velocity, delta)
            with torch.no_grad():
                teacher_tokens = teacher(target_images)
        if tuple(source_tokens.shape) != (8, 64, 768) or velocity.shape != source_tokens.shape or prediction.shape != source_tokens.shape:
            raise RuntimeError("B1 smoke did not traverse the real 64x768 token path")
        if not torch.equal(prediction, source_tokens + delta[:, None, None].to(source_tokens.dtype) * velocity):
            raise RuntimeError("B1 residual endpoint algebra changed")
        constructed_velocity = (teacher_tokens - source_tokens.detach()) / delta[:, None, None].to(teacher_tokens.dtype)
        reconstructed = residual_endpoint(source_tokens.detach(), constructed_velocity, delta)
        endpoint_error = (reconstructed.float() - teacher_tokens.float()).abs().max()
        if endpoint_error > 3e-2:
            raise RuntimeError("B1 velocity reconstruction has the wrong endpoint sign")
        endpoint_errors.append(float(endpoint_error))
        values = b1_loss(
            prediction,
            teacher_tokens.detach(),
            source_tokens,
            lambda_reg=0.10,
            covariance_weight=0.04,
            variance_target=1.0,
        )
        if not all(torch.isfinite(value) for value in values.values()):
            raise FloatingPointError("B1 smoke has non-finite loss terms")
        values["total"].backward()
        if any(parameter.grad is not None for parameter in teacher.parameters()):
            raise RuntimeError("B1 teacher received gradients")
        for prefix in ("severity_conditioner", "action_embedding", "delta_conditioner"):
            gradients = [parameter.grad for name, parameter in predictor.named_parameters() if name.startswith(prefix)]
            value = sum(float(gradient.abs().sum()) for gradient in gradients if gradient is not None)
            if not gradients or value <= 0.0:
                raise RuntimeError(f"B1 {prefix} did not receive a gradient")
            condition_gradients[prefix] = value
        patch_grad = student.model.vision_model.embeddings.patch_embedding.weight.grad
        if patch_grad is None or float(patch_grad.abs().sum()) <= 0.0:
            raise RuntimeError("B1 student encoder did not receive a patch-embedding gradient")
        if not all(torch.isfinite(parameter.grad).all() for parameter in list(student.parameters()) + list(predictor.parameters()) if parameter.grad is not None):
            raise FloatingPointError("B1 smoke has non-finite gradients")
        before_teacher = next(teacher.parameters()).detach().clone()
        optimizer.step()
        update_ema(student, teacher, 0.996)
        expected_teacher = before_teacher * 0.996 + next(student.parameters()).detach() * 0.004
        if not torch.allclose(next(teacher.parameters()), expected_teacher, atol=1e-6, rtol=1e-5):
            raise RuntimeError("B1 EMA update is inconsistent")
        losses.append(float(values["total"].detach()))
    if initial_student == _parameter_sha(student):
        raise RuntimeError("B1 smoke did not update the student")
    if (output / "test_started.json").exists():
        raise RuntimeError("B1 smoke must not run after any one-time test marker")
    verify_protected_manifest(output)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    summary = {
        "passed": True,
        "gpu_steps": 3,
        "bf16": True,
        "image_shape": list(images.shape),
        "token_shape": [8, 64, 768],
        "velocity_shape": [8, 64, 768],
        "actions": ["defocus", "resolution"],
        "adjacent_transitions": ["1.00_to_0.75", "0.75_to_0.50", "0.50_to_0.25", "0.25_to_0.00"],
        "delta": -0.25,
        "losses": losses,
        "condition_gradient_sums": condition_gradients,
        "maximum_constructed_endpoint_error": max(endpoint_errors),
        "teacher_stopgradient_and_ema_verified": True,
        "pretrained_weight_loading_forbidden": True,
        "ssl_source_records": 7901,
        "smoke_decoded_splits": ["unlabeled_ssl_source"],
        "validation_images_decoded_by_smoke": False,
        "test_images_decoded_by_smoke": False,
        "images_persisted": False,
        "protected_b0_ijepa_unchanged": True,
        "seconds": seconds,
        "samples_per_second": 24 / seconds,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    atomic_json_dump(summary, output / "smoke_test.json")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
