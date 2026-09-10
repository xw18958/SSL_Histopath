#!/usr/bin/env python
"""Comprehensive real-image B0-LN GPU smoke; it never builds a probe/test loader."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPVisionModel

from pannuke_ssl.b0_ln_training import (
    _endpoints,
    layer_norm_teacher_target,
    validate_config,
    verify_implementation_manifest,
    verify_protected_manifest,
)
from pannuke_ssl.config import load_yaml
from pannuke_ssl.degradations import degrade_batch
from pannuke_ssl.losses import b0_loss
from pannuke_ssl.models import update_ema
from pannuke_ssl.parquet import build_source_index, preload_images
from pannuke_ssl.training import _make_optimizer, build_b0_models
from pannuke_ssl.utils import atomic_json_dump, seed_everything


def _smoke_transitions(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    actions = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], device=device)
    source = torch.tensor([1.0, 1.0, 0.75, 0.75, 0.50, 0.50, 0.25, 0.25], device=device)
    target = source - 0.25
    return actions, source, target, target - source


def _parameter_sha(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B0-LN's comprehensive GPU smoke")
    parser.add_argument("--config", default="configs/b0_ln_duration_pilot.yaml")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("B0-LN smoke requires CUDA")
    config = load_yaml(args.config)
    validate_config(config)
    output = Path(config["output_dir"])
    if (output / "pretrain_metrics.csv").exists():
        raise RuntimeError("Refusing to smoke after B0-LN duration training has started")
    if (output / "smoke_test.json").exists():
        raise FileExistsError("Refusing to overwrite an existing B0-LN smoke result")
    if (output / "test_started.json").exists():
        raise RuntimeError("B0-LN smoke must not run after a test marker")
    verify_protected_manifest(output)
    verify_implementation_manifest(output)
    started = time.perf_counter()
    seed_everything(int(config["seed"]))
    torch.set_num_threads(4)
    device = torch.device("cuda")
    source_index = build_source_index(Path(config["data_root"]))
    if len(source_index) != 7901:
        raise RuntimeError("B0-LN smoke requires exactly 7,901 unlabeled SSL source records")
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
        raise RuntimeError(f"Unexpected B0-LN smoke image shape: {tuple(images.shape)}")
    with patch.object(CLIPVisionModel, "from_pretrained", side_effect=AssertionError("Pretrained PLIP weights are forbidden")):
        student, teacher, predictor = build_b0_models(config, device)
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("B0-LN teacher has trainable parameters")
    if any(not torch.equal(s, t) for s, t in zip(student.parameters(), teacher.parameters(), strict=True)):
        raise RuntimeError("B0-LN teacher must begin as the EMA copy of the student")
    optimizer = _make_optimizer(list(student.parameters()) + list(predictor.parameters()), 1e-4, 0.04)
    endpoints = _endpoints(config["endpoints_json"])
    actions, source, target, delta = _smoke_transitions(device)
    if not torch.equal(source - target, torch.full_like(source, 0.25)) or not torch.equal(delta, torch.full_like(delta, -0.25)):
        raise RuntimeError("B0-LN smoke transition setup is not adjacent reverse conditioning")
    observed_predictor_inputs: list[torch.Tensor] = []
    hook = predictor.input_projection.register_forward_pre_hook(lambda _module, values: observed_predictor_inputs.append(values[0].detach().clone()))
    try:
        torch.cuda.reset_peak_memory_stats()
        total_losses: list[float] = []
        target_mean_abs_maxes: list[float] = []
        target_variance_means: list[float] = []
        initial_student = _parameter_sha(student)
        for _step in range(3):
            student.train()
            predictor.train()
            optimizer.zero_grad(set_to_none=True)
            source_images = degrade_batch(images, actions, source, endpoints)
            target_images = degrade_batch(images, actions, target, endpoints)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                student_tokens = student(source_images)
                prediction = predictor(student_tokens, source, actions, delta)
                teacher_tokens = layer_norm_teacher_target(teacher, target_images)
                with torch.no_grad():
                    raw_teacher_tokens = teacher(target_images)
            expected_target = F.layer_norm(raw_teacher_tokens.float(), (raw_teacher_tokens.shape[-1],))
            if tuple(teacher_tokens.shape) != (8, 64, 768) or prediction.shape != student_tokens.shape:
                raise RuntimeError("B0-LN smoke did not traverse the matched [B,64,768] token path")
            if teacher_tokens.dtype != torch.float32 or not torch.equal(teacher_tokens, expected_target):
                raise RuntimeError("B0-LN target is not the exact I-JEPA-style stateless FP32 LayerNorm target")
            if not observed_predictor_inputs or not torch.equal(observed_predictor_inputs[-1], student_tokens.detach()):
                raise RuntimeError("B0-LN modified the student representation before the predictor")
            target_mean = teacher_tokens.mean(dim=-1)
            target_variance = teacher_tokens.var(dim=-1, unbiased=False)
            if float(target_mean.abs().max()) > 3e-4 or abs(float(target_variance.mean()) - 1.0) > 5e-3:
                raise RuntimeError("B0-LN target statistics are inconsistent with LayerNorm")
            target_mean_abs_maxes.append(float(target_mean.abs().max()))
            target_variance_means.append(float(target_variance.mean()))
            values = b0_loss(
                prediction,
                teacher_tokens.detach(),
                student_tokens,
                lambda_reg=0.10,
                covariance_weight=0.04,
                variance_target=1.0,
            )
            if not all(torch.isfinite(value) for value in values.values()):
                raise FloatingPointError("B0-LN smoke has non-finite loss terms")
            values["total"].backward()
            if any(parameter.grad is not None for parameter in teacher.parameters()):
                raise RuntimeError("B0-LN teacher received gradients")
            patch_grad = student.model.vision_model.embeddings.patch_embedding.weight.grad
            if patch_grad is None or float(patch_grad.abs().sum()) <= 0.0:
                raise RuntimeError("B0-LN student encoder did not receive a patch-embedding gradient")
            predictor_grads = [parameter.grad for parameter in predictor.parameters()]
            if not any(gradient is not None and float(gradient.abs().sum()) > 0.0 for gradient in predictor_grads):
                raise RuntimeError("B0-LN predictor did not receive gradients")
            if not all(torch.isfinite(parameter.grad).all() for parameter in list(student.parameters()) + list(predictor.parameters()) if parameter.grad is not None):
                raise FloatingPointError("B0-LN smoke has non-finite gradients")
            before_teacher = next(teacher.parameters()).detach().clone()
            optimizer.step()
            update_ema(student, teacher, 0.996)
            expected_teacher = before_teacher * 0.996 + next(student.parameters()).detach() * 0.004
            if not torch.allclose(next(teacher.parameters()), expected_teacher, atol=1e-6, rtol=1e-5):
                raise RuntimeError("B0-LN EMA update differs from B0")
            total_losses.append(float(values["total"].detach()))
        if initial_student == _parameter_sha(student):
            raise RuntimeError("B0-LN smoke did not update the student")
    finally:
        hook.remove()
    verify_protected_manifest(output)
    verify_implementation_manifest(output)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    summary = {
        "passed": True,
        "gpu_steps": 3,
        "bf16": True,
        "image_shape": list(images.shape),
        "teacher_target_shape": [8, 64, 768],
        "teacher_target_dtype": "float32",
        "teacher_target_layer_norm": "F.layer_norm(raw_teacher_tokens.float(), (raw_teacher_tokens.shape[-1],))",
        "teacher_target_mean_abs_max": max(target_mean_abs_maxes),
        "teacher_target_variance_mean": sum(target_variance_means) / len(target_variance_means),
        "student_input_to_predictor_unchanged": True,
        "teacher_stopgradient_and_ema_verified": True,
        "student_and_predictor_gradients_verified": True,
        "actions": ["defocus", "resolution"],
        "adjacent_transitions": ["1.00_to_0.75", "0.75_to_0.50", "0.50_to_0.25", "0.25_to_0.00"],
        "total_losses": total_losses,
        "ssl_source_records": 7901,
        "smoke_decoded_splits": ["unlabeled_ssl_source"],
        "validation_images_decoded_by_smoke": False,
        "test_images_decoded_by_smoke": False,
        "images_persisted": False,
        "protected_references_unchanged": True,
        "implementation_unchanged": True,
        "seconds": seconds,
        "samples_per_second": 24 / seconds,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    atomic_json_dump(summary, output / "smoke_test.json")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
