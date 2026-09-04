#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch

from pannuke_ssl.calibration import _vif, fit_piecewise_hinge
from pannuke_ssl.config import load_yaml
from pannuke_ssl.degradations import DegradationEndpoints, degrade_batch, sample_adjacent_transitions
from pannuke_ssl.losses import b0_loss
from pannuke_ssl.models import B0Predictor, FreshPLIPVisionEncoder, make_teacher, update_ema
from pannuke_ssl.data import PanNukeImageDataset
from pannuke_ssl.parquet import build_source_index, preload_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast GPU smoke and batch-throughput test for B0")
    parser.add_argument("--config", default="configs/b0.yaml")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config = load_yaml(args.config)
    device = torch.device("cuda")
    source_index = build_source_index(Path(config["data_root"]))
    selected = [source_index[key] for key in sorted(source_index)[: args.batch_size]]
    rows = [
        {"fold": row.fold, "sample_index": row.sample_index, "tissue_label": row.tissue_label, "class_id": row.class_id}
        for row in selected
    ]
    cache = preload_images(rows, source_index)
    dataset = PanNukeImageDataset(rows, source_index, cache, include_label=True)
    images = torch.stack([dataset[index][0] for index in range(args.batch_size)]).to(device=device, dtype=torch.float32).div_(255)
    student = FreshPLIPVisionEncoder(config["plip_config_dir"]).to(device)
    teacher = make_teacher(student).to(device)
    predictor = B0Predictor().to(device)
    optimizer = torch.optim.AdamW(list(student.parameters()) + list(predictor.parameters()), lr=1e-4)
    endpoints = DegradationEndpoints(defocus_radius=8.0, resolution_factor=8.0)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    actions, source_severity, target_severity, delta = sample_adjacent_transitions(args.batch_size, device)
    source_images = degrade_batch(images, actions, source_severity, endpoints)
    target_images = degrade_batch(images, actions, target_severity, endpoints)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        student_tokens = student(source_images)
        prediction = predictor(student_tokens, source_severity, actions, delta)
        with torch.no_grad():
            target_tokens = teacher(target_images)
    losses = b0_loss(
        prediction,
        target_tokens,
        student_tokens,
        lambda_reg=0.05,
        covariance_weight=0.04,
        variance_target=1.0,
    )
    losses["total"].backward()
    condition_parameters = {
        name: float(parameter.grad.abs().sum())
        for name, parameter in predictor.named_parameters()
        if any(part in name for part in ("severity_conditioner", "action_embedding", "delta_conditioner"))
    }
    if not condition_parameters or min(condition_parameters.values()) <= 0:
        raise RuntimeError("A conditioning parameter did not receive a gradient")
    optimizer.step()
    update_ema(student, teacher, 0.996)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    vif = _vif(images[:1].float(), source_images[:1].float())
    if not torch.isfinite(vif).all():
        raise RuntimeError("VIF smoke result is non-finite")
    synthetic_x = list(range(11))
    synthetic_y = [1 - 0.1 * x + 0.09 * max(0, x - 5) for x in synthetic_x]
    fit = fit_piecewise_hinge(synthetic_x, synthetic_y)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint_path = Path(directory) / "student.pt"
        torch.save(student.state_dict(), checkpoint_path)
        restored = FreshPLIPVisionEncoder(config["plip_config_dir"]).to(device)
        restored.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    summary = {
        "batch_size": args.batch_size,
        "image_shape": list(images.shape),
        "token_shape": list(student_tokens.shape),
        "prediction_shape": list(prediction.shape),
        "loss": float(losses["total"].detach()),
        "vif": float(vif.reshape(-1)[0]),
        "piecewise_breakpoint": fit.breakpoint,
        "seconds": seconds,
        "samples_per_second": args.batch_size / seconds,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
