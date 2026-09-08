from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pannuke_ssl.change_jepa import build_models, map_signed_change_to_unit, objective
from pannuke_ssl.change_jepa_training import validate_config
from pannuke_ssl.config import load_yaml
from pannuke_ssl.data import build_ssl_loader
from pannuke_ssl.degradations import degrade_batch, sample_adjacent_transitions
from pannuke_ssl.training import _endpoints
from pannuke_ssl.utils import seed_everything


ROOT = Path("/raid1/xwan0900/SSL_proj")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/change_jepa_duration_pilot.yaml"),
    )
    args = parser.parse_args()
    config = load_yaml(args.config)
    validate_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("Smoke test requires CUDA")

    seed_everything(int(config["seed"]))
    device = torch.device("cuda")
    train = config["train"]
    endpoints = _endpoints(config["endpoints_json"])
    loader, _ = build_ssl_loader(
        config["data_root"],
        batch_size=int(train["batch_size"]),
        num_workers=int(train["num_workers"]),
        cache_in_ram=bool(train["cache_in_ram"]),
    )
    uint8_images = next(iter(loader))
    images = uint8_images.to(device=device, dtype=torch.float32).div_(255.0)
    actions, source_severity, target_severity, delta = sample_adjacent_transitions(images.shape[0], device)
    source_images = degrade_batch(images, actions, source_severity, endpoints)
    target_images = degrade_batch(images, actions, target_severity, endpoints)

    mapped = map_signed_change_to_unit(source_images, target_images)
    reconstructed = 2.0 * mapped.float() - 1.0
    original_change = target_images.float() - source_images.float()
    reconstruction_error = float((reconstructed - original_change).abs().max())
    if reconstruction_error > 2e-6:
        raise AssertionError(f"Change mapping is not numerically invertible: {reconstruction_error}")

    student, teacher, predictor = build_models(config, device)
    trainable = list(student.parameters()) + list(predictor.parameters())
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(train["bf16"])):
        loss, prediction, target, student_tokens, change_images = objective(
            student,
            teacher,
            predictor,
            source_images,
            target_images,
            source_severity,
            actions,
            delta,
        )
    if prediction.shape != target.shape or prediction.shape[1:] != (64, 768):
        raise AssertionError(f"Unexpected prediction/target shapes: {prediction.shape}, {target.shape}")
    if student_tokens.shape[1:] != (64, 768):
        raise AssertionError(f"Unexpected student token shape: {student_tokens.shape}")
    if not torch.isfinite(loss):
        raise AssertionError("Non-finite objective")
    if float(change_images.min()) < 0.0 or float(change_images.max()) > 1.0:
        raise AssertionError("Mapped change image is outside [0, 1]")

    loss.backward()
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in trainable
    )
    if not finite_gradients:
        raise AssertionError("Non-finite gradients")
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise AssertionError("EMA teacher unexpectedly received gradients")

    print(
        json.dumps(
            {
                "passed": True,
                "loss": float(loss.detach()),
                "prediction_shape": list(prediction.shape),
                "target_shape": list(target.shape),
                "change_range": [float(change_images.min()), float(change_images.max())],
                "mapping_max_reconstruction_error": reconstruction_error,
                "teacher_gradients_none": True,
                "student_predictor_gradients_finite": True,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
