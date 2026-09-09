#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pannuke_ssl.config import load_yaml
from pannuke_ssl.ijepa import IJEPA_Predictor
from pannuke_ssl.models import B0Predictor, FreshPLIPVisionEncoder, update_ema
from pannuke_ssl.self_flow import (
    PixelSelfFlowB32,
    make_self_flow_teacher,
    sample_dual_timesteps,
    self_flow_objective,
)
from pannuke_ssl.utils import atomic_json_dump, seed_everything


def trainable_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def capacity_reference(config: dict) -> dict[str, float | int]:
    """Count the actual trainable modules used by the existing B0 and I-JEPA runs."""
    encoder = FreshPLIPVisionEncoder(config["plip_config_dir"], image_size=256)
    encoder_count = trainable_parameters(encoder)

    b0_predictor = B0Predictor(
        input_dim=768,
        predictor_dim=384,
        depth=2,
        heads=6,
        mlp_ratio=4,
        dropout=0.0,
    )
    b0_count = encoder_count + trainable_parameters(b0_predictor)
    del b0_predictor

    ijepa_predictor = IJEPA_Predictor()
    ijepa_count = encoder_count + trainable_parameters(ijepa_predictor)
    del ijepa_predictor, encoder

    model = config["model"]
    self_flow = PixelSelfFlowB32(
        image_size=int(model["image_size"]),
        patch_size=int(model["patch_size"]),
        hidden_size=int(model["hidden_size"]),
        depth=int(model["depth"]),
        num_heads=int(model["num_heads"]),
        mlp_ratio=float(model["mlp_ratio"]),
        student_rep_layer=int(model["student_rep_layer"]),
        teacher_rep_layer=int(model["teacher_rep_layer"]),
    )
    self_flow_count = trainable_parameters(self_flow)
    del self_flow

    ratios = {
        "self_flow_over_b0": self_flow_count / b0_count,
        "self_flow_over_ijepa": self_flow_count / ijepa_count,
    }
    if not all(0.90 <= ratio <= 1.10 for ratio in ratios.values()):
        raise AssertionError(f"Self-Flow trainable capacity is not within 10% of both baselines: {ratios}")
    return {
        "fresh_vit_encoder": encoder_count,
        "b0_student_plus_predictor": b0_count,
        "ijepa_student_plus_predictor": ijepa_count,
        "self_flow_student": self_flow_count,
        **ratios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test fair Self-Flow-Pixel-B/32 implementation")
    parser.add_argument("--config", default="configs/self_flow_pixel_b32_duration_pilot.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    seed_everything(int(config["seed"]))
    if int(config["seed"]) != 20260903:
        raise ValueError("Fair benchmark seed must be 20260903")

    capacity = capacity_reference(config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the training smoke test")
    device = torch.device("cuda")
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

    batch_size = 2
    images = torch.rand((batch_size, 3, 256, 256), device=device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        features = student(images)
    if features.shape != (batch_size, 64, 768):
        raise AssertionError(f"Unexpected clean feature shape: {tuple(features.shape)}")
    if not torch.isfinite(features).all():
        raise FloatingPointError("Non-finite clean Self-Flow features")

    t, s, mask, student_tau, teacher_tau = sample_dual_timesteps(
        batch_size,
        64,
        mask_ratio=float(model["mask_ratio"]),
        device=device,
    )
    if not torch.equal(mask.sum(dim=1), torch.full((batch_size,), 16, device=device, dtype=torch.long)):
        raise AssertionError("25% Dual-Timestep mask must contain exactly 16/64 tokens per image")
    if torch.any(teacher_tau > student_tau):
        raise AssertionError("Teacher must never be noisier than a student token")
    if not all(torch.all((value >= 0) & (value <= 1)) for value in (t, s, student_tau, teacher_tau)):
        raise AssertionError("Timesteps left the [0,1] interval")

    student.train()
    teacher.eval()
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=0.04)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, metrics = self_flow_objective(
            student,
            teacher,
            images,
            mask_ratio=float(model["mask_ratio"]),
            representation_weight=float(model["representation_weight"]),
            student_rep_layer=int(model["student_rep_layer"]),
            teacher_rep_layer=int(model["teacher_rep_layer"]),
        )
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite Self-Flow smoke loss")
    loss.backward()
    final_grad = student.final_layer.linear.weight.grad
    projector_grad = student.projector.linear2.weight.grad
    if final_grad is None or not torch.isfinite(final_grad).all() or float(final_grad.norm()) == 0.0:
        raise AssertionError("Flow loss did not reach the velocity head")
    if projector_grad is None or not torch.isfinite(projector_grad).all() or float(projector_grad.norm()) == 0.0:
        raise AssertionError("Representation loss did not reach the Self-Flow projector")
    gradient_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError("Non-finite clipped gradient norm")
    optimizer.step()
    update_ema(student, teacher, 0.996)
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise AssertionError("EMA teacher unexpectedly has trainable parameters")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "passed": True,
        "method": "Self-Flow-Pixel-B32",
        "capacity": capacity,
        "clean_feature_shape": list(features.shape),
        "mask_tokens_per_image": 16,
        "mask_ratio": float(mask.float().mean()),
        "teacher_never_noisier": True,
        "loss": float(loss.detach()),
        "flow_loss": float(metrics["flow"].detach()),
        "representation_loss": float(metrics["representation"].detach()),
        "gradient_norm_before_clip": float(gradient_norm.detach()),
        "flow_head_gradient_norm": float(final_grad.norm().detach()),
        "projector_gradient_norm": float(projector_grad.norm().detach()),
        "no_labels": True,
        "no_pretrained_vae": True,
        "class_conditioning": False,
    }
    atomic_json_dump(result, output_dir / "smoke_test.json")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
