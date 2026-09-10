from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from pannuke_ssl.b0_ln_training import (
    adaptive_failure_count,
    layer_norm_teacher_target,
    verify_protected_manifest,
)
from pannuke_ssl.losses import b0_loss
from pannuke_ssl.models import B0Predictor, make_teacher, update_ema


ROOT = Path("/raid1/xwan0900/SSL_proj")
PILOT = ROOT / "outputs/b0_ln_duration_pilot"


def test_teacher_target_matches_ijepa_stateless_fp32_layer_norm() -> None:
    torch.manual_seed(7)
    teacher = torch.nn.Linear(768, 768).eval().requires_grad_(False)
    images = torch.randn(2, 64, 768, requires_grad=True)
    with torch.no_grad():
        raw = teacher(images)
    target = layer_norm_teacher_target(teacher, images)
    expected = F.layer_norm(raw.float(), (raw.shape[-1],))
    assert target.shape == (2, 64, 768)
    assert target.dtype == torch.float32
    assert not target.requires_grad
    assert torch.equal(target, expected)
    assert float(target.mean(dim=-1).abs().max()) < 3e-6
    assert torch.allclose(target.var(dim=-1, unbiased=False).mean(), torch.tensor(1.0), atol=1e-4, rtol=0)


def test_layer_norm_is_target_only_and_student_predictor_get_gradients() -> None:
    torch.manual_seed(8)
    teacher = torch.nn.Linear(768, 768).eval().requires_grad_(False)
    student_tokens = torch.randn(2, 64, 768, requires_grad=True)
    target_images = torch.randn(2, 64, 768)
    predictor = B0Predictor()
    seen_inputs: list[torch.Tensor] = []
    hook = predictor.input_projection.register_forward_pre_hook(lambda _module, values: seen_inputs.append(values[0].detach().clone()))
    try:
        prediction = predictor(student_tokens, torch.tensor([0.25, 1.0]), torch.tensor([0, 1]), torch.tensor([-0.25, -0.25]))
    finally:
        hook.remove()
    target = layer_norm_teacher_target(teacher, target_images)
    values = b0_loss(prediction, target.detach(), student_tokens, lambda_reg=0.10, covariance_weight=0.04, variance_target=1.0)
    values["total"].backward()
    assert torch.equal(seen_inputs[-1], student_tokens.detach())
    assert student_tokens.grad is not None and float(student_tokens.grad.abs().sum()) > 0
    assert any(parameter.grad is not None and float(parameter.grad.abs().sum()) > 0 for parameter in predictor.parameters())
    assert not any(parameter.grad is not None for parameter in teacher.parameters())
    assert all(torch.isfinite(value) for value in values.values())


def test_ema_and_adaptive_stop_counter_are_unchanged_or_explicit() -> None:
    student = torch.nn.Linear(4, 4)
    teacher = make_teacher(student)
    with torch.no_grad():
        student.weight.add_(1.0)
        before = teacher.weight.detach().clone()
        update_ema(student, teacher, 0.996)
    assert torch.allclose(teacher.weight, before * 0.996 + student.weight.detach() * 0.004)
    assert adaptive_failure_count(100, False, 0) == 0
    assert adaptive_failure_count(110, False, 0) == 1
    assert adaptive_failure_count(120, False, 4) == 5
    assert adaptive_failure_count(120, True, 4) == 0


def test_existing_reference_manifest_is_unchanged() -> None:
    manifest = verify_protected_manifest(PILOT)
    assert manifest["purpose"] == "pre-B0-LN protected reference manifest"
    assert len(manifest["protected_paths"]) == 205
