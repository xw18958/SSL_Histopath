from __future__ import annotations

import torch

from pannuke_ssl.b0_delta_training import _batch_residual_diagnostics, teacher_delta_target


def test_teacher_delta_direction_and_detach() -> None:
    source = torch.tensor([[[1.0, 3.0]]], requires_grad=True)
    target = torch.tensor([[[4.0, 2.0]]], requires_grad=True)
    delta = teacher_delta_target(source, target)
    assert torch.equal(delta, torch.tensor([[[3.0, -1.0]]]))
    assert delta.dtype == torch.float32
    assert not delta.requires_grad


def test_residual_diagnostics_are_finite_and_reward_better_than_zero() -> None:
    target = torch.tensor([[[2.0, -2.0]], [[1.0, -1.0]]])
    prediction = target * 0.5
    values = _batch_residual_diagnostics(prediction, target)
    assert values["prediction"].shape == (2,)
    assert values["zero_prediction"].shape == (2,)
    assert values["cosine"].shape == (2,)
    assert torch.isfinite(values["prediction"]).all()
    assert torch.isfinite(values["zero_prediction"]).all()
    assert torch.isfinite(values["cosine"]).all()
    assert torch.all(values["prediction"] < values["zero_prediction"])
    assert torch.allclose(values["cosine"], torch.ones(2))


def test_identical_teacher_endpoints_have_zero_residual() -> None:
    values = torch.randn(3, 64, 768)
    delta = teacher_delta_target(values, values)
    assert torch.count_nonzero(delta) == 0
