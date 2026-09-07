from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pannuke_ssl.b1 import B1VelocityPredictor, residual_endpoint
from pannuke_ssl.b1_losses import b1_loss
from pannuke_ssl.b1_training import verify_protected_manifest
from pannuke_ssl.degradations import sample_adjacent_transitions
from pannuke_ssl.monitor import improves_macro_f1


def test_b1_only_uses_adjacent_reverse_transitions() -> None:
    actions, source, target, delta = sample_adjacent_transitions(256, torch.device("cpu"))
    assert set(actions.tolist()) <= {0, 1}
    assert torch.equal(source - target, torch.full_like(source, 0.25))
    assert torch.equal(delta, torch.full_like(delta, -0.25))
    assert set(source.tolist()) == {0.25, 0.5, 0.75, 1.0}


def test_b1_residual_endpoint_algebra_and_sign() -> None:
    source = torch.randn(3, 4, 5)
    delta = torch.full((3,), -0.25)
    zero = residual_endpoint(source, torch.zeros_like(source), delta)
    assert torch.equal(zero, source)
    target = torch.randn_like(source)
    velocity = (target - source) / delta[:, None, None]
    assert torch.allclose(residual_endpoint(source, velocity, delta), target)
    assert not torch.allclose(source - delta[:, None, None] * velocity, target)
    with pytest.raises(ValueError):
        residual_endpoint(source, velocity[:2], delta[:2])


def test_b1_predictor_has_b0_matched_conditioning_and_gradients() -> None:
    predictor = B1VelocityPredictor(input_dim=16, predictor_dim=12, depth=2, heads=3, mlp_ratio=4)
    assert not any("pos" in name.lower() for name, _ in predictor.named_parameters())
    assert all(module.p == 0 for module in predictor.modules() if isinstance(module, torch.nn.Dropout))
    tokens = torch.randn(3, 4, 16, requires_grad=True)
    velocity = predictor(tokens, torch.tensor([1.0, 0.75, 0.25]), torch.tensor([0, 1, 0]), torch.full((3,), -0.25))
    assert velocity.shape == tokens.shape
    velocity.square().mean().backward()
    for prefix in ("severity_conditioner", "action_embedding", "delta_conditioner"):
        gradients = [parameter.grad for name, parameter in predictor.named_parameters() if name.startswith(prefix)]
        assert gradients and all(gradient is not None for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0


def test_b1_endpoint_loss_is_finite_and_keeps_vicreg_control() -> None:
    values = b1_loss(
        torch.randn(8, 4, 16),
        torch.randn(8, 4, 16),
        torch.randn(8, 4, 16),
        lambda_reg=0.10,
        covariance_weight=0.04,
        variance_target=1.0,
    )
    assert set(values) == {"total", "prediction", "regularizer", "variance", "covariance", "embedding_std"}
    assert all(torch.isfinite(value) for value in values.values())


def test_duration_checkpoint_replacement_is_strict_and_earlier_ties_hold() -> None:
    assert not improves_macro_f1(0.205, 0.200, 0.005)
    assert not improves_macro_f1(0.204, 0.200, 0.005)
    assert improves_macro_f1(0.206, 0.200, 0.005)


def test_remote_b0_ijepa_protected_manifest_is_unchanged() -> None:
    output = Path("/raid1/xwan0900/SSL_proj/outputs/b1_duration_pilot")
    if not output.exists():
        pytest.skip("B1 remote output root is unavailable")
    manifest = verify_protected_manifest(output)
    assert len(manifest["protected_paths"]) >= 1
