from __future__ import annotations

import torch

from pannuke_ssl.change_jepa import ChangeJEPAPredictor, map_signed_change_to_unit
from pannuke_ssl.change_jepa_training import should_early_stop


def test_change_mapping_is_invertible_and_preserves_sign() -> None:
    source = torch.tensor([[[[0.0, 0.8, 0.4]]]], dtype=torch.float32)
    target = torch.tensor([[[[1.0, 0.2, 0.4]]]], dtype=torch.float32)
    mapped = map_signed_change_to_unit(source, target)
    expected_change = target - source
    reconstructed = 2.0 * mapped - 1.0
    assert torch.allclose(
        mapped,
        torch.tensor([[[[1.0, 0.2, 0.5]]]], dtype=torch.float32),
        atol=1e-7,
        rtol=0.0,
    )
    assert torch.allclose(reconstructed, expected_change, atol=1e-7, rtol=0.0)
    assert mapped[0, 0, 0, 0] > 0.5
    assert mapped[0, 0, 0, 1] < 0.5
    assert mapped[0, 0, 0, 2] == 0.5


def test_change_jepa_predictor_output_shape_and_gradients() -> None:
    predictor = ChangeJEPAPredictor()
    context = torch.randn(4, 64, 768, requires_grad=True)
    source_severity = torch.tensor([1.0, 0.75, 0.5, 0.25])
    action = torch.tensor([0, 1, 0, 1])
    delta = torch.full((4,), -0.25)
    prediction = predictor(context, source_severity, action, delta)
    assert prediction.shape == (4, 64, 768)
    prediction.square().mean().backward()
    assert context.grad is not None
    assert torch.isfinite(context.grad).all()
    assert predictor.query_token.grad is not None


def test_three_monitor_early_stop_rule() -> None:
    assert not should_early_stop(0, 3)
    assert not should_early_stop(1, 3)
    assert not should_early_stop(2, 3)
    assert should_early_stop(3, 3)
    assert should_early_stop(4, 3)
