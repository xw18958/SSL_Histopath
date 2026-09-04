from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pannuke_ssl.calibration import fit_piecewise_hinge
from pannuke_ssl.degradations import (
    DEFOCUS,
    RESOLUTION,
    DegradationEndpoints,
    degrade_uniform,
    disk_kernel,
    sample_adjacent_transitions,
)
from pannuke_ssl.losses import b0_loss
from pannuke_ssl.monitor import feature_diagnostics, improves_macro_f1, weighted_knn_predictions
from pannuke_ssl.models import B0Predictor, FreshPLIPVisionEncoder, make_teacher, update_ema
from pannuke_ssl.training import _weight_decay


PLIP_CONFIG = Path("/raid1/xwan0900/models/plip_model")
DATA_ROOT = Path("/raid1/xwan0900/datasets/PanNuke/data")
METADATA = Path("/raid1/xwan0900/SSL_proj/pannuke19_metadata.csv")


def test_piecewise_fit_recovers_known_breakpoint() -> None:
    x = np.arange(0.0, 11.0)
    y = 1.0 - 0.10 * x + 0.09 * np.maximum(0.0, x - 5.0)
    fit = fit_piecewise_hinge(x, y, minimum_points_per_segment=4)
    assert fit.breakpoint == pytest.approx(5.0)
    assert fit.slope_before == pytest.approx(-0.10)
    assert fit.slope_after == pytest.approx(-0.01)


def test_piecewise_fit_rejects_non_flattening_curve() -> None:
    x = np.arange(0.0, 11.0)
    y = 1.0 - 0.1 * x - 0.1 * np.maximum(0.0, x - 5.0)
    with pytest.raises(RuntimeError, match="no valid flattening"):
        fit_piecewise_hinge(x, y, minimum_points_per_segment=4)


def test_degradation_identity_kernel_and_transition_order() -> None:
    images = torch.rand(3, 3, 32, 32)
    endpoints = DegradationEndpoints(defocus_radius=8.0, resolution_factor=8.0)
    assert torch.equal(degrade_uniform(images, DEFOCUS, 0.0, endpoints), images)
    assert torch.equal(degrade_uniform(images, RESOLUTION, 0.0, endpoints), images)
    kernel = disk_kernel(3.5, device=torch.device("cpu"), dtype=torch.float32)
    assert kernel.sum() == pytest.approx(1.0)
    action, source, target, delta = sample_adjacent_transitions(100, torch.device("cpu"))
    assert set(action.tolist()) <= {0, 1}
    assert torch.all(source - target == 0.25)
    assert torch.all(delta == -0.25)


def test_predictor_shape_no_position_or_dropout_and_condition_gradients() -> None:
    predictor = B0Predictor()
    assert not any("pos" in name.lower() for name, _ in predictor.named_parameters())
    assert all(module.p == 0 for module in predictor.modules() if isinstance(module, torch.nn.Dropout))
    tokens = torch.randn(2, 64, 768, requires_grad=True)
    severity = torch.tensor([0.25, 1.0])
    action = torch.tensor([0, 1])
    delta = torch.full((2,), -0.25)
    output = predictor(tokens, severity, action, delta)
    assert output.shape == (2, 64, 768)
    output.square().mean().backward()
    for prefix in ("severity_conditioner", "action_embedding", "delta_conditioner"):
        grads = [p.grad for name, p in predictor.named_parameters() if name.startswith(prefix)]
        assert grads and all(gradient is not None for gradient in grads)
        assert sum(float(gradient.abs().sum()) for gradient in grads) > 0


def test_loss_and_ema_are_finite() -> None:
    student = torch.nn.Linear(4, 4)
    teacher = make_teacher(student)
    with torch.no_grad():
        student.weight.add_(1)
        before = teacher.weight.clone()
        update_ema(student, teacher, 0.5)
    assert not torch.equal(before, teacher.weight)
    assert not any(parameter.requires_grad for parameter in teacher.parameters())
    values = b0_loss(
        torch.randn(8, 4, 16),
        torch.randn(8, 4, 16),
        torch.randn(8, 4, 16),
        lambda_reg=0.05,
        covariance_weight=0.04,
        variance_target=1.0,
    )
    assert all(torch.isfinite(value) for value in values.values())


def test_weighted_knn_diagnostics_and_checkpoint_tie_break() -> None:
    features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    labels = torch.tensor([0, 0, 1, 1])
    predicted = weighted_knn_predictions(features, labels, torch.tensor([[0.8, 0.2], [0.2, 0.8]]), classes=2, k=2)
    assert predicted.tolist() == [0, 1]
    diagnostics = feature_diagnostics(features, near_constant_std_threshold=0.01)
    assert 0 < diagnostics["feature_effective_rank"] <= 2
    assert 0 <= diagnostics["feature_near_constant_fraction"] <= 1
    assert not improves_macro_f1(0.204, 0.200, 0.005)
    assert improves_macro_f1(0.206, 0.200, 0.005)


def test_cosine_weight_decay_schedule() -> None:
    config = {"weight_decay": 0.04, "final_weight_decay": 0.40}
    assert _weight_decay(config, 0, 101) == pytest.approx(0.04)
    assert _weight_decay(config, 100, 101) == pytest.approx(0.40)


@pytest.mark.skipif(not PLIP_CONFIG.exists(), reason="remote PLIP config is unavailable")
def test_fresh_plip_architecture_and_patch_shape() -> None:
    torch.manual_seed(1)
    first = FreshPLIPVisionEncoder(PLIP_CONFIG)
    torch.manual_seed(2)
    second = FreshPLIPVisionEncoder(PLIP_CONFIG)
    assert first.hidden_size == 768 and first.num_patches == 64
    assert not torch.equal(next(first.parameters()), next(second.parameters()))
    with torch.inference_mode():
        output = first(torch.rand(1, 3, 256, 256))
    assert output.shape == (1, 64, 768)


@pytest.mark.skipif(not DATA_ROOT.exists() or not METADATA.exists(), reason="remote PanNuke data is unavailable")
def test_remote_source_and_balanced_metadata() -> None:
    from collections import Counter

    from pannuke_ssl.parquet import build_source_index, read_metadata, verify_records

    source = build_source_index(DATA_ROOT)
    rows = read_metadata(METADATA)
    verify_records(rows, source)
    assert len(source) == 7901
    assert len(rows) == 2546
    assert Counter(int(row["class_id"]) for row in rows) == Counter({index: 134 for index in range(19)})
    assert Counter(str(row["split"]) for row in rows) == {"train": 2052, "val": 247, "test": 247}
