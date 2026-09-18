import torch

from pannuke_ssl.ijepa_lejepa_fairness_source import SlicedEppsPulley
from pannuke_ssl.ssl_framework import apply_tuned_hyperparameters, load_standard_config, load_tuning_spec
from pannuke_ssl.ssl_methods.simplex_sigreg_lejepa_standard import (
    SlicedSimplexEppsPulley,
    regular_simplex_centers,
)


def test_regular_simplex_geometry_matches_automatic_spacing_rule():
    k, feature_dim, sigma = 8, 32, 1.0
    centers = regular_simplex_centers(k, feature_dim, sigma)
    assert centers.shape == (k, feature_dim)
    assert torch.allclose(centers.mean(0), torch.zeros(feature_dim), atol=1e-7, rtol=0)
    distances = torch.cdist(centers, centers)
    off_diagonal = ~torch.eye(k, dtype=torch.bool)
    assert torch.allclose(distances[off_diagonal], torch.full((k * (k - 1),), 2.0 * sigma), atol=1e-6, rtol=0)
    expected_c = 2.0 * sigma**2 * (k - 1) / k
    assert torch.allclose(centers.square().sum(1), torch.full((k,), expected_c), atol=1e-6, rtol=0)


def test_k1_sigma1_is_original_sigreg_target():
    torch.manual_seed(7)
    x = torch.randn(64, 32)
    original = SlicedEppsPulley(num_slices=64, t_max=3.0, n_points=17)
    simplex = SlicedSimplexEppsPulley(
        feature_dim=32,
        num_components=1,
        sigma=1.0,
        num_slices=64,
        t_max=3.0,
        n_points=17,
    )
    assert torch.allclose(original(x), simplex(x), atol=1e-6, rtol=1e-6)


def test_simplex_method_keeps_lejepa_non_target_settings_identical():
    baseline = load_standard_config("lejepa")
    simplex = load_standard_config("simplex_sigreg_lejepa")
    for key in ("views", "augmentations", "projector", "optimizer", "gradient_clip_norm"):
        assert simplex["method"][key] == baseline["method"][key]
    for key in ("sigreg_lambda", "sigreg_slices", "sigreg_points", "sigreg_t_max"):
        assert simplex["method"]["objective"][key] == baseline["method"]["objective"][key]
    for key in ("seed", "data", "backbone", "training", "representation", "validation", "early_stopping", "downstream"):
        assert simplex[key] == baseline[key]


def test_simplex_sigreg_tuning_grid_and_application():
    c = load_standard_config("simplex_sigreg_lejepa")
    spec = load_tuning_spec("simplex_sigreg_lejepa")
    assert c["method"]["objective"]["simplex_sigma"] == 1.0
    assert spec["parameters"]["simplex_components"]["candidates"] == [2, 4, 8, 16, 32, 64]
    tuned = apply_tuned_hyperparameters(c, {"learning_rate": 0.001, "simplex_components": 16})
    assert tuned["method"]["optimizer"]["peak_lr"] == 0.001
    assert tuned["method"]["objective"]["simplex_components"] == 16
    assert c["method"]["objective"]["simplex_components"] == 2


def test_simplex_requires_enough_projector_dimensions():
    try:
        regular_simplex_centers(6, 4, 1.0)
    except ValueError as exc:
        assert "requires feature_dim" in str(exc)
    else:
        raise AssertionError("Expected K-1 > D to be rejected")
