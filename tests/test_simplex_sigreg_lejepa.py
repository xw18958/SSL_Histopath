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


def test_simplex_sigreg_uses_sequential_k_then_lr_search_with_fixed_sigma():
    c = load_standard_config("simplex_sigreg_lejepa")
    spec = load_tuning_spec("simplex_sigreg_lejepa")
    assert c["method"]["objective"]["simplex_sigma"] == 1.0
    assert spec["parameters"]["simplex_components"]["candidates"] == [2, 4, 8, 16, 32, 64]
    assert spec["search"]["strategy"] == "sequential_greedy"
    assert spec["search"]["order"] == ["simplex_components", "learning_rate"]
    assert spec["search"]["fixed_simplex_sigma"] == 1.0
    assert spec["search"]["k_stage_learning_rate"] == spec["parameters"]["learning_rate"]["source_value"]
    tuned = apply_tuned_hyperparameters(
        c,
        {"learning_rate": 0.001, "simplex_components": 16, "simplex_sigma": 1.0},
    )
    assert tuned["method"]["optimizer"]["peak_lr"] == 0.001
    assert tuned["method"]["objective"]["simplex_components"] == 16
    assert tuned["method"]["objective"]["simplex_sigma"] == 1.0
    assert c["method"]["objective"]["simplex_components"] == 2


def test_simplex_sigma_cannot_be_tuned_away_from_one():
    c = load_standard_config("simplex_sigreg_lejepa")
    try:
        apply_tuned_hyperparameters(c, {"simplex_sigma": 0.5})
    except ValueError as exc:
        assert "fixed to 1.0" in str(exc)
    else:
        raise AssertionError("Expected non-unit simplex sigma to be rejected")


def test_simplex_requires_enough_projector_dimensions():
    try:
        regular_simplex_centers(6, 4, 1.0)
    except ValueError as exc:
        assert "requires feature_dim" in str(exc)
    else:
        raise AssertionError("Expected K-1 > D to be rejected")


def test_simplex_tuning_executes_six_k_trials_then_reuses_source_lr(monkeypatch, tmp_path):
    import pannuke_ssl.ssl_framework.tuning as tuning_module

    c = load_standard_config("simplex_sigreg_lejepa")
    c["output"]["root"] = str(tmp_path)
    calls = []

    def fake_train(config, out, *, epochs, interval, early_stop):
        k = int(config["method"]["objective"]["simplex_components"])
        lr = float(config["method"]["optimizer"]["peak_lr"])
        sigma = float(config["method"]["objective"]["simplex_sigma"])
        calls.append((str(out), k, lr, sigma, epochs, interval, early_stop))
        if "stage_1_K" in str(out):
            score = {2: 0.60, 4: 0.65, 8: 0.80, 16: 0.75, 32: 0.70, 64: 0.68}[k]
        else:
            score = {0.0001: 0.81, 0.0005: 0.82, 0.001: 0.85}[lr]
        return {"best_epoch": 20, "best_validation_linear_macro_f1": score}

    monkeypatch.setattr(tuning_module, "train_ssl", fake_train)
    result = tuning_module.run_tuning(c)

    assert len(calls) == 8
    first_stage = calls[:6]
    second_stage = calls[6:]
    assert [call[1] for call in first_stage] == [2, 4, 8, 16, 32, 64]
    assert all(call[2] == 0.0005 and call[3] == 1.0 for call in first_stage)
    assert all(call[1] == 8 and call[3] == 1.0 for call in second_stage)
    assert [call[2] for call in second_stage] == [0.0001, 0.001]
    assert all(call[4:] == (20, 5, False) for call in calls)
    assert result["search_strategy"] == "sequential_greedy"
    assert result["executed_trial_count"] == 8
    assert len(result["rows"]) == 9
    reused = [row for row in result["rows"] if row.get("reused_from_stage") == "simplex_components"]
    assert len(reused) == 1
    assert reused[0]["stage"] == "learning_rate"
    assert reused[0]["candidate_value"] == 0.0005
    assert reused[0]["status"] == "completed"
    assert result["selected_parameters"] == {
        "learning_rate": 0.001,
        "simplex_components": 8,
        "simplex_sigma": 1.0,
    }
