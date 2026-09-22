import pytest
from pannuke_ssl.ssl_framework import EarlyStopper, load_standard_config, load_tuning_spec
from pannuke_ssl.data import _split_rows_with_balanced_holdouts


def test_safe_early_stop_gate():
    s = EarlyStopper(enabled=True, min_epochs=60, patience=5, delta=0.005)
    assert s.update(10, 0.5)["improved"]
    for e in (20, 30, 40, 50, 60):
        assert s.update(e, 0.503)["stale_monitors"] == 0
    for i, e in enumerate((70, 80, 90, 100), 1):
        u = s.update(e, 0.503)
        assert u["stale_monitors"] == i and not u["should_stop"]
    assert s.update(110, 0.503)["should_stop"]


def test_delta_improvement_resets_patience():
    s = EarlyStopper(enabled=True, min_epochs=60, patience=5, delta=0.005)
    s.update(10, 0.5)
    s.update(70, 0.502)
    u = s.update(80, 0.505)
    assert u["improved"] and u["best_score"] == pytest.approx(0.505) and u["stale_monitors"] == 0


def test_source_learning_rates_must_be_in_tuning_candidates():
    expected = {"ijepa": 0.001, "lejepa": 0.0005, "simplex_sigreg_lejepa": 0.0005, "dinov3": 0.001}
    for name, source in expected.items():
        load_standard_config(name)
        spec = load_tuning_spec(name)
        p = spec["parameters"]["learning_rate"]
        assert float(p["source_value"]) == source
        assert source in [float(x) for x in p["candidates"]]


def test_pannuke_protocol_is_train_only_ssl_with_balanced_holdouts():
    c = load_standard_config("lejepa")
    assert c["data"]["source_images"] == 7901
    assert c["data"]["ssl_split"] == "train"
    assert c["data"]["expected_ssl_images"] == 6305
    assert c["downstream"]["train_count"] == 6305
    assert c["downstream"]["validation_count"] == 798
    assert c["downstream"]["test_count"] == 798
    assert c["downstream"]["validation_count"] // 19 == 42
    assert c["downstream"]["test_count"] // 19 == 42



def test_generic_split_helper_allows_imbalanced_train_but_requires_balanced_holdouts():
    rows = []
    sample_index = 0
    for class_id in range(3):
        for _ in range(class_id + 2):
            rows.append({"fold": 1, "sample_index": sample_index, "tissue_label": str(class_id), "class_id": class_id, "split": "train"})
            sample_index += 1
        for split in ("val", "test"):
            for _ in range(2):
                rows.append({"fold": 1, "sample_index": sample_index, "tissue_label": str(class_id), "class_id": class_id, "split": split})
                sample_index += 1
    parts = _split_rows_with_balanced_holdouts(rows, num_classes=3)
    assert [len(parts[name]) for name in ("train", "val", "test")] == [9, 6, 6]

    broken = list(rows)
    broken.pop(next(i for i, row in enumerate(broken) if row["split"] == "val" and row["class_id"] == 2))
    with pytest.raises(ValueError, match="val split must be balanced"):
        _split_rows_with_balanced_holdouts(broken, num_classes=3)

def test_ijepa_source_eval_and_ema_metadata():
    c = load_standard_config("ijepa")
    meta = c["method"]["source_metadata"]
    assert meta["evaluation_encoder"] == "target_encoder_ema"
    assert meta["evaluation_pooling"] == "average_patch_tokens"
    assert meta["source_ema"] == [0.996, 1.0]
    assert meta["source_ema_schedule"] == "linear"


def test_dinov3_source_eval_and_base_objective_metadata():
    c = load_standard_config("dinov3")
    m = c["method"]
    meta = m["source_metadata"]
    assert meta["evaluation_encoder"] == "ema_teacher"
    assert meta["evaluation_pooling"] == "mean_patch_tokens"
    assert meta["gram_anchoring"] is False
    assert m["views"]["global_count"] == 2 and m["views"]["local_count"] == 8
    assert m["objective"]["mask_sample_probability"] == pytest.approx(0.5)
    assert m["objective"]["mask_ratio"] == [0.1, 0.5]


def test_common_protocol_matches_across_methods():
    configs = [load_standard_config(name) for name in ("ijepa", "lejepa", "simplex_sigreg_lejepa", "dinov3")]
    paths = [
        ("seed",),
        ("data", "source_images"),
        ("data", "expected_ssl_images"),
        ("data", "ssl_split"),
        ("training", "max_epochs"),
        ("training", "batch_size"),
        ("validation", "selection_metric"),
        ("validation", "interval_epochs"),
        ("early_stopping", "min_epochs"),
        ("early_stopping", "patience_monitors"),
        ("downstream", "train_count"),
        ("downstream", "validation_count"),
        ("downstream", "test_count"),
    ]
    for path in paths:
        values = []
        for config in configs:
            value = config
            for key in path:
                value = value[key]
            values.append(value)
        assert values[1:] == values[:-1]
