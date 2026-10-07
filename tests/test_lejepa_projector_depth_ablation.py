from __future__ import annotations

from pathlib import Path

import torch
import yaml

from pannuke_ssl.ijepa_lejepa_fairness_source import LeJEPAFair

ROOT = Path(__file__).resolve().parents[1]


def _compat(mlp_channels=None):
    return {
        "lejepa": {
            "projector_dim": 512,
            "projector_hidden_dim": 2048,
            "projector_mlp_channels": mlp_channels,
            "sigreg_slices": 8,
            "sigreg_t_max": 3.0,
            "sigreg_points": 17,
            "lambda": 0.02,
        }
    }


def _signature(mlp_channels=None):
    model = LeJEPAFair(_compat(mlp_channels), torch.nn.Identity())
    linear_shapes = [
        (module.in_features, module.out_features)
        for module in model.projector.modules()
        if isinstance(module, torch.nn.Linear)
    ]
    bn_dims = [
        module.num_features
        for module in model.projector.modules()
        if isinstance(module, torch.nn.BatchNorm1d)
    ]
    params = sum(parameter.numel() for parameter in model.projector.parameters())
    return model, linear_shapes, bn_dims, params


def test_released_baseline_projector_is_exactly_preserved():
    model, shapes, bn_dims, _ = _signature()
    assert shapes == [(768, 512), (512, 2048), (2048, 2048), (2048, 512)]
    assert bn_dims == [2048, 2048]

    state = model.state_dict()
    assert state["projector.0.weight"].shape == (512, 768)
    assert state["projector.1.0.weight"].shape == (2048, 512)
    assert state["projector.1.4.weight"].shape == (2048, 2048)
    assert state["projector.1.8.weight"].shape == (512, 2048)


def test_controlled_depth_variants_keep_the_same_768_to_512_bottleneck():
    _, shallow_shapes, shallow_bn, shallow_params = _signature([2048, 512])
    _, baseline_shapes, baseline_bn, baseline_params = _signature()
    _, deep_shapes, deep_bn, deep_params = _signature([2048, 2048, 2048, 512])

    assert shallow_shapes == [(768, 512), (512, 2048), (2048, 512)]
    assert baseline_shapes == [(768, 512), (512, 2048), (2048, 2048), (2048, 512)]
    assert deep_shapes == [
        (768, 512),
        (512, 2048),
        (2048, 2048),
        (2048, 2048),
        (2048, 512),
    ]
    assert shallow_bn == [2048]
    assert baseline_bn == [2048, 2048]
    assert deep_bn == [2048, 2048, 2048]
    assert shallow_params < baseline_params < deep_params


def test_registry_uses_only_internal_mlp_depth_and_matched_lr_schedule():
    registry = yaml.safe_load(
        (ROOT / "configs/ssl_standard/experiment_registry.yaml").read_text()
    )["experiments"]
    shallow = registry["ssl-lejepa-proj-shallow-s20260903"]
    deep = registry["ssl-lejepa-proj-deep-s20260903"]

    assert shallow["method"] == deep["method"] == "lejepa"
    assert shallow["role"] == deep["role"] == "projector_depth_controlled_ablation"
    assert shallow["overrides"]["method.projector.mlp_channels"] == [2048, 512]
    assert deep["overrides"]["method.projector.mlp_channels"] == [2048, 2048, 2048, 512]

    for spec in (shallow, deep):
        assert spec["overrides"]["training.max_epochs"] == 250
        assert spec["overrides"]["training.schedule_epochs"] == 300
        assert spec["overrides"]["training.checkpoint_epochs"] == [100, 150, 200, 250]
