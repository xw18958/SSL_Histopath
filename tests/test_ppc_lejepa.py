from __future__ import annotations

import copy
import inspect

import pytest
import torch
from torch import nn

from pannuke_ssl.ssl_framework.config import (
    apply_tuned_hyperparameters,
    load_standard_config,
    load_tuning_spec,
)
from pannuke_ssl.ssl_framework.ppc_campaign import (
    config as campaign_config,
    matrix as campaign_matrix,
    plan as campaign_plan,
)
from pannuke_ssl.ssl_framework.trainer import save_checkpoint
from pannuke_ssl.ssl_methods.lejepa_standard import StandardLeJEPA
from pannuke_ssl.ssl_methods.ppc_lejepa_standard import StandardPPCLeJEPA, _module_sha256


def _runtime_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SSL_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("SSL_MODEL_ROOT", str(tmp_path / "models"))
    monkeypatch.setenv("SSL_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("SSL_PPC_RUN_ROOT", str(tmp_path / "ppc_runs"))
    monkeypatch.setenv("SSL_PROJECT_ROOT", str(tmp_path / "project"))


class _EncoderHolder(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 4, bias=True)

    def forward(self, x):
        return self.base(x)


class _ToyLeJEPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _EncoderHolder()
        self.projector = nn.Sequential(
            nn.Linear(4, 8, bias=True),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 4, bias=True),
        )
        self.sigreg = _ToySIGReg()
        self.lamb = 0.02

    def _features(self, images):
        return self.encoder(images)

    def forward(self, global_views, local_views):
        batch = global_views[0].shape[0]
        global_features = self._features(torch.cat(global_views, dim=0))
        local_features = self._features(torch.cat(local_views, dim=0))
        all_features = torch.cat((global_features, local_features), dim=0)
        projected = self.projector(all_features)
        projected = projected.view(len(global_views) + len(local_views), batch, -1)
        center = projected[: len(global_views)].mean(0)
        inv = (center.unsqueeze(0) - projected).square().mean()
        sig = self.sigreg(projected.reshape(-1, projected.size(-1)))
        return inv + self.lamb * sig, inv, sig, projected


class _ToySIGReg(nn.Module):
    def forward(self, x):
        return 0.5 * x.square().mean()


def _dummy_ppc(ppc_lambda=0.05):
    torch.manual_seed(11)
    method = StandardPPCLeJEPA.__new__(StandardPPCLeJEPA)
    nn.Module.__init__(method)
    method.device = torch.device("cpu")
    method.config = {
        "seed": 20260903,
        "training": {"max_epochs": 250, "schedule_epochs": 250},
        "method": {
            "optimizer": {
                "peak_lr": 5e-4,
                "weight_decay": 0.05,
                "warmup_fraction": 0.10,
                "final_lr_ratio": 0.001,
            }
        },
    }
    method.model = _ToyLeJEPA()
    method.ppc_lambda = float(ppc_lambda)
    method.ppc_epsilon = 1e-8
    method.reference_projector = copy.deepcopy(method.model.projector)
    method.reference_projector.requires_grad_(False)
    method.reference_projector.eval()
    method._initial_trainable_projector_sha256 = _module_sha256(method.model.projector)
    method._initial_reference_projector_sha256 = _module_sha256(method.reference_projector)
    return method


def _baseline_from(ppc):
    baseline = StandardLeJEPA.__new__(StandardLeJEPA)
    nn.Module.__init__(baseline)
    baseline.device = torch.device("cpu")
    baseline.config = copy.deepcopy(ppc.config)
    baseline.model = copy.deepcopy(ppc.model)
    return baseline


def _batch():
    generator = torch.Generator().manual_seed(123)
    return {
        "global_views": torch.randn(16, 2, 4, generator=generator),
        "local_views": torch.randn(16, 6, 4, generator=generator),
    }


def _bn_state(module):
    return {
        name: (
            child.running_mean.detach().clone(),
            child.running_var.detach().clone(),
            child.num_batches_tracked.detach().clone(),
        )
        for name, child in module.named_modules()
        if isinstance(child, nn.modules.batchnorm._BatchNorm)
    }


def test_previous_reset_ppc_is_removed(monkeypatch, tmp_path):
    _runtime_env(monkeypatch, tmp_path)
    c = load_standard_config("ppc_lejepa")
    assert "projector_lifecycle" not in c["method"]
    assert not hasattr(StandardPPCLeJEPA, "after_epoch")
    source = inspect.getsource(StandardPPCLeJEPA)
    assert "optimizer.state.pop" not in source
    assert "_reset_projector" not in source


def test_ppc_config_matches_standard_lejepa_except_plasticity_regularizer(monkeypatch, tmp_path):
    _runtime_env(monkeypatch, tmp_path)
    baseline = load_standard_config("lejepa")
    ppc = load_standard_config("ppc_lejepa")

    for key in (
        "seed",
        "data",
        "backbone",
        "training",
        "representation",
        "validation",
        "early_stopping",
        "downstream",
    ):
        assert ppc[key] == baseline[key]

    for key in ("views", "augmentations", "projector", "objective", "optimizer", "gradient_clip_norm"):
        assert ppc["method"][key] == baseline["method"][key]

    plasticity = ppc["method"]["projector_plasticity"]
    assert plasticity["lambda"] == pytest.approx(0.05)
    assert plasticity["epsilon"] == pytest.approx(1e-8)
    assert plasticity["lambda_candidates"] == [0.01, 0.05, 0.10]
    assert plasticity["standard_lejepa_lambda"] == 0.0
    assert ppc["training"]["max_epochs"] == 250
    assert ppc["training"].get("schedule_epochs", ppc["training"]["max_epochs"]) == 250
    assert ppc["training"]["checkpoint_epochs"] == [100, 150, 200, 250]


def test_ppc_tuning_is_sequential_greedy_lambda_only(monkeypatch, tmp_path):
    _runtime_env(monkeypatch, tmp_path)
    spec = load_tuning_spec("ppc_lejepa")
    assert spec["search"]["strategy"] == "sequential_greedy"
    assert spec["search"]["order"] == ["ppc_lambda"]
    assert spec["search"]["fixed_learning_rate"] == pytest.approx(5e-4)
    assert spec["parameters"]["ppc_lambda"]["candidates"] == [0.01, 0.05, 0.10]


def test_ppc_tuning_executes_three_lambda_trials_with_fixed_lr(monkeypatch, tmp_path):
    import pannuke_ssl.ssl_framework.tuning as tuning_module

    _runtime_env(monkeypatch, tmp_path)
    c = load_standard_config("ppc_lejepa")
    c["output"]["root"] = str(tmp_path)
    calls = []

    def fake_train(config, out, *, epochs, interval, early_stop):
        value = float(config["method"]["projector_plasticity"]["lambda"])
        calls.append(
            (
                value,
                float(config["method"]["optimizer"]["peak_lr"]),
                epochs,
                interval,
                early_stop,
                str(out),
            )
        )
        return {
            "best_epoch": 20,
            "best_validation_linear_macro_f1": {0.01: 0.70, 0.05: 0.80, 0.10: 0.75}[value],
        }

    monkeypatch.setattr(tuning_module, "train_ssl", fake_train)
    result = tuning_module.run_tuning(c)

    assert [call[0] for call in calls] == [0.01, 0.05, 0.10]
    assert all(call[1] == pytest.approx(5e-4) for call in calls)
    assert all(call[2:5] == (20, 5, False) for call in calls)
    assert result["search_strategy"] == "sequential_greedy"
    assert result["executed_trial_count"] == 3
    assert result["selected_parameters"] == {"ppc_lambda": 0.05}
    assert result["selected_value"] == pytest.approx(0.05)
    assert result["test_used"] is False


def test_ppc_campaign_preserves_250_protocol_and_72_evaluations(monkeypatch, tmp_path):
    _runtime_env(monkeypatch, tmp_path)
    p = campaign_plan()
    c = campaign_config()
    rows = campaign_matrix()
    assert p["lambda_candidates"] == [0.01, 0.05, 0.10]
    assert p["smoke_lambda"] == pytest.approx(0.05)
    assert p["stop_epoch"] == p["schedule_epochs"] == 250
    assert p["checkpoint_epochs"] == [100, 150, 200, 250]
    assert c["method"]["name"] == "ppc_lejepa"
    assert c["training"]["max_epochs"] == 250
    assert c["training"]["schedule_epochs"] == 250
    assert c["training"]["checkpoint_epochs"] == [100, 150, 200, 250]
    assert len(rows) == 72
    assert sum(row["task"] == "classification" for row in rows) == 60
    assert sum(row["task"] == "retrieval" for row in rows) == 12


def test_reference_projector_is_exact_frozen_initial_snapshot():
    method = _dummy_ppc()
    assert _module_sha256(method.projector) == _module_sha256(method.reference_projector)
    assert all(not parameter.requires_grad for parameter in method.reference_projector.parameters())
    assert not method.reference_projector.training

    x = torch.randn(32, 4)
    method.projector.eval()
    with torch.no_grad():
        trainable = method.projector(x)
        reference = method.reference_projector(x)
    assert torch.equal(trainable, reference)


def test_ppc_regularizer_is_zero_at_initialization():
    method = _dummy_ppc()
    features = torch.randn(64, 4, requires_grad=True)
    regularizer = method.ppc_regularizer_from_features(features)
    assert float(regularizer.detach()) == pytest.approx(0.0, abs=1e-12)


def test_ppc_regularizer_gradients_update_projector_not_encoder():
    method = _dummy_ppc()
    with torch.no_grad():
        next(method.projector.parameters()).add_(0.01)

    features = method.model._features(torch.randn(64, 4))
    regularizer = method.ppc_regularizer_from_features(features)
    regularizer.backward()

    projector_grad = sum(
        float(parameter.grad.detach().square().sum())
        for parameter in method.projector.parameters()
        if parameter.grad is not None
    )
    assert projector_grad > 0.0
    assert all(parameter.grad is None for parameter in method.encoder.parameters())


def test_reference_parameters_and_bn_stats_never_change_during_training():
    method = _dummy_ppc()
    optimizer = method.build_optimizer()
    reference_hash = _module_sha256(method.reference_projector)
    reference_bn = _bn_state(method.reference_projector)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert all(id(parameter) not in optimizer_ids for parameter in method.reference_projector.parameters())

    for _ in range(3):
        method.train_mode()
        optimizer.zero_grad(set_to_none=True)
        result = method.training_step(_batch(), bf16=False)
        result.loss.backward()
        optimizer.step()

    assert _module_sha256(method.reference_projector) == reference_hash
    assert _module_sha256(method.projector) != method._initial_trainable_projector_sha256
    assert not method.reference_projector.training
    for name, state in _bn_state(method.reference_projector).items():
        before = reference_bn[name]
        assert all(torch.equal(current, previous) for current, previous in zip(state, before))


def test_ordinary_lejepa_loss_still_backpropagates_into_encoder():
    method = _dummy_ppc()
    method.zero_grad(set_to_none=True)
    result = StandardLeJEPA.training_step(method, _batch(), bf16=False)
    result.loss.backward()
    encoder_grad = sum(
        float(parameter.grad.detach().square().sum())
        for parameter in method.encoder.parameters()
        if parameter.grad is not None
    )
    assert encoder_grad > 0.0


def test_lambda_zero_reproduces_standard_lejepa_behavior_exactly():
    ppc = _dummy_ppc(ppc_lambda=0.0)
    baseline = _baseline_from(ppc)
    batch = _batch()
    baseline.train_mode()
    ppc.train_mode()

    baseline_result = baseline.training_step(batch, bf16=False)
    ppc_result = ppc.training_step(batch, bf16=False)

    assert torch.equal(ppc_result.loss, baseline_result.loss)
    assert ppc_result.metrics["invariance_loss"] == baseline_result.metrics["invariance_loss"]
    assert ppc_result.metrics["sigreg_loss"] == baseline_result.metrics["sigreg_loss"]
    assert ppc_result.metrics["ppc_regularizer"] == 0.0
    assert ppc_result.metrics["ppc_lambda"] == 0.0
    for (key_a, value_a), (key_b, value_b) in zip(
        ppc.model.state_dict().items(), baseline.model.state_dict().items()
    ):
        assert key_a == key_b
        assert torch.equal(value_a, value_b)


def test_total_loss_exactly_matches_lejepa_plus_lambda_ppc():
    method = _dummy_ppc(ppc_lambda=0.05)
    method.train_mode()
    result = method.training_step(_batch(), bf16=False)
    metrics = result.metrics
    expected = (
        metrics["invariance_loss"]
        + 0.02 * metrics["sigreg_loss"]
        + 0.05 * metrics["ppc_regularizer"]
    )
    assert float(result.loss.detach()) == pytest.approx(expected, rel=1e-6, abs=1e-7)
    assert metrics["total_loss"] == pytest.approx(expected, rel=1e-6, abs=1e-7)
    assert metrics["ppc_lambda"] == pytest.approx(0.05)


def test_ppc_checkpoint_logs_frozen_reference_and_trainable_drift(tmp_path):
    method = _dummy_ppc()
    optimizer = method.build_optimizer()
    optimizer.zero_grad(set_to_none=True)
    result = method.training_step(_batch(), bf16=False)
    result.loss.backward()
    optimizer.step()

    config = {
        "method": {"name": "ppc_lejepa"},
        "training": {"max_epochs": 250, "schedule_epochs": 250},
    }
    checkpoint = tmp_path / "epoch_100.pt"
    save_checkpoint(checkpoint, method, config, 100, {"fixed_checkpoint": True})
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    meta = payload["method_metadata"]
    assert meta["method"] == "ppc_lejepa"
    assert meta["ppc_lambda"] == pytest.approx(0.05)
    assert meta["ppc_epsilon"] == pytest.approx(1e-8)
    assert meta["ppc_reference_unchanged"]
    assert meta["ppc_reference_frozen"]
    assert meta["ppc_reference_bn_eval"]
    assert meta["ppc_trainable_changed_from_initial"]
    assert meta["ppc_encoder_stop_gradient"]


def test_apply_selected_lambda_only_changes_ppc_regularizer(monkeypatch, tmp_path):
    _runtime_env(monkeypatch, tmp_path)
    c = load_standard_config("ppc_lejepa")
    tuned = apply_tuned_hyperparameters(c, {"ppc_lambda": 0.10})
    assert tuned["method"]["projector_plasticity"]["lambda"] == pytest.approx(0.10)
    assert tuned["method"]["optimizer"] == c["method"]["optimizer"]
    assert tuned["method"]["objective"] == c["method"]["objective"]
