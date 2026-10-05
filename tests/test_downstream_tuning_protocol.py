import torch

from pannuke_ssl.ssl_framework import downstream
from pannuke_ssl.ssl_framework import load_standard_config


def test_probe_config_freezes_sequential_greedy_five_epoch_tuning():
    probe = load_standard_config("lejepa")["downstream"]["probe"]
    assert probe["search"] == "sequential_greedy"
    assert probe["learning_rates"] == [0.001, 0.003, 0.01]
    assert probe["weight_decays"] == [0.0, 0.0001]
    assert probe["tuning_epochs"] == 5


def test_sequential_greedy_runs_three_lr_then_two_wd_trials(monkeypatch):
    calls = []

    def fake_fit(features, *, learning_rate, weight_decay, maximum_epochs, patience, seed, num_classes):
        calls.append((learning_rate, weight_decay, maximum_epochs, patience, seed, num_classes))
        lr_score = {0.001: 0.2, 0.003: 0.5, 0.01: 0.3}[learning_rate]
        score = lr_score + (0.1 if weight_decay == 0.0001 else 0.0)
        return {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "best_epoch": 5,
            "val_macro_f1": score,
            "state": {},
            "history": [],
        }

    monkeypatch.setattr(downstream, "_fit_probe", fake_fit)
    features = {
        "train": (torch.empty(2, 3), torch.tensor([0, 1])),
        "val": (torch.empty(2, 3), torch.tensor([0, 1])),
    }
    probe = load_standard_config("lejepa")["downstream"]["probe"]
    lr, wd, board, best = downstream._tune_probe_sequential_greedy(
        features, probe, seed=20260903, num_classes=2
    )

    assert lr == 0.003
    assert wd == 0.0001
    assert len(calls) == 5
    assert calls[:3] == [
        (0.001, 0.0, 5, 5, 20260903, 2),
        (0.003, 0.0, 5, 5, 20260903, 2),
        (0.01, 0.0, 5, 5, 20260903, 2),
    ]
    assert calls[3:] == [
        (0.003, 0.0, 5, 5, 20260903, 2),
        (0.003, 0.0001, 5, 5, 20260903, 2),
    ]
    assert [row["stage"] for row in board] == ["lr", "lr", "lr", "weight_decay", "weight_decay"]
    assert best["val_macro_f1"] == 0.6
