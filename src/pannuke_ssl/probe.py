from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .data import build_balanced_loaders
from .training import load_student_checkpoint
from .utils import atomic_json_dump, plot_history, seed_everything, write_csv


@torch.inference_mode()
def extract_features(config: dict, output_dir: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    if not torch.cuda.is_available():
        raise RuntimeError("Feature extraction requires CUDA")
    device = torch.device("cuda")
    loaders, _ = build_balanced_loaders(
        config["data_root"],
        config["metadata_csv"],
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        cache_in_ram=bool(config.get("cache_in_ram", True)),
    )
    encoder = load_student_checkpoint(config["plip_config_dir"], config["encoder_checkpoint"], device)
    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    cache_values: dict[str, np.ndarray] = {}
    for split, loader in loaders.items():
        features: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for uint8_images, batch_labels in loader:
            images = uint8_images.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                batch_features = encoder(images).mean(dim=1)
            features.append(batch_features.float().cpu())
            labels.append(batch_labels.long())
        split_features = torch.cat(features)
        split_labels = torch.cat(labels)
        result[split] = (split_features, split_labels)
        cache_values[f"{split}_features"] = split_features.numpy()
        cache_values[f"{split}_labels"] = split_labels.numpy()
    np.savez_compressed(output_dir / "frozen_features.npz", **cache_values)
    return result


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
    }


@torch.inference_mode()
def _evaluate(classifier: nn.Module, features: torch.Tensor, labels: torch.Tensor, device: torch.device) -> tuple[float, dict[str, float], np.ndarray]:
    logits = classifier(features.to(device))
    loss = float(nn.functional.cross_entropy(logits, labels.to(device)))
    predictions = logits.argmax(dim=1).cpu().numpy()
    return loss, _metrics(labels.numpy(), predictions), predictions


def _fit_probe(
    features: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    learning_rate: float,
    weight_decay: float,
    maximum_epochs: int,
    patience: int,
    seed: int,
) -> dict[str, Any]:
    seed_everything(seed)
    device = torch.device("cuda")
    classifier = nn.Linear(768, 19).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_features, train_labels = features["train"]
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=256, shuffle=True, generator=generator)
    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    history: list[dict[str, float]] = []
    epochs_without_improvement = 0
    for epoch in range(maximum_epochs):
        classifier.train()
        train_loss = 0.0
        seen = 0
        train_predictions: list[np.ndarray] = []
        train_targets: list[np.ndarray] = []
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(batch_features)
            loss = nn.functional.cross_entropy(logits, batch_labels)
            loss.backward()
            optimizer.step()
            count = batch_labels.shape[0]
            train_loss += float(loss.detach()) * count
            seen += count
            train_predictions.append(logits.argmax(dim=1).detach().cpu().numpy())
            train_targets.append(batch_labels.cpu().numpy())
        classifier.eval()
        val_features, val_labels = features["val"]
        val_loss, val_metrics, _ = _evaluate(classifier, val_features, val_labels, device)
        train_pred = np.concatenate(train_predictions)
        train_true = np.concatenate(train_targets)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss / seen,
            "train_accuracy": float(accuracy_score(train_true, train_pred)),
            "train_macro_f1": float(f1_score(train_true, train_pred, average="macro", zero_division=0)),
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        if val_metrics["macro_f1"] > best_score:
            best_score = val_metrics["macro_f1"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(classifier.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    assert best_state is not None
    return {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "best_epoch": best_epoch,
        "val_macro_f1": best_score,
        "state": best_state,
        "history": history,
    }


def run_linear_probe(config: dict, *, maximum_epochs_override: int | None = None) -> dict[str, Any]:
    seed = int(config["seed"])
    seed_everything(seed)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(config, output_dir / "resolved_config.json")
    started = time.perf_counter()
    features = extract_features(config, output_dir)
    maximum_epochs = int(maximum_epochs_override or config["maximum_epochs"])
    trials: list[dict[str, Any]] = []
    for trial_index, (learning_rate, weight_decay) in enumerate(
        (lr, wd) for lr in config["learning_rates"] for wd in config["weight_decays"]
    ):
        trial = _fit_probe(
            features,
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            maximum_epochs=maximum_epochs,
            patience=min(int(config["early_stopping_patience"]), maximum_epochs),
            seed=seed + trial_index,
        )
        trials.append(trial)
        print(json.dumps({key: value for key, value in trial.items() if key not in ("state", "history")}), flush=True)
    best = max(trials, key=lambda result: result["val_macro_f1"])
    classifier = nn.Linear(768, 19).cuda()
    classifier.load_state_dict(best["state"])
    classifier.eval()
    test_features, test_labels = features["test"]
    test_loss, test_metrics, predictions = _evaluate(classifier, test_features, test_labels, torch.device("cuda"))
    torch.save(
        {
            "classifier": best["state"],
            "learning_rate": best["learning_rate"],
            "weight_decay": best["weight_decay"],
            "best_epoch": best["best_epoch"],
            "validation_macro_f1": best["val_macro_f1"],
        },
        output_dir / "best_linear_probe.pt",
    )
    leaderboard = [
        {key: value for key, value in trial.items() if key not in ("state", "history")} for trial in trials
    ]
    write_csv(leaderboard, output_dir / "probe_leaderboard.csv")
    write_csv(best["history"], output_dir / "probe_metrics.csv")
    plot_history(best["history"], ["train_loss", "val_loss"], output_dir / "probe_losses.png", "Linear-probe loss")
    plot_history(
        best["history"],
        ["train_accuracy", "val_accuracy", "train_macro_f1", "val_macro_f1"],
        output_dir / "probe_metrics.png",
        "Linear-probe metrics",
    )
    report = classification_report(test_labels.numpy(), predictions, output_dict=True, zero_division=0)
    class_rows = [
        {"class_id": class_id, **report[str(class_id)]} for class_id in range(19)
    ]
    write_csv(class_rows, output_dir / "test_per_class_metrics.csv")
    matrix = confusion_matrix(test_labels.numpy(), predictions, labels=list(range(19)))
    np.savetxt(output_dir / "test_confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
    fig, axis = plt.subplots(figsize=(10, 9))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks(range(19))
    axis.set_yticks(range(19))
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(output_dir / "test_confusion_matrix.png", dpi=160)
    plt.close(fig)
    summary = {
        "selection": {
            "learning_rate": best["learning_rate"],
            "weight_decay": best["weight_decay"],
            "best_epoch": best["best_epoch"],
            "validation_macro_f1": best["val_macro_f1"],
        },
        "test_loss": test_loss,
        "test": test_metrics,
        "seconds": time.perf_counter() - started,
        "evaluation_protocol": "transductive_ssl_linear_probe",
    }
    atomic_json_dump(summary, output_dir / "linear_probe_summary.json")
    return summary


def evaluate_saved_probe(config: dict) -> dict[str, Any]:
    output_dir = Path(config["output_dir"])
    features = extract_features(config, output_dir)
    state = torch.load(output_dir / "best_linear_probe.pt", map_location="cuda", weights_only=False)
    classifier = nn.Linear(768, 19).cuda()
    classifier.load_state_dict(state["classifier"])
    classifier.eval()
    test_features, test_labels = features["test"]
    loss, metrics, _ = _evaluate(classifier, test_features, test_labels, torch.device("cuda"))
    return {"test_loss": loss, **metrics}
