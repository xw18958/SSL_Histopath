"""Frozen linear-probe evaluation for the Self-Flow pixel-space baseline."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .data import build_balanced_loaders
from .probe import _evaluate, _fit_probe
from .self_flow_training import load_self_flow_checkpoint
from .utils import atomic_json_dump, plot_history, seed_everything, write_csv


@torch.inference_mode()
def extract_self_flow_features(
    config: dict,
    output_dir: Path,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    if not torch.cuda.is_available():
        raise RuntimeError("Self-Flow feature extraction requires CUDA")
    device = torch.device("cuda")
    loaders, _ = build_balanced_loaders(
        config["data_root"],
        config["metadata_csv"],
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        cache_in_ram=bool(config.get("cache_in_ram", True)),
    )
    encoder = load_self_flow_checkpoint(config["encoder_checkpoint"], device)
    if config.get("feature_layer", "final") != "final":
        raise ValueError("Main fair comparison fixes Self-Flow probe features to the final transformer layer")

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


def run_self_flow_linear_probe(config: dict, *, maximum_epochs_override: int | None = None) -> dict[str, Any]:
    """Use the exact B0/I-JEPA probe grid and validation-only model selection."""
    if not torch.cuda.is_available():
        raise RuntimeError("Self-Flow linear probe requires CUDA")
    seed = int(config["seed"])
    if seed != 20260903:
        raise ValueError("Fair benchmark seed is fixed to 20260903")
    seed_everything(seed)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "linear_probe_summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing final probe: {output_dir}")
    atomic_json_dump(config, output_dir / "resolved_config.json")

    started = time.perf_counter()
    features = extract_self_flow_features(config, output_dir)
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
        print(
            json.dumps({key: value for key, value in trial.items() if key not in ("state", "history")}),
            flush=True,
        )

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
            "feature_layer": "final",
        },
        output_dir / "best_linear_probe.pt",
    )
    leaderboard = [
        {key: value for key, value in trial.items() if key not in ("state", "history")} for trial in trials
    ]
    write_csv(leaderboard, output_dir / "probe_leaderboard.csv")
    write_csv(best["history"], output_dir / "probe_metrics.csv")
    plot_history(
        best["history"],
        ["train_loss", "val_loss"],
        output_dir / "probe_losses.png",
        "Self-Flow linear-probe loss",
    )
    plot_history(
        best["history"],
        ["train_accuracy", "val_accuracy", "train_macro_f1", "val_macro_f1"],
        output_dir / "probe_metrics.png",
        "Self-Flow linear-probe metrics",
    )

    report = classification_report(test_labels.numpy(), predictions, output_dict=True, zero_division=0)
    class_rows = [{"class_id": class_id, **report[str(class_id)]} for class_id in range(19)]
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
        "method": "Self-Flow-Pixel-B32",
        "feature_representation": "clean_t0_final_layer_mean_pool",
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
        "test_used_for_selection": False,
    }
    atomic_json_dump(summary, output_dir / "linear_probe_summary.json")
    return summary
