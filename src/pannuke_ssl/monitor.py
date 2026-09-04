"""Validation-only representation monitoring for SSL duration selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from .data import build_balanced_loaders
from .utils import plot_history, write_csv


def _classification_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    target = labels.cpu().numpy()
    predicted = predictions.cpu().numpy()
    return {
        "accuracy": float(accuracy_score(target, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "macro_f1": float(f1_score(target, predicted, average="macro", zero_division=0)),
    }


@torch.inference_mode()
def weighted_knn_predictions(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    query_features: torch.Tensor,
    *,
    classes: int,
    k: int = 20,
    temperature: float = 0.07,
) -> torch.Tensor:
    """DINO-style cosine k-NN classifier with exponentially weighted votes."""
    if k <= 0 or k > train_features.shape[0]:
        raise ValueError("k must be in [1, number of training examples]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    train = F.normalize(train_features.float(), dim=1)
    query = F.normalize(query_features.float(), dim=1)
    similarities = query @ train.T
    values, indices = similarities.topk(k, dim=1, largest=True, sorted=False)
    weights = torch.exp((values - values.max(dim=1, keepdim=True).values) / temperature)
    votes = torch.zeros((query.shape[0], classes), device=query.device, dtype=torch.float32)
    votes.scatter_add_(1, train_labels[indices].long(), weights)
    return votes.argmax(dim=1)


@torch.inference_mode()
def feature_diagnostics(features: torch.Tensor, *, near_constant_std_threshold: float) -> dict[str, float]:
    """Collapse diagnostics on mean-pooled patch-token features."""
    values = features.float()
    centered = values - values.mean(dim=0, keepdim=True)
    dimension_std = centered.std(dim=0, unbiased=True)
    covariance = centered.T @ centered / max(1, values.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(torch.finfo(eigenvalues.dtype).eps)
    entropy = -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).eps).log()).sum()
    effective_rank = entropy.exp()
    sample_count = min(1024, values.shape[0])
    normalized = F.normalize(values[:sample_count], dim=1)
    pairwise = normalized @ normalized.T
    mean_pairwise_cosine = (pairwise.sum() - sample_count) / max(1, sample_count * (sample_count - 1))
    return {
        "feature_embedding_std": float(dimension_std.mean().cpu()),
        "feature_effective_rank": float(effective_rank.cpu()),
        "feature_effective_rank_fraction": float((effective_rank / values.shape[1]).cpu()),
        "feature_mean_pairwise_cosine": float(mean_pairwise_cosine.cpu()),
        "feature_near_constant_fraction": float((dimension_std < near_constant_std_threshold).float().mean().cpu()),
    }


def _fixed_linear_predictions(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    query_features: torch.Tensor,
    *,
    classes: int,
    seed: int,
    max_iter: int,
    device: torch.device,
) -> torch.Tensor:
    """A fixed full-batch LBFGS probe, deliberately not tuned per checkpoint."""
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = ((train_features - mean) / std).to(device=device, dtype=torch.float32)
    y_train = train_labels.to(device=device, dtype=torch.long)
    x_query = ((query_features - mean) / std).to(device=device, dtype=torch.float32)
    # Resetting the seed makes model selection comparable across epochs.
    torch.manual_seed(seed)
    classifier = torch.nn.Linear(x_train.shape[1], classes, device=device)
    optimizer = torch.optim.LBFGS(
        classifier.parameters(),
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(classifier(x_train), y_train)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.inference_mode():
        return classifier(x_query).argmax(dim=1).cpu()


class SSLRepresentationMonitor:
    """Keeps only train/validation data resident for duration selection.

    The test split is intentionally never iterated in this class.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        data_root: str | Path,
        output_dir: str | Path,
        seed: int,
        device: torch.device,
    ) -> None:
        self.config = dict(config)
        self.output_dir = Path(output_dir)
        self.device = device
        self.seed = int(seed)
        loaders, _ = build_balanced_loaders(
            data_root,
            self.config["metadata_csv"],
            batch_size=int(self.config.get("batch_size", 256)),
            num_workers=int(self.config.get("num_workers", 8)),
            cache_in_ram=bool(self.config.get("cache_in_ram", True)),
        )
        self.loaders = {"train": loaders["train"], "val": loaders["val"]}
        self.history: list[dict[str, float]] = []

    def _features(self, encoder: torch.nn.Module, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        vectors: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        # This is model selection, never an optimization step.  Inference mode
        # is essential here: retaining a graph for each feature batch would
        # exhaust the GPU long before the validation pass completes.
        with torch.inference_mode():
            for images, batch_labels in self.loaders[split]:
                images = images.to(self.device, dtype=torch.float32, non_blocking=True).div_(255.0)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    tokens = encoder(images)
                vectors.append(tokens.float().mean(dim=1).cpu())
                labels.append(batch_labels.long().cpu())
        return torch.cat(vectors), torch.cat(labels)

    def evaluate(self, encoder: torch.nn.Module, epoch: int) -> dict[str, float]:
        was_training = encoder.training
        encoder.eval()
        train_features, train_labels = self._features(encoder, "train")
        val_features, val_labels = self._features(encoder, "val")
        classes = 19
        knn_predictions = weighted_knn_predictions(
            train_features,
            train_labels,
            val_features,
            classes=classes,
            k=int(self.config.get("knn_k", 20)),
            temperature=float(self.config.get("knn_temperature", 0.07)),
        )
        linear_predictions = _fixed_linear_predictions(
            train_features,
            train_labels,
            val_features,
            classes=classes,
            seed=self.seed,
            max_iter=int(self.config.get("linear_lbfgs_max_iter", 100)),
            device=self.device,
        )
        row: dict[str, float] = {"epoch": float(epoch)}
        row.update({f"knn_val_{name}": value for name, value in _classification_metrics(knn_predictions, val_labels).items()})
        row.update({f"linear_val_{name}": value for name, value in _classification_metrics(linear_predictions, val_labels).items()})
        row.update(
            feature_diagnostics(
                torch.cat((train_features, val_features)).to(self.device, non_blocking=True),
                near_constant_std_threshold=float(self.config.get("near_constant_std_threshold", 0.05)),
            )
        )
        self.history.append(row)
        write_csv(self.history, self.output_dir / "monitor_metrics.csv")
        if was_training:
            encoder.train()
        return row

    def plot(self) -> None:
        plot_history(
            self.history,
            ["knn_val_macro_f1", "linear_val_macro_f1", "knn_val_balanced_accuracy", "linear_val_balanced_accuracy"],
            self.output_dir / "representation_validation_curves.png",
            "B0 validation-only representation monitoring",
        )
        plot_history(
            self.history,
            ["feature_embedding_std", "feature_effective_rank_fraction", "feature_mean_pairwise_cosine", "feature_near_constant_fraction"],
            self.output_dir / "representation_health.png",
            "B0 representation health diagnostics",
        )


def improves_macro_f1(candidate: float, incumbent: float, minimum_delta: float) -> bool:
    """Tie-break toward earlier checkpoints, avoiding noise-driven replacements."""
    return candidate > incumbent + minimum_delta
