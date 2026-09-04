from __future__ import annotations

import torch
import torch.nn.functional as F


def off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    n, m = matrix.shape
    if n != m:
        raise ValueError("Expected a square covariance matrix")
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def variance_covariance_loss(
    representations: torch.Tensor,
    *,
    variance_target: float = 1.0,
    covariance_weight: float = 0.04,
    epsilon: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = representations.float()
    if values.shape[0] < 2:
        raise ValueError("Variance/covariance loss needs at least two samples")
    centered = values - values.mean(dim=0)
    std = torch.sqrt(values.var(dim=0, unbiased=True) + epsilon)
    variance = F.relu(variance_target - std).mean()
    covariance_matrix = centered.T @ centered / (values.shape[0] - 1)
    covariance = off_diagonal(covariance_matrix).pow(2).sum() / values.shape[1]
    return variance + covariance_weight * covariance, variance, covariance


def b0_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    student_tokens: torch.Tensor,
    *,
    lambda_reg: float,
    covariance_weight: float,
    variance_target: float,
) -> dict[str, torch.Tensor]:
    prediction_loss = F.smooth_l1_loss(prediction.float(), target.float())
    regularizer, variance, covariance = variance_covariance_loss(
        student_tokens.float().mean(dim=1),
        variance_target=variance_target,
        covariance_weight=covariance_weight,
    )
    total = prediction_loss + lambda_reg * regularizer
    return {
        "total": total,
        "prediction": prediction_loss,
        "regularizer": regularizer,
        "variance": variance,
        "covariance": covariance,
        "embedding_std": student_tokens.float().mean(dim=1).std(dim=0).mean(),
    }
