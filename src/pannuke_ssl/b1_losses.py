"""Endpoint loss for B1.

The target is still the clean adjacent teacher endpoint.  B1 differs from B0
only in how that endpoint is parameterized, so the VICReg control is identical.
"""
from __future__ import annotations

import torch

from .losses import b0_loss


def b1_loss(
    predicted_target: torch.Tensor,
    teacher_target: torch.Tensor,
    student_tokens: torch.Tensor,
    *,
    lambda_reg: float,
    covariance_weight: float,
    variance_target: float,
) -> dict[str, torch.Tensor]:
    return b0_loss(
        predicted_target,
        teacher_target,
        student_tokens,
        lambda_reg=lambda_reg,
        covariance_weight=covariance_weight,
        variance_target=variance_target,
    )
