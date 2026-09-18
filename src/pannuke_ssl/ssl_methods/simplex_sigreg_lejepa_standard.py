from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .lejepa_standard import StandardLeJEPA


def regular_simplex_centers(
    num_components: int,
    feature_dim: int,
    sigma: float,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Construct deterministic regular-simplex centers with pairwise distance 2*sigma."""
    k = int(num_components)
    d = int(feature_dim)
    s = float(sigma)
    if k < 1:
        raise ValueError("num_components must be >= 1")
    if d < 1:
        raise ValueError("feature_dim must be >= 1")
    if s <= 0:
        raise ValueError("sigma must be > 0")
    if k - 1 > d:
        raise ValueError(f"A regular simplex with K={k} requires feature_dim >= {k - 1}, got {d}")

    centers = torch.zeros((k, d), dtype=dtype)
    if k == 1:
        return centers

    # Deterministic Helmert basis: K rows in R^(K-1), zero centroid, and
    # row Gram matrix I - 11^T/K. Scaling by sqrt(2)*sigma makes every
    # pairwise center distance exactly 2*sigma.
    basis = torch.zeros((k, k - 1), dtype=torch.float64)
    for j in range(1, k):
        scale = math.sqrt(j * (j + 1))
        basis[:j, j - 1] = 1.0 / scale
        basis[j, j - 1] = -float(j) / scale
    centers[:, : k - 1] = (math.sqrt(2.0) * s * basis).to(dtype=dtype)
    return centers


class SimplexEppsPulley(nn.Module):
    """Epps-Pulley characteristic-function loss for a fixed simplex isotropic GMM."""

    def __init__(self, sigma: float = 1.0, t_max: float = 3.0, n_points: int = 17) -> None:
        super().__init__()
        self.sigma = float(sigma)
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0")
        t = torch.linspace(0, t_max, n_points)
        dt = t_max / (n_points - 1)

        # Preserve LeJEPA's original positive Epps-Pulley integration weighting.
        integration_phi = (-0.5 * t**2).exp()
        weights = torch.full((n_points,), 2 * dt)
        weights[[0, -1]] = dt
        self.register_buffer("t", t)
        self.register_buffer("weights", weights * integration_phi)

        # The GMM component covariance controls the target characteristic function.
        self.register_buffer("target_decay", (-0.5 * self.sigma**2 * t**2).exp())

    def forward(self, x: torch.Tensor, projected_means: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or projected_means.ndim != 2:
            raise ValueError("Expected projected samples [N,S] and projected means [K,S]")
        if x.size(1) != projected_means.size(1):
            raise ValueError("Sample and simplex projections must use the same slices")

        n = x.size(0)
        x_t = x.unsqueeze(-1) * self.t
        empirical_real = x_t.cos().mean(0)
        empirical_imag = x_t.sin().mean(0)

        phase = projected_means.unsqueeze(-1) * self.t
        target_real = phase.cos().mean(0) * self.target_decay
        target_imag = phase.sin().mean(0) * self.target_decay
        err = (empirical_real - target_real).square() + (empirical_imag - target_imag).square()
        return (err @ self.weights) * n


class SlicedSimplexEppsPulley(nn.Module):
    """LeJEPA SIGReg slicing with only the target changed to a fixed simplex GMM."""

    def __init__(
        self,
        feature_dim: int,
        num_components: int,
        sigma: float = 1.0,
        num_slices: int = 1024,
        t_max: float = 3.0,
        n_points: int = 17,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_components = int(num_components)
        self.sigma = float(sigma)
        self.num_slices = int(num_slices)
        self.ep = SimplexEppsPulley(sigma=self.sigma, t_max=t_max, n_points=n_points)
        self.register_buffer(
            "centers",
            regular_simplex_centers(self.num_components, self.feature_dim, self.sigma),
        )
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("LeJEPA SIGReg expects [N,D]")
        if x.size(-1) != self.feature_dim:
            raise ValueError(f"Expected feature dimension {self.feature_dim}, got {x.size(-1)}")
        with torch.no_grad():
            step = int(self.global_step.item())
            generator = torch.Generator(device=x.device).manual_seed(step)
            directions = torch.randn(
                x.size(-1), self.num_slices, device=x.device, generator=generator
            )
            directions = directions / directions.norm(p=2, dim=0)
            projected_means = self.centers @ directions
            self.global_step.add_(1)
        return self.ep(x @ directions, projected_means).mean()


class StandardSimplexSIGRegLeJEPA(StandardLeJEPA):
    """Standard LeJEPA with only SIGReg's target changed to a regular-simplex GMM."""

    def __init__(self, c: dict[str, Any], device: torch.device) -> None:
        super().__init__(c, device)
        method = c["method"]
        objective = method["objective"]
        k = int(objective["simplex_components"])
        sigma = float(objective["simplex_sigma"])
        projector_dim = int(method["projector"]["dim"])

        # Spacing is derived, never tuned independently.
        distance = 2.0 * sigma
        separation_ratio = 2.0
        simplex_scale_c = 2.0 * sigma**2 * (k - 1) / k
        objective["derived_center_distance"] = distance
        objective["derived_separation_ratio"] = separation_ratio
        objective["derived_simplex_scale_C"] = simplex_scale_c

        self.model.sigreg = SlicedSimplexEppsPulley(
            feature_dim=projector_dim,
            num_components=k,
            sigma=sigma,
            num_slices=int(objective["sigreg_slices"]),
            t_max=float(objective["sigreg_t_max"]),
            n_points=int(objective["sigreg_points"]),
        ).to(device)
