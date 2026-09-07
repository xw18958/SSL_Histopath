"""B1 residual discrete-velocity predictor.

B1 deliberately preserves B0's encoder, conditioning sequence, and predictor
capacity.  Its only functional change is parameterizing the adjacent clean-endpoint
prediction as ``Z_s + delta * v_phi(Z_s, s, a, delta)``.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .models import FreshPLIPVisionEncoder, ScalarConditioner, make_teacher


class B1VelocityPredictor(nn.Module):
    """B0-capacity conditioned predictor that returns a token velocity."""

    def __init__(
        self,
        input_dim: int = 768,
        predictor_dim: int = 384,
        depth: int = 2,
        heads: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dropout != 0.0:
            raise ValueError("B1 predictor dropout is fixed to zero")
        self.input_projection = nn.Linear(input_dim, predictor_dim)
        self.severity_conditioner = ScalarConditioner(predictor_dim)
        self.action_embedding = nn.Embedding(2, predictor_dim)
        # Retained for exact B0 input/capacity matching, even though all B1
        # adjacent transitions have the fixed delta -0.25.
        self.delta_conditioner = ScalarConditioner(predictor_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=heads,
            dim_feedforward=predictor_dim * mlp_ratio,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(predictor_dim)
        self.velocity_projection = nn.Linear(predictor_dim, input_dim)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        severity: torch.Tensor,
        action: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        if patch_tokens.ndim != 3:
            raise ValueError("B1 patch tokens must have shape [batch, tokens, channels]")
        if not (severity.shape == action.shape == delta.shape == patch_tokens.shape[:1]):
            raise ValueError("B1 conditioning tensors must have one value per example")
        visual = self.input_projection(patch_tokens)
        conditioning = torch.stack(
            (
                self.severity_conditioner(severity),
                self.action_embedding(action.long()),
                self.delta_conditioner(delta),
            ),
            dim=1,
        )
        output = self.transformer(torch.cat((conditioning, visual), dim=1))[:, 3:, :]
        return self.velocity_projection(self.final_norm(output))


def residual_endpoint(
    source_tokens: torch.Tensor,
    velocity: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """Map a discrete velocity to the adjacent target endpoint exactly once."""
    if source_tokens.shape != velocity.shape or source_tokens.ndim != 3:
        raise ValueError("source_tokens and velocity must share shape [batch, tokens, channels]")
    if delta.shape != source_tokens.shape[:1]:
        raise ValueError("delta must have shape [batch]")
    return source_tokens + delta[:, None, None].to(dtype=source_tokens.dtype) * velocity


def build_b1_models(config: dict[str, Any], device: torch.device) -> tuple[nn.Module, nn.Module, nn.Module]:
    """Construct a fresh ViT-B/32 student/EMA teacher and B1 velocity head."""
    model = config["model"]
    if int(model["image_size"]) != 256 or int(model["patch_size"]) != 32:
        raise ValueError("B1 is fixed to native 256x256 PanNuke inputs and 32x32 patches")
    student = FreshPLIPVisionEncoder(config["plip_config_dir"], image_size=256).to(device)
    if student.hidden_size != 768 or student.num_patches != 64:
        raise ValueError("The local PLIP config is not the expected fresh ViT-B/32 architecture")
    teacher = make_teacher(student).to(device)
    predictor = B1VelocityPredictor(
        input_dim=768,
        predictor_dim=int(model["predictor_dim"]),
        depth=int(model["predictor_depth"]),
        heads=int(model["predictor_heads"]),
        mlp_ratio=int(model["predictor_mlp_ratio"]),
        dropout=float(model["dropout"]),
    ).to(device)
    return student, teacher, predictor
