from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .models import FreshPLIPVisionEncoder, ScalarConditioner, make_teacher


def map_signed_change_to_unit(
    source_images: torch.Tensor,
    target_images: torch.Tensor,
) -> torch.Tensor:
    """Losslessly map target-source RGB change from [-1, 1] to [0, 1].

    B0 constructs source_images and target_images in [0, 1]. Therefore
    signed_change = target_images - source_images lies in [-1, 1], and
    (signed_change + 1) / 2 is exactly invertible by 2*x - 1.
    """
    if source_images.shape != target_images.shape:
        raise ValueError("source_images and target_images must have identical shapes")
    if not torch.is_floating_point(source_images) or not torch.is_floating_point(target_images):
        raise TypeError("change-image mapping expects floating-point tensors")

    source = source_images.float()
    target = target_images.float()
    tolerance = 1e-6
    if float(source.min()) < -tolerance or float(source.max()) > 1.0 + tolerance:
        raise ValueError("source_images must be in [0, 1]")
    if float(target.min()) < -tolerance or float(target.max()) > 1.0 + tolerance:
        raise ValueError("target_images must be in [0, 1]")

    signed_change = target - source
    mapped = (signed_change + 1.0) * 0.5
    if float(mapped.min()) < -tolerance or float(mapped.max()) > 1.0 + tolerance:
        raise RuntimeError("mapped change image escaped [0, 1]")
    return mapped.to(dtype=source_images.dtype)


def positional_grid(width: int = 384, grid_size: int = 8) -> torch.Tensor:
    """Fixed 2-D sinusoidal positions, matching the repository I-JEPA predictor."""
    if width % 4:
        raise ValueError("predictor width must be divisible by 4")
    y, x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size), indexing="ij")
    frequency = 1.0 / (10000 ** (torch.arange(width // 4).float() / (width // 4)))
    components: list[torch.Tensor] = []
    for axis in (x.flatten(), y.flatten()):
        phase = axis.float()[:, None] * frequency[None]
        components.extend((phase.sin(), phase.cos()))
    return torch.cat(components, dim=1).unsqueeze(0)


class ChangeJEPAPredictor(nn.Module):
    """JEPA-style per-patch predictor conditioned on degradation dynamics.

    Context tokens come from the full more-degraded image. A learned shared query
    is repeated once per spatial patch and augmented by patch position plus the
    B0 degradation action, source state, and delta-s conditions. Only query-token
    outputs are projected back to the 768-D teacher target space.
    """

    def __init__(
        self,
        input_dim: int = 768,
        predictor_dim: int = 384,
        depth: int = 2,
        heads: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        grid_size: int = 8,
    ) -> None:
        super().__init__()
        if dropout != 0.0:
            raise ValueError("Change-JEPA predictor dropout is fixed to zero")
        self.input_dim = int(input_dim)
        self.predictor_dim = int(predictor_dim)
        self.grid_size = int(grid_size)
        self.num_patches = self.grid_size * self.grid_size

        self.input_projection = nn.Linear(self.input_dim, self.predictor_dim)
        self.query_token = nn.Parameter(torch.zeros(1, 1, self.predictor_dim))
        nn.init.trunc_normal_(self.query_token, std=0.02)
        self.register_buffer(
            "position",
            positional_grid(self.predictor_dim, self.grid_size),
            persistent=True,
        )
        self.severity_conditioner = ScalarConditioner(self.predictor_dim)
        self.action_embedding = nn.Embedding(2, self.predictor_dim)
        self.delta_conditioner = ScalarConditioner(self.predictor_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=self.predictor_dim,
            nhead=heads,
            dim_feedforward=self.predictor_dim * mlp_ratio,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=depth,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(self.predictor_dim)
        self.output_projection = nn.Linear(self.predictor_dim, self.input_dim)

    def forward(
        self,
        context_tokens: torch.Tensor,
        source_severity: torch.Tensor,
        action: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        if context_tokens.ndim != 3:
            raise ValueError("context_tokens must have shape [B, N, D]")
        batch, patches, width = context_tokens.shape
        if patches != self.num_patches or width != self.input_dim:
            raise ValueError(
                f"Expected context shape [B, {self.num_patches}, {self.input_dim}], "
                f"got {tuple(context_tokens.shape)}"
            )
        if source_severity.shape != (batch,) or action.shape != (batch,) or delta.shape != (batch,):
            raise ValueError("source_severity, action, and delta must each have shape [B]")

        context = self.input_projection(context_tokens)
        positions = self.position.expand(batch, -1, -1).to(dtype=context.dtype)
        context = context + positions

        condition = (
            self.severity_conditioner(source_severity)
            + self.action_embedding(action.long())
            + self.delta_conditioner(delta)
        ).to(dtype=context.dtype)
        queries = self.query_token.to(dtype=context.dtype) + positions + condition[:, None, :]

        sequence = torch.cat((context, queries), dim=1)
        predicted_queries = self.transformer(sequence)[:, patches:, :]
        return self.output_projection(self.final_norm(predicted_queries))


def build_models(config: dict, device: torch.device) -> tuple[nn.Module, nn.Module, nn.Module]:
    model = config["model"]
    student = FreshPLIPVisionEncoder(
        config["plip_config_dir"],
        image_size=int(model["image_size"]),
    ).to(device)
    if student.hidden_size != int(model["hidden_size"]) or student.num_patches != 64:
        raise ValueError("Change-JEPA requires the repository PLIP ViT-B/32 8x8 patch architecture")
    teacher = make_teacher(student).to(device)
    predictor = ChangeJEPAPredictor(
        input_dim=int(model["hidden_size"]),
        predictor_dim=int(model["predictor_dim"]),
        depth=int(model["predictor_depth"]),
        heads=int(model["predictor_heads"]),
        mlp_ratio=int(model["predictor_mlp_ratio"]),
        dropout=float(model["dropout"]),
        grid_size=8,
    ).to(device)
    return student, teacher, predictor


def objective(
    student: nn.Module,
    teacher: nn.Module,
    predictor: nn.Module,
    source_images: torch.Tensor,
    target_images: torch.Tensor,
    source_severity: torch.Tensor,
    action: torch.Tensor,
    delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict the EMA representation of the mapped signed visual-change image."""
    student_tokens = student(source_images)
    prediction = predictor(student_tokens, source_severity, action, delta)

    with torch.no_grad():
        change_images = map_signed_change_to_unit(source_images, target_images)
        teacher_tokens = teacher(change_images)
        # Match the existing repository I-JEPA target normalization.
        target = F.layer_norm(teacher_tokens.float(), (teacher_tokens.shape[-1],)).detach()

    loss = F.smooth_l1_loss(prediction.float(), target.float())
    return loss, prediction, target, student_tokens, change_images
