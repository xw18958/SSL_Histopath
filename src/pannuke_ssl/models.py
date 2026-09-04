from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn
from transformers import CLIPConfig, CLIPVisionModel


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def normalize_clip(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    return (images - mean) / std


class FreshPLIPVisionEncoder(nn.Module):
    """PLIP's vision architecture instantiated from config with fresh weights only."""

    def __init__(self, config_dir: str | Path, image_size: int = 256) -> None:
        super().__init__()
        clip_config = CLIPConfig.from_pretrained(str(config_dir), local_files_only=True)
        vision_config = copy.deepcopy(clip_config.vision_config)
        vision_config.image_size = image_size
        vision_config.dropout = 0.0
        vision_config.attention_dropout = 0.0
        # Constructing from the config is intentional: no from_pretrained/model state call.
        self.model = CLIPVisionModel(vision_config)
        self.image_size = image_size
        self.patch_size = int(vision_config.patch_size)
        self.hidden_size = int(vision_config.hidden_size)
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.num_patches = (self.image_size // self.patch_size) ** 2

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(pixel_values=normalize_clip(images), return_dict=True)
        patches = output.last_hidden_state[:, 1:, :]
        # CLIP post-normalizes only CLS; use the same learned LN for patch targets/features.
        patches = self.model.vision_model.post_layernorm(patches)
        if patches.shape[1:] != (self.num_patches, self.hidden_size):
            raise RuntimeError(f"Unexpected patch tensor shape: {tuple(patches.shape)}")
        return patches


class ScalarConditioner(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value.reshape(-1, 1))


class B0Predictor(nn.Module):
    """Conditioned patch predictor with no positional encoding."""

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
            raise ValueError("B0 predictor dropout is fixed to zero")
        self.input_projection = nn.Linear(input_dim, predictor_dim)
        self.severity_conditioner = ScalarConditioner(predictor_dim)
        self.action_embedding = nn.Embedding(2, predictor_dim)
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
        self.output_projection = nn.Linear(predictor_dim, input_dim)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        severity: torch.Tensor,
        action: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
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
        return self.output_projection(self.final_norm(output))


@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    for student_value, teacher_value in zip(student.parameters(), teacher.parameters(), strict=True):
        teacher_value.mul_(momentum).add_(student_value, alpha=1.0 - momentum)
    for student_value, teacher_value in zip(student.buffers(), teacher.buffers(), strict=True):
        teacher_value.copy_(student_value)


def make_teacher(student: nn.Module) -> nn.Module:
    teacher = copy.deepcopy(student)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher
