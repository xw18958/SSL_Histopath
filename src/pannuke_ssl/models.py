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

    @property
    def vision_backbone(self) -> nn.Module:
        """Return CLIP's vision transformer across supported Transformers layouts."""
        nested_vision = getattr(self.model, "vision_model", None)
        vision = nested_vision if nested_vision is not None else self.model
        required_components = ("embeddings", "pre_layrnorm", "encoder", "post_layernorm")
        missing = [name for name in required_components if not hasattr(vision, name)]
        if missing:
            raise RuntimeError(
                "CLIP vision backbone is missing required components "
                f"{missing}; expected a CLIPVisionModel or its vision_model submodule."
            )
        return vision

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(pixel_values=normalize_clip(images), return_dict=True)
        patches = output.last_hidden_state[:, 1:, :]
        # CLIP post-normalizes only CLS; use the same learned LN for patch targets/features.
        patches = self.vision_backbone.post_layernorm(patches)
        if patches.shape[1:] != (self.num_patches, self.hidden_size):
            raise RuntimeError(f"Unexpected patch tensor shape: {tuple(patches.shape)}")
        return patches


class PretrainedPLIPVisionEncoder(nn.Module):
    """Frozen PLIP vision encoder with an explicit 256px evaluation readout.

    PLIP was pretrained at 224px.  This adapter deliberately retains those
    weights and uses CLIP's built-in bicubic positional interpolation when
    evaluating the shared 256px PanNuke protocol.  ``patch_mean`` is the
    fairness-critical final-patch-token readout; ``cls`` retains PLIP's native
    post-layernorm CLS representation as a supplementary baseline.
    """

    READOUTS = frozenset(("patch_mean", "cls"))

    def __init__(
        self,
        model_dir: str | Path,
        *,
        image_size: int = 256,
        readout: str = "patch_mean",
    ) -> None:
        super().__init__()
        if readout not in self.READOUTS:
            raise ValueError(f"Unsupported PLIP readout {readout!r}; expected one of {sorted(self.READOUTS)}")
        self.model = CLIPVisionModel.from_pretrained(str(model_dir), local_files_only=True)
        self.source_image_size = int(self.model.config.image_size)
        self.image_size = int(image_size)
        self.patch_size = int(self.model.config.patch_size)
        self.hidden_size = int(self.model.config.hidden_size)
        self.readout = readout
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by the pretrained CLIP patch size")
        self.num_patches = (self.image_size // self.patch_size) ** 2
        if (self.source_image_size, self.patch_size, self.hidden_size) != (224, 32, 768):
            raise RuntimeError(
                "Unexpected PLIP vision configuration; expected pretrained 224px / patch-32 / 768-dimensional vision encoder"
            )
        self.model.requires_grad_(False)
        self.model.eval()

    @property
    def vision_backbone(self) -> nn.Module:
        nested_vision = getattr(self.model, "vision_model", None)
        vision = nested_vision if nested_vision is not None else self.model
        required_components = ("embeddings", "pre_layrnorm", "encoder", "post_layernorm")
        missing = [name for name in required_components if not hasattr(vision, name)]
        if missing:
            raise RuntimeError(
                "CLIP vision backbone is missing required components "
                f"{missing}; expected a CLIPVisionModel or its vision_model submodule."
            )
        return vision

    def train(self, mode: bool = True) -> "PretrainedPLIPVisionEncoder":
        """Keep the baseline in evaluation mode even if a caller invokes train()."""
        super().train(False)
        return self

    @torch.inference_mode()
    def _sequence(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected [B,3,H,W] image batch, got {tuple(images.shape)}")
        h, w = images.shape[-2:]
        if (h, w) != (self.image_size, self.image_size):
            raise ValueError(f"Expected fixed {self.image_size}px evaluation images, got {h}x{w}")
        output = self.model(
            pixel_values=normalize_clip(images),
            interpolate_pos_encoding=(h, w) != (self.source_image_size, self.source_image_size),
            return_dict=True,
        )
        sequence = output.last_hidden_state
        expected = (images.shape[0], self.num_patches + 1, self.hidden_size)
        if sequence.shape != expected:
            raise RuntimeError(f"Unexpected PLIP sequence shape {tuple(sequence.shape)} != {expected}")
        return sequence, self.vision_backbone.post_layernorm(sequence[:, 0, :])

    @torch.inference_mode()
    def patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Return post-layernorm final patch tokens, [B,64,768] at 256px."""
        sequence, _ = self._sequence(images)
        patches = self.vision_backbone.post_layernorm(sequence[:, 1:, :])
        expected = (images.shape[0], self.num_patches, self.hidden_size)
        if patches.shape != expected:
            raise RuntimeError(f"Unexpected PLIP patch shape {tuple(patches.shape)} != {expected}")
        return patches

    @torch.inference_mode()
    def cls_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return PLIP's native post-layernorm CLS representation, [B,768]."""
        _, cls = self._sequence(images)
        return cls

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.readout == "patch_mean":
            return self.patch_tokens(images).mean(dim=1)
        return self.cls_features(images)


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
