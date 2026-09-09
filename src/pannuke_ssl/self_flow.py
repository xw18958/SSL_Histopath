"""Capacity-matched pixel-space Self-Flow for the PanNuke SSL benchmark.

This is intentionally a *pixel-space adaptation* of Self-Flow, not the released
latent ImageNet model. It preserves the defining Self-Flow mechanisms:
per-token timestep conditioning, Dual-Timestep Scheduling, rectified-flow
velocity prediction, EMA-teacher feature reconstruction, and the two-layer
Self-Flow projector. No class labels or pretrained tokenizer/VAE are used.
"""
from __future__ import annotations

import copy
import math
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def normalize_flow_pixels(images: torch.Tensor) -> torch.Tensor:
    """Map benchmark RGB inputs from [0, 1] to the symmetric flow domain [-1, 1]."""
    return images.mul(2.0).sub(1.0)


def patchify(images: torch.Tensor, patch_size: int = 32) -> torch.Tensor:
    """Convert [B,C,H,W] images to non-overlapping [B,N,C*P*P] patch vectors."""
    if images.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(images.shape)}")
    if images.shape[-2] % patch_size or images.shape[-1] % patch_size:
        raise ValueError("Image dimensions must be divisible by patch_size")
    patches = F.unfold(images, kernel_size=patch_size, stride=patch_size)
    return patches.transpose(1, 2).contiguous()


def _sincos_1d(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    if embed_dim % 2:
        raise ValueError("1D sinusoidal embedding dimension must be even")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)
    out = np.einsum("m,d->md", positions.reshape(-1), omega)
    return np.concatenate((np.sin(out), np.cos(out)), axis=1)


def fixed_2d_sincos(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Self-Flow/DiT-style fixed 2D sinusoidal position embedding."""
    if embed_dim % 4:
        raise ValueError("2D sinusoidal embedding dimension must be divisible by 4")
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, -1)
    embedding = np.concatenate(
        (_sincos_1d(embed_dim // 2, grid[0]), _sincos_1d(embed_dim // 2, grid[1])),
        axis=1,
    )
    return torch.from_numpy(embedding).float().unsqueeze(0)


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding followed by the Self-Flow two-layer MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float().reshape(-1, 1) * frequencies.reshape(1, -1)
        embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
        if dim % 2:
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
        return embedding

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        original_shape = timesteps.shape
        flat = timesteps.reshape(-1)
        embedded = self.mlp(self.timestep_embedding(flat, self.frequency_embedding_size))
        return embedded.reshape(*original_shape, embedded.shape[-1])


class SelfFlowAttention(nn.Module):
    """Bias-QKV multi-head self-attention matching the released DiT block semantics."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, hidden = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (value.transpose(1, 2) for value in (q, k, v))
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attended = attended.transpose(1, 2).reshape(batch, tokens, hidden)
        return self.proj(attended)


class SelfFlowMLP(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float) -> None:
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


def _modulate_per_token(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


class PerTokenDiTBlock(nn.Module):
    """adaLN-Zero DiT block with independent timestep conditioning per token."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfFlowAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = SelfFlowMLP(hidden_size, mlp_ratio)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        modulation = self.adaLN_modulation(conditioning)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        x = x + gate_msa * self.attn(_modulate_per_token(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(_modulate_per_token(self.norm2(x), shift_mlp, scale_mlp))
        return x


class PerTokenFlowHead(nn.Module):
    """Per-token adaLN flow head that predicts one velocity vector per raw pixel patch."""

    def __init__(self, hidden_size: int, patch_dim: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.linear = nn.Linear(hidden_size, patch_dim)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(conditioning).chunk(2, dim=-1)
        return self.linear(_modulate_per_token(self.norm_final(x), shift, scale))


class SimpleHead(nn.Module):
    """Two-layer Self-Flow self-distillation projector."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, in_dim + out_dim)
        self.linear2 = nn.Linear(in_dim + out_dim, out_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.act(self.linear1(x)))


class PixelSelfFlowB32(nn.Module):
    """Parameter-matched Self-Flow adaptation for 256x256 PanNuke RGB patches.

    Eight 768-wide blocks are used because per-token adaLN makes each DiT block
    substantially heavier than a standard ViT-B block. This keeps the total
    trainable parameter count close to B0/I-JEPA (student + predictor), while
    preserving 64 tokens and 768-dimensional downstream representations.
    """

    def __init__(
        self,
        *,
        image_size: int = 256,
        patch_size: int = 32,
        hidden_size: int = 768,
        depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        student_rep_layer: int = 2,
        teacher_rep_layer: int = 6,
    ) -> None:
        super().__init__()
        if image_size != 256 or patch_size != 32:
            raise ValueError("Self-Flow benchmark is fixed to 256x256 images and 32x32 patches")
        if depth != 8 or hidden_size != 768 or num_heads != 12:
            raise ValueError("Capacity-matched Self-Flow is fixed to depth=8, hidden=768, heads=12")
        if not (1 <= student_rep_layer <= depth and 1 <= teacher_rep_layer <= depth):
            raise ValueError("Representation layers must lie inside the transformer")
        if student_rep_layer >= teacher_rep_layer:
            raise ValueError("Student representation layer must precede teacher representation layer")

        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.hidden_size = int(hidden_size)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.student_rep_layer = int(student_rep_layer)
        self.teacher_rep_layer = int(teacher_rep_layer)
        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.grid_size**2
        self.patch_dim = 3 * self.patch_size * self.patch_size

        self.x_embedder = nn.Linear(self.patch_dim, self.hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.register_buffer(
            "pos_embed",
            fixed_2d_sincos(self.hidden_size, self.grid_size),
            persistent=True,
        )
        self.blocks = nn.ModuleList(
            [PerTokenDiTBlock(self.hidden_size, self.num_heads, self.mlp_ratio) for _ in range(self.depth)]
        )
        self.final_layer = PerTokenFlowHead(self.hidden_size, self.patch_dim)
        self.projector = SimpleHead(self.hidden_size, self.hidden_size)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(basic_init)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def _encode_patches(
        self,
        patches: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        return_layers: Iterable[int] = (),
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor], torch.Tensor]:
        if patches.shape[1:] != (self.num_patches, self.patch_dim):
            raise ValueError(f"Expected patches [B,{self.num_patches},{self.patch_dim}], got {tuple(patches.shape)}")
        batch = patches.shape[0]
        if timesteps.ndim == 1:
            timesteps = timesteps[:, None].expand(batch, self.num_patches)
        if timesteps.shape != (batch, self.num_patches):
            raise ValueError(f"Expected timesteps [B,{self.num_patches}], got {tuple(timesteps.shape)}")
        x = self.x_embedder(patches)
        x = x + self.pos_embed.to(device=x.device, dtype=x.dtype)
        conditioning = self.t_embedder(timesteps)
        wanted = set(int(layer) for layer in return_layers)
        features: dict[int, torch.Tensor] = {}
        for layer_index, block in enumerate(self.blocks, start=1):
            x = block(x, conditioning)
            if layer_index in wanted:
                features[layer_index] = x
        return x, features, conditioning

    def flow_and_features(
        self,
        patches: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        feature_layer: int,
        project_features: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        final_tokens, features, conditioning = self._encode_patches(
            patches,
            timesteps,
            return_layers=(feature_layer,),
        )
        representation = features[feature_layer]
        if project_features:
            representation = self.projector(representation)
        velocity = self.final_layer(final_tokens, conditioning)
        return velocity, representation

    def raw_features_from_patches(
        self,
        patches: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        layer: int,
    ) -> torch.Tensor:
        _, features, _ = self._encode_patches(patches, timesteps, return_layers=(layer,))
        return features[layer]

    def clean_features(self, images: torch.Tensor, *, layer: int | None = None) -> torch.Tensor:
        """Raw hidden tokens for clean images at flow timestep t=0."""
        layer = self.depth if layer is None else int(layer)
        x0 = patchify(normalize_flow_pixels(images), self.patch_size)
        timesteps = torch.zeros((images.shape[0], self.num_patches), device=images.device, dtype=images.dtype)
        return self.raw_features_from_patches(x0, timesteps, layer=layer)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Shared-monitor interface: final clean hidden tokens [B,64,768]."""
        return self.clean_features(images, layer=self.depth)


def make_self_flow_teacher(student: PixelSelfFlowB32) -> PixelSelfFlowB32:
    teacher = copy.deepcopy(student)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


def sample_dual_timesteps(
    batch_size: int,
    num_patches: int,
    *,
    mask_ratio: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample Self-Flow Dual-Timestep Scheduling with an exact per-image mask count."""
    if not 0.0 < mask_ratio <= 0.5:
        raise ValueError("Self-Flow mask_ratio must be in (0, 0.5]")
    masked = int(round(num_patches * mask_ratio))
    if masked <= 0 or masked >= num_patches:
        raise ValueError("mask_ratio yields an invalid number of masked tokens")
    t = torch.rand(batch_size, device=device, dtype=torch.float32)
    s = torch.rand(batch_size, device=device, dtype=torch.float32)
    scores = torch.rand((batch_size, num_patches), device=device)
    indices = scores.topk(masked, dim=1, largest=False, sorted=False).indices
    mask = torch.zeros((batch_size, num_patches), device=device, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    student_tau = torch.where(mask, s[:, None], t[:, None])
    teacher_scalar = torch.minimum(t, s)
    teacher_tau = teacher_scalar[:, None].expand(-1, num_patches)
    return t, s, mask, student_tau, teacher_tau


def self_flow_objective(
    student: PixelSelfFlowB32,
    teacher: PixelSelfFlowB32,
    images: torch.Tensor,
    *,
    mask_ratio: float,
    representation_weight: float,
    student_rep_layer: int,
    teacher_rep_layer: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Rectified-flow velocity loss plus EMA teacher feature reconstruction."""
    if representation_weight < 0:
        raise ValueError("representation_weight must be non-negative")
    x0 = patchify(normalize_flow_pixels(images), student.patch_size)
    noise = torch.randn_like(x0)
    t, s, mask, student_tau, teacher_tau = sample_dual_timesteps(
        images.shape[0],
        student.num_patches,
        mask_ratio=mask_ratio,
        device=images.device,
    )
    student_mix = (1.0 - student_tau[..., None]) * x0 + student_tau[..., None] * noise
    teacher_mix = (1.0 - teacher_tau[..., None]) * x0 + teacher_tau[..., None] * noise
    velocity_target = noise - x0

    velocity_prediction, student_features = student.flow_and_features(
        student_mix,
        student_tau.to(dtype=student_mix.dtype),
        feature_layer=student_rep_layer,
        project_features=True,
    )
    with torch.no_grad():
        teacher_features = teacher.raw_features_from_patches(
            teacher_mix,
            teacher_tau.to(dtype=teacher_mix.dtype),
            layer=teacher_rep_layer,
        )

    flow_loss = F.mse_loss(velocity_prediction.float(), velocity_target.float())
    cosine = F.cosine_similarity(student_features.float(), teacher_features.float(), dim=-1)
    representation_loss = 1.0 - cosine.mean()
    total = flow_loss + float(representation_weight) * representation_loss
    harder = student_tau > teacher_tau
    metrics = {
        "total": total,
        "flow": flow_loss,
        "representation": representation_loss,
        "mean_t": t.mean(),
        "mean_s": s.mean(),
        "mean_student_tau": student_tau.mean(),
        "mean_teacher_tau": teacher_tau.mean(),
        "harder_token_fraction": harder.float().mean(),
        "mask_fraction": mask.float().mean(),
        "velocity_target_rms": velocity_target.float().square().mean().sqrt(),
    }
    return total, metrics
