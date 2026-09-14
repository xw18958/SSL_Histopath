from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pannuke_ssl.models import FreshPLIPVisionEncoder, normalize_clip

class DINOHead(nn.Module):
    """DINOv3-style projection head, implemented locally for the shared backbone."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        nlayers: int = 3,
    ) -> None:
        super().__init__()
        nlayers = max(int(nlayers), 1)
        if nlayers == 1:
            layers: list[nn.Module] = [nn.Linear(in_dim, bottleneck_dim)]
        else:
            layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
            for _ in range(nlayers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2, eps=1e-12)
        return self.last_layer(x)


class KoLeoLoss(nn.Module):
    """Single-GPU Kozachenko-Leonenko entropy regularizer used by DINOv3."""

    def forward(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        with torch.autocast("cuda", enabled=False):
            x = F.normalize(x.float(), dim=-1, p=2, eps=eps)
            dots = x @ x.T
            dots.fill_diagonal_(-1.0)
            nn_index = dots.argmax(dim=1)
            distance = torch.linalg.vector_norm(x - x[nn_index], ord=2, dim=-1)
            return -torch.log(distance + eps).mean()


class MaskingGenerator:
    """Block-mask sampler matching DINOv3's iBOT masking semantics on an 8x8 grid."""

    def __init__(
        self,
        input_size: tuple[int, int],
        *,
        min_num_patches: int = 4,
        min_aspect: float = 0.3,
    ) -> None:
        self.height, self.width = (int(input_size[0]), int(input_size[1]))
        self.min_num_patches = int(min_num_patches)
        self.log_aspect = (math.log(float(min_aspect)), math.log(1.0 / float(min_aspect)))

    def _place_block(self, mask: np.ndarray, maximum: int) -> int:
        for _ in range(10):
            area = random.uniform(self.min_num_patches, max(self.min_num_patches, maximum))
            aspect = math.exp(random.uniform(*self.log_aspect))
            h = int(round(math.sqrt(area * aspect)))
            w = int(round(math.sqrt(area / aspect)))
            if not (0 < h < self.height and 0 < w < self.width):
                continue
            top = random.randint(0, self.height - h)
            left = random.randint(0, self.width - w)
            region = mask[top : top + h, left : left + w]
            available = int((~region).sum())
            if available <= 0 or available > maximum:
                continue
            before = int(mask.sum())
            mask[top : top + h, left : left + w] = True
            return int(mask.sum()) - before
        return 0

    def __call__(self, count: int) -> torch.Tensor:
        count = max(0, min(int(count), self.height * self.width))
        mask = np.zeros((self.height, self.width), dtype=bool)
        while int(mask.sum()) < count:
            remaining = count - int(mask.sum())
            if self._place_block(mask, remaining) == 0:
                break
        missing = count - int(mask.sum())
        if missing > 0:
            candidates = np.flatnonzero(~mask.reshape(-1))
            chosen = np.random.choice(candidates, size=missing, replace=False)
            flat = mask.reshape(-1)
            flat[chosen] = True
        return torch.from_numpy(mask.reshape(-1).copy())


class SharedPLIPDINOBackbone(nn.Module):
    """Fresh shared PLIP/CLIP ViT with DINOv3-compatible embedding masking."""

    def __init__(self, base: FreshPLIPVisionEncoder) -> None:
        super().__init__()
        self.base = base
        self.mask_token = nn.Parameter(torch.zeros(1, 1, int(base.hidden_size)))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # The standard downstream protocol always evaluates 256x256 images.
        return self.base(images)

    def features(
        self,
        images: torch.Tensor,
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = images.shape[-2:]
        if h % self.base.patch_size or w % self.base.patch_size:
            raise ValueError("DINOv3 crop dimensions must be divisible by the shared patch size")
        vision = self.base.model.vision_model
        interpolate = h != self.base.image_size or w != self.base.image_size
        embedded = vision.embeddings(
            normalize_clip(images),
            interpolate_pos_encoding=interpolate,
        )
        n_patches = (h // self.base.patch_size) * (w // self.base.patch_size)
        if masks is not None:
            if interpolate:
                raise ValueError("iBOT masks are only applied to global 256x256 crops")
            expected = (images.shape[0], n_patches)
            if tuple(masks.shape) != expected:
                raise ValueError(f"Mask shape {tuple(masks.shape)} != {expected}")
            embeddings = vision.embeddings
            positions = embeddings.position_embedding(embeddings.position_ids).to(embedded.dtype)
            masked_patch = self.mask_token.to(embedded.dtype) + positions[:, 1 : n_patches + 1]
            patches = torch.where(masks.unsqueeze(-1), masked_patch.expand(images.shape[0], -1, -1), embedded[:, 1:])
            embedded = torch.cat((embedded[:, :1], patches), dim=1)
        encoded = vision.encoder(
            inputs_embeds=vision.pre_layrnorm(embedded),
            return_dict=True,
        ).last_hidden_state
        normalized = vision.post_layernorm(encoded)
        cls = normalized[:, 0]
        patches = normalized[:, 1:]
        if patches.shape[1:] != (n_patches, self.base.hidden_size):
            raise RuntimeError(f"Unexpected DINOv3 patch-token shape: {tuple(patches.shape)}")
        return cls, patches


@torch.no_grad()
def sinkhorn_knopp(logits: torch.Tensor, temperature: float, iterations: int = 3) -> torch.Tensor:
    """Single-GPU Sinkhorn-Knopp teacher assignment used for DINO/iBOT targets."""
    values = logits.float() / float(temperature)
    values = values - values.max()
    q = values.exp().T
    q = q / q.sum().clamp_min(1e-12)
    prototypes, batch = q.shape
    for _ in range(int(iterations)):
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)
        q = q / prototypes
        q = q / q.sum(dim=0, keepdim=True).clamp_min(1e-12)
        q = q / batch
    return (q * batch).T


def cross_view_dino_loss(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    *,
    student_temperature: float,
    ignore_diagonal: bool,
) -> torch.Tensor:
    """Cross-entropy across student/teacher crops with optional matching-view exclusion."""
    student_log = F.log_softmax(student_logits.float() / float(student_temperature), dim=-1)
    total = student_log.new_zeros(())
    terms = 0
    for s in range(student_log.shape[0]):
        for t in range(teacher_probs.shape[0]):
            if ignore_diagonal and s == t:
                continue
            total = total - (teacher_probs[t] * student_log[s]).sum(dim=-1).mean()
            terms += 1
    if terms == 0:
        raise RuntimeError("DINO crop loss has no valid cross-view terms")
    return total / terms

