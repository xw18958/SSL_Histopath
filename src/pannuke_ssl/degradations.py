from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


DEFOCUS = 0
RESOLUTION = 1
ANCHORS = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])


@dataclass(frozen=True)
class DegradationEndpoints:
    defocus_radius: float
    resolution_factor: float


def disk_kernel(radius: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if radius <= 0:
        return torch.ones((1, 1, 1, 1), device=device, dtype=dtype)
    extent = max(1, math.ceil(radius))
    axis = torch.arange(-extent, extent + 1, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    # A soft half-pixel boundary gives sensible fractional-radius anchor kernels.
    kernel = (radius + 0.5 - torch.sqrt(xx.square() + yy.square())).clamp(0.0, 1.0)
    if not torch.any(kernel):
        kernel[extent, extent] = 1.0
    kernel = (kernel / kernel.sum()).to(dtype=dtype)
    return kernel.view(1, 1, *kernel.shape)


def defocus_blur(images: torch.Tensor, radius: float) -> torch.Tensor:
    if radius <= 0:
        return images
    kernel = disk_kernel(radius, device=images.device, dtype=images.dtype)
    padding = kernel.shape[-1] // 2
    padded = F.pad(images, (padding, padding, padding, padding), mode="reflect")
    weight = kernel.expand(images.shape[1], 1, -1, -1)
    return F.conv2d(padded, weight, groups=images.shape[1]).clamp(0.0, 1.0)


def resolution_degrade(images: torch.Tensor, factor: float) -> torch.Tensor:
    if factor <= 1.0:
        return images
    side = max(1, int(round(images.shape[-1] / factor)))
    reduced = F.interpolate(images, size=(side, side), mode="area")
    return F.interpolate(reduced, size=images.shape[-2:], mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)


def degrade_uniform(images: torch.Tensor, action: int, severity: float, endpoints: DegradationEndpoints) -> torch.Tensor:
    severity = float(min(1.0, max(0.0, severity)))
    if action == DEFOCUS:
        return defocus_blur(images, severity * endpoints.defocus_radius)
    if action == RESOLUTION:
        factor = 1.0 + severity * (endpoints.resolution_factor - 1.0)
        return resolution_degrade(images, factor)
    raise ValueError(f"Unknown degradation action: {action}")


def degrade_batch(
    images: torch.Tensor,
    actions: torch.Tensor,
    severities: torch.Tensor,
    endpoints: DegradationEndpoints,
) -> torch.Tensor:
    output = torch.empty_like(images)
    # B0 has only two actions and five anchors, so grouping avoids per-image convolutions.
    for action in (DEFOCUS, RESOLUTION):
        action_indices = torch.nonzero(actions == action, as_tuple=False).flatten()
        if action_indices.numel() == 0:
            continue
        for severity in torch.unique(severities[action_indices]).tolist():
            indices = action_indices[severities[action_indices] == severity]
            output[indices] = degrade_uniform(images[indices], action, severity, endpoints)
    return output


def sample_adjacent_transitions(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    actions = torch.randint(0, 2, (batch_size,), device=device)
    target_anchor = torch.randint(0, 4, (batch_size,), device=device)
    target = ANCHORS.to(device=device)[target_anchor]
    source = ANCHORS.to(device=device)[target_anchor + 1]
    return actions, source, target, target - source
