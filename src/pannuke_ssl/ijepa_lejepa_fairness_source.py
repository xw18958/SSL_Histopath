"""Source-faithful compute diagnostic for I-JEPA vs LeJEPA.

Source hierarchy used here:
- I-JEPA: facebookresearch/ijepa official training, mask, predictor and transform code.
- LeJEPA: authors' released stable-pretraining LeJEPA module and 2G+6L benchmark recipe.

Only adaptations required by this experiment are retained: the same fresh PLIP/CLIP
image encoder is used by both methods; image size is 256; LeJEPA local views are
96x96 so the fixed patch-32 backbone has a 3x3 local grid; and the I-JEPA predictor
depth is 4 as selected for this smaller fixed encoder. Existing project pipelines
are imported read-only and are never modified.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import MLP
from torchvision.transforms import v2

from .data import PanNukeImageDataset, all_source_rows
from .ijepa import gather_tokens, objective, positional_grid
from .ijepa_training import protected_manifest
from .ijepa_lejepa_fairness import FlexiblePLIPVisionEncoder
from .models import FreshPLIPVisionEncoder, make_teacher, update_ema
from .parquet import build_source_index, preload_images
from .utils import atomic_json_dump, seed_everything, write_csv


SETTINGS = ("ijepa_4layer", "lejepa_2g4l", "lejepa_2g6l")


def validate_config(c: dict) -> None:
    assert c["seed"] == 20260903
    b = c["benchmark"]
    assert b["batch_size"] == 128
    assert b["warmup_steps"] == 20 and b["measured_steps"] == 100
    assert b["bf16"] and b["cache_in_ram"]

    i = c["ijepa"]
    assert i["image_size"] == 256 and i["patch_size"] == 32 and i["hidden_size"] == 768
    assert i["crop_scale"] == [0.3, 1.0]
    assert i["predictor_dim"] == 384 and i["predictor_depth"] == 4
    assert i["predictor_heads"] == 12 and i["predictor_mlp_ratio"] == 4
    assert i["target_count"] == 4
    assert i["context_scale"] == [0.85, 1.0]
    assert i["target_scale"] == [0.15, 0.20]
    assert i["target_aspect"] == [0.75, 1.5]
    assert i["minimum_context_tokens"] == 10
    assert i["target_target_overlap"] is True
    assert i["context_target_overlap"] is False
    assert float(i["ema_momentum"]) == 0.996

    l = c["lejepa"]
    assert l["global_views"] == 2 and l["local_views"] == [4, 6]
    assert l["global_size"] == 256 and l["local_size"] == 96
    assert l["global_scale"] == [0.3, 1.0] and l["local_scale"] == [0.05, 0.3]
    assert float(l["lambda"]) == 0.02
    assert l["projector_dim"] == 512 and l["projector_hidden_dim"] == 2048
    assert l["sigreg_slices"] == 1024 and l["sigreg_points"] == 17
    assert float(l["sigreg_t_max"]) == 3.0


def _repo_root(config: dict) -> Path:
    return Path(config["output_dir"]).resolve().parents[1]


def _setting_dir(config: dict, setting: str) -> Path:
    return Path(config["output_dir"]) / setting


def _encoder_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        if value.dtype == torch.bfloat16:
            digest.update(value.view(torch.uint16).numpy().tobytes())
        else:
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _count_trainable_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _seed_worker(_worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_ijepa_transform(config: dict) -> nn.Module:
    """Official I-JEPA image recipe, adapted only from 224 to our fixed 256 input."""
    i = config["ijepa"]
    return v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop(
                (int(i["image_size"]), int(i["image_size"])),
                scale=tuple(float(x) for x in i["crop_scale"]),
                antialias=True,
            ),
        ]
    )


def _lejepa_photometric(config: dict) -> list[nn.Module]:
    l = config["lejepa"]
    return [
        v2.RandomHorizontalFlip(p=float(l["horizontal_flip_p"])),
        v2.RandomApply(
            [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
            p=float(l["color_jitter_p"]),
        ),
        v2.RandomGrayscale(p=float(l["grayscale_p"])),
        v2.RandomApply(
            [v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))],
            p=float(l["gaussian_blur_p"]),
        ),
        v2.RandomSolarize(threshold=128, p=float(l["solarize_p"])),
    ]


def make_lejepa_transforms(config: dict) -> tuple[nn.Module, nn.Module]:
    l = config["lejepa"]
    global_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop(
                (int(l["global_size"]), int(l["global_size"])),
                scale=tuple(float(x) for x in l["global_scale"]),
                antialias=True,
            ),
            *_lejepa_photometric(config),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    local_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop(
                (int(l["local_size"]), int(l["local_size"])),
                scale=tuple(float(x) for x in l["local_scale"]),
                antialias=True,
            ),
            *_lejepa_photometric(config),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    return global_transform, local_transform


class FairnessDataset(Dataset):
    def __init__(self, base: Dataset, setting: str, config: dict) -> None:
        self.base = base
        self.setting = setting
        self.ijepa_transform = make_ijepa_transform(config) if setting == "ijepa_4layer" else None
        self.global_transform = None
        self.local_transform = None
        self.n_local = 0
        if setting.startswith("lejepa"):
            self.global_transform, self.local_transform = make_lejepa_transforms(config)
            self.n_local = 4 if setting.endswith("2g4l") else 6

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, fold, sample_index = self.base[index]
        if image.dtype != torch.uint8 or tuple(image.shape) != (3, 256, 256):
            raise RuntimeError(f"Unexpected PanNuke image: {image.dtype}, {tuple(image.shape)}")
        if self.setting == "ijepa_4layer":
            assert self.ijepa_transform is not None
            return {
                "image": self.ijepa_transform(image),
                "fold": int(fold),
                "sample_index": int(sample_index),
            }
        assert self.global_transform is not None and self.local_transform is not None
        return {
            "global_views": torch.stack([self.global_transform(image) for _ in range(2)]),
            "local_views": torch.stack([self.local_transform(image) for _ in range(self.n_local)]),
            "fold": int(fold),
            "sample_index": int(sample_index),
        }


def build_fairness_loader(config: dict, setting: str) -> DataLoader:
    source_index = build_source_index(Path(config["data_root"]))
    rows = all_source_rows(source_index)
    if len(rows) != 7901 or len({(int(r["fold"]), int(r["sample_index"])) for r in rows}) != 7901:
        raise ValueError("Fairness benchmark requires exactly 7,901 unique PanNuke sources.")
    cache = preload_images(rows, source_index) if config["benchmark"]["cache_in_ram"] else None
    base = PanNukeImageDataset(rows, source_index, cache, include_label=False, include_key=True)
    dataset = FairnessDataset(base, setting, config)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    workers = int(config["benchmark"]["num_workers"])
    kwargs = {
        "batch_size": int(config["benchmark"]["batch_size"]),
        "shuffle": True,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
        "drop_last": True,
        "generator": generator,
        "worker_init_fn": _seed_worker,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def _infinite(loader: DataLoader) -> Iterator[dict]:
    while True:
        yield from loader


def _update_source_hash(digest, batch: dict) -> None:
    for fold, index in zip(batch["fold"].tolist(), batch["sample_index"].tolist(), strict=True):
        digest.update(f"{int(fold)}:{int(index)};".encode("ascii"))
    digest.update(b"|")


class IJEPAOfficialMaskSampler:
    """Official multi-block mask semantics adapted to the fixed 8x8 patch grid.

    Predictor blocks are sampled independently and may overlap each other.
    The context region is constrained to avoid all predictor blocks. The original
    min_keep=10 is retained for context; a smaller block-level minimum is necessary
    on an 8x8 grid because a 15%-20% target contains only about 9-12 patches.
    """

    def __init__(self, config: dict, seed: int) -> None:
        i = config["ijepa"]
        self.height = self.width = 8
        self.enc_scale = tuple(float(x) for x in i["context_scale"])
        self.pred_scale = tuple(float(x) for x in i["target_scale"])
        self.aspect = tuple(float(x) for x in i["target_aspect"])
        self.npred = int(i["target_count"])
        self.context_min = int(i["minimum_context_tokens"])
        self.block_min = int(i["minimum_block_tokens"])
        self.generator = torch.Generator().manual_seed(int(seed))

    def _block_size(self, scale: tuple[float, float], aspect: tuple[float, float]) -> tuple[int, int]:
        u = torch.rand(1, generator=self.generator).item()
        mask_scale = scale[0] + u * (scale[1] - scale[0])
        max_keep = int(self.height * self.width * mask_scale)
        ar = aspect[0] + u * (aspect[1] - aspect[0])
        h = int(round(math.sqrt(max_keep * ar)))
        w = int(round(math.sqrt(max_keep / ar)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1
        return max(1, h), max(1, w)

    def _sample_mask(
        self,
        size: tuple[int, int],
        acceptable: list[torch.Tensor] | None,
        min_keep: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = size
        for _ in range(1000):
            top = int(torch.randint(0, max(1, self.height - h), (1,), generator=self.generator))
            left = int(torch.randint(0, max(1, self.width - w), (1,), generator=self.generator))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top : top + h, left : left + w] = 1
            if acceptable is not None:
                for region in acceptable:
                    mask *= region
            indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
            if len(indices) > min_keep:
                complement = torch.ones((self.height, self.width), dtype=torch.int32)
                complement[top : top + h, left : left + w] = 0
                return indices, complement
        raise RuntimeError("Could not sample source-faithful I-JEPA mask on 8x8 grid")

    def sample(self, batch_size: int, device: torch.device | None = None):
        p_size = self._block_size(self.pred_scale, self.aspect)
        e_size = self._block_size(self.enc_scale, (1.0, 1.0))
        all_context, all_targets = [], []
        min_context, min_target = 64, 64
        for _ in range(batch_size):
            targets, complements = [], []
            for _ in range(self.npred):
                mask, complement = self._sample_mask(p_size, None, self.block_min)
                targets.append(mask)
                complements.append(complement)
                min_target = min(min_target, len(mask))
            context, _ = self._sample_mask(e_size, complements, self.context_min)
            min_context = min(min_context, len(context))
            all_context.append(context)
            all_targets.append(targets)
        contexts = torch.stack([x[:min_context] for x in all_context]).to(device=device, dtype=torch.long)
        targets = torch.stack(
            [torch.stack([x[:min_target] for x in blocks]) for blocks in all_targets]
        ).to(device=device, dtype=torch.long)
        return contexts, targets


def _init_predictor_module(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


class IJEPAFairPredictor(nn.Module):
    """Official predictor layout with experiment-specific depth=4 and shared-encoder heads."""

    def __init__(self, num_heads: int = 12) -> None:
        super().__init__()
        self.input_projection = nn.Linear(768, 384)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.register_buffer("position", positional_grid(384))
        layer = nn.TransformerEncoderLayer(
            d_model=384,
            nhead=int(num_heads),
            dim_feedforward=1536,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, 4, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(384)
        self.output_projection = nn.Linear(384, 768)
        self.apply(_init_predictor_module)
        for layer_id, block in enumerate(self.transformer.layers, start=1):
            nn.init.trunc_normal_(block.self_attn.in_proj_weight, std=0.02)
            if block.self_attn.in_proj_bias is not None:
                nn.init.zeros_(block.self_attn.in_proj_bias)
            block.self_attn.out_proj.weight.data.div_(math.sqrt(2.0 * layer_id))
            block.linear2.weight.data.div_(math.sqrt(2.0 * layer_id))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, tokens, context_indices, target_indices):
        b, blocks, target_length = target_indices.shape
        positions = self.position.expand(b, -1, -1)
        context = self.input_projection(tokens) + gather_tokens(positions, context_indices)
        context = context[:, None].expand(-1, blocks, -1, -1).reshape(
            b * blocks, tokens.shape[1], 384
        )
        target_pos = gather_tokens(positions, target_indices.flatten(1)).reshape(
            b * blocks, target_length, 384
        )
        queries = self.mask_token + target_pos
        predicted = self.transformer(torch.cat((context, queries), dim=1))[:, -target_length:]
        return self.output_projection(self.final_norm(predicted)).reshape(
            b, blocks, target_length, 768
        )


def build_ijepa_fair(config: dict, device: torch.device):
    student = FreshPLIPVisionEncoder(config["plip_config_dir"], 256).to(device)
    if student.num_patches != 64 or student.hidden_size != 768 or student.patch_size != 32:
        raise RuntimeError("Unexpected shared PLIP/CLIP backbone")
    encoder_heads = int(student.model.config.num_attention_heads)
    expected_heads = int(config["ijepa"]["predictor_heads"])
    if encoder_heads != expected_heads:
        raise RuntimeError(f"Expected shared encoder to have {expected_heads} heads, got {encoder_heads}")
    teacher = make_teacher(student)
    predictor = IJEPAFairPredictor(num_heads=encoder_heads).to(device)
    return student, teacher, predictor


def _ijepa_optimizer(student: nn.Module, predictor: nn.Module, config: dict):
    i = config["ijepa"]
    decay, no_decay = [], []
    for module in (student, predictor):
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            if "bias" in name or parameter.ndim == 1:
                no_decay.append(parameter)
            else:
                decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(i["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(i["learning_rate"]),
    )


class EppsPulley(nn.Module):
    """Matches the released stable-pretraining LeJEPA Epps-Pulley implementation."""

    def __init__(self, t_max: float = 3.0, n_points: int = 17) -> None:
        super().__init__()
        t = torch.linspace(0, t_max, n_points)
        dt = t_max / (n_points - 1)
        phi = (-0.5 * t**2).exp()
        weights = torch.full((n_points,), 2 * dt)
        weights[[0, -1]] = dt
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        x_t = x.unsqueeze(-1) * self.t
        err = (x_t.cos().mean(0) - self.phi).square() + x_t.sin().mean(0).square()
        return (err @ self.weights) * n


class SlicedEppsPulley(nn.Module):
    """Matches released stable-pretraining: one SIGReg over flattened view samples."""

    def __init__(self, num_slices: int = 1024, t_max: float = 3.0, n_points: int = 17) -> None:
        super().__init__()
        self.num_slices = int(num_slices)
        self.ep = EppsPulley(t_max=t_max, n_points=n_points)
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("Released LeJEPA SIGReg expects [N,D]")
        with torch.no_grad():
            step = int(self.global_step.item())
            generator = torch.Generator(device=x.device).manual_seed(step)
            directions = torch.randn(
                x.size(-1), self.num_slices, device=x.device, generator=generator
            )
            directions = directions / directions.norm(p=2, dim=0)
            self.global_step.add_(1)
        return self.ep(x @ directions).mean()


class LeJEPAFair(nn.Module):
    """Released LeJEPA objective with only the shared-backbone adaptation."""

    def __init__(self, config: dict, base: FreshPLIPVisionEncoder) -> None:
        super().__init__()
        l = config["lejepa"]
        self.encoder = FlexiblePLIPVisionEncoder(base)
        proj = int(l["projector_dim"])
        hidden = int(l["projector_hidden_dim"])
        self.projector = nn.Sequential(
            nn.Linear(768, proj, bias=True),
            MLP(
                in_channels=proj,
                hidden_channels=[hidden, hidden, proj],
                norm_layer=nn.BatchNorm1d,
                activation_layer=nn.ReLU,
                inplace=True,
                dropout=0.0,
            ),
        )
        self.sigreg = SlicedEppsPulley(
            num_slices=int(l["sigreg_slices"]),
            t_max=float(l["sigreg_t_max"]),
            n_points=int(l["sigreg_points"]),
        )
        self.lamb = float(l["lambda"])

    def _features(self, images: torch.Tensor) -> torch.Tensor:
        # Necessary shared-backbone adaptation: FreshPLIP exposes final patch tokens;
        # its established downstream representation is their mean.
        return self.encoder(images).mean(dim=1)

    def forward(self, global_views: list[torch.Tensor], local_views: list[torch.Tensor]):
        if len(global_views) != 2 or len(local_views) not in (4, 6):
            raise ValueError("Expected LeJEPA 2G+4L or 2G+6L")
        batch = global_views[0].shape[0]
        global_features = self._features(torch.cat(global_views, dim=0))
        local_features = self._features(torch.cat(local_views, dim=0))
        all_features = torch.cat((global_features, local_features), dim=0)
        projected = self.projector(all_features)
        n_views = len(global_views) + len(local_views)
        projected = projected.view(n_views, batch, -1)
        center = projected[: len(global_views)].mean(0)
        inv_loss = (center.unsqueeze(0) - projected).square().mean()
        sigreg_loss = self.sigreg(projected.reshape(-1, projected.size(-1)))
        loss = inv_loss + self.lamb * sigreg_loss
        return loss, inv_loss, sigreg_loss, projected


def build_lejepa_fair(config: dict, device: torch.device) -> LeJEPAFair:
    base = FreshPLIPVisionEncoder(config["plip_config_dir"], 256).to(device)
    if base.num_patches != 64 or base.hidden_size != 768 or base.patch_size != 32:
        raise RuntimeError("Unexpected shared PLIP/CLIP backbone")
    return LeJEPAFair(config, base).to(device)


def _lejepa_optimizer(model: nn.Module, config: dict):
    l = config["lejepa"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(l["learning_rate"]),
        weight_decay=float(l["weight_decay"]),
        betas=(0.9, 0.999),
    )


@dataclass
class StepState:
    loss: float
    extras: dict[str, float]


class BenchmarkStepper:
    def __init__(self, config: dict, setting: str, device: torch.device) -> None:
        self.config = config
        self.setting = setting
        self.device = device
        self.masker = None
        self.teacher = None
        self.predictor = None
        self.model = None
        if setting == "ijepa_4layer":
            student, teacher, predictor = build_ijepa_fair(config, device)
            self.encoder = student
            self.teacher = teacher
            self.predictor = predictor
            self.trainable = list(student.parameters()) + list(predictor.parameters())
            self.optimizer = _ijepa_optimizer(student, predictor, config)
            self.masker = IJEPAOfficialMaskSampler(config, int(config["seed"]))
        else:
            self.model = build_lejepa_fair(config, device)
            self.encoder = self.model.encoder.base
            self.trainable = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = _lejepa_optimizer(self.model, config)

    def train_mode(self) -> None:
        if self.setting == "ijepa_4layer":
            self.encoder.train()
            self.predictor.train()
            self.teacher.eval()
        else:
            self.model.train()

    def step(self, batch: dict) -> StepState:
        self.train_mode()
        self.optimizer.zero_grad(set_to_none=True)
        use_bf16 = bool(self.config["benchmark"]["bf16"])
        if self.setting == "ijepa_4layer":
            images = batch["image"].to(self.device, dtype=torch.float32, non_blocking=True).div_(255.0)
            context, targets = self.masker.sample(images.shape[0], self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                loss, _, _ = objective(
                    self.encoder, self.teacher, self.predictor, images, context, targets
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite I-JEPA loss")
            loss.backward()
            self.optimizer.step()
            update_ema(self.encoder, self.teacher, float(self.config["ijepa"]["ema_momentum"]))
            return StepState(
                float(loss.detach()),
                {
                    "context_tokens": float(context.shape[1]),
                    "target_tokens_per_block": float(targets.shape[2]),
                },
            )

        global_batch = batch["global_views"].to(self.device, non_blocking=True)
        local_batch = batch["local_views"].to(self.device, non_blocking=True)
        global_views = [global_batch[:, i] for i in range(global_batch.shape[1])]
        local_views = [local_batch[:, i] for i in range(local_batch.shape[1])]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            loss, inv, sigreg, _ = self.model(global_views, local_views)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite LeJEPA loss")
        loss.backward()
        self.optimizer.step()
        return StepState(
            float(loss.detach()),
            {
                "invariance_loss": float(inv.detach()),
                "sigreg_loss": float(sigreg.detach()),
                "num_views": float(len(global_views) + len(local_views)),
            },
        )

    def parameter_summary(self) -> dict[str, int]:
        encoder = _count_parameters(self.encoder)
        if self.setting == "ijepa_4layer":
            auxiliary = _count_parameters(self.predictor)
            teacher = _count_parameters(self.teacher)
            total_trainable = _count_trainable_parameters(self.encoder) + _count_trainable_parameters(
                self.predictor
            )
            resident = encoder + auxiliary + teacher
        else:
            auxiliary = _count_parameters(self.model.projector)
            teacher = 0
            total_trainable = _count_trainable_parameters(self.model)
            resident = _count_parameters(self.model)
        return {
            "encoder_parameters": int(encoder),
            "auxiliary_trainable_parameters": int(auxiliary),
            "teacher_nontrainable_parameters": int(teacher),
            "total_trainable_parameters": int(total_trainable),
            "total_resident_parameters": int(resident),
        }


def _workload_summary(setting: str, measured_rows: list[dict]) -> dict[str, float]:
    if setting == "ijepa_4layer":
        c = statistics.mean(float(r["context_tokens"]) for r in measured_rows)
        t = statistics.mean(float(r["target_tokens_per_block"]) for r in measured_rows)
        return {
            "logical_encoder_views_per_source": 2.0,
            "physical_encoder_calls_per_step": 2.0,
            "input_pixels_per_source": float(2 * 256 * 256),
            "patch_tokens_per_source": 128.0,
            "encoder_transformer_tokens_per_source": float((1 + c) + 65),
            "predictor_transformer_tokens_per_source": float(4 * (c + t)),
            "mean_context_tokens": c,
            "mean_target_tokens_per_block": t,
        }
    local = 4 if setting.endswith("2g4l") else 6
    views = 2 + local
    patch_tokens = 2 * 64 + local * 9
    pixels = 2 * 256 * 256 + local * 96 * 96
    return {
        "logical_encoder_views_per_source": float(views),
        "physical_encoder_calls_per_step": 2.0,
        "input_pixels_per_source": float(pixels),
        "patch_tokens_per_source": float(patch_tokens),
        "encoder_transformer_tokens_per_source": float(patch_tokens + views),
        "predictor_transformer_tokens_per_source": 0.0,
        "global_views": 2.0,
        "local_views": float(local),
    }


def _stats(values: Iterable[float], prefix: str) -> dict[str, float]:
    values = [float(x) for x in values]
    ordered = sorted(values)
    p90_index = min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)
    return {
        f"{prefix}_mean": statistics.mean(values),
        f"{prefix}_median": statistics.median(values),
        f"{prefix}_std": statistics.pstdev(values),
        f"{prefix}_p90": ordered[p90_index],
    }


def _profile_one_step(stepper: BenchmarkStepper, batch: dict) -> dict:
    try:
        from torch.profiler import ProfilerActivity, profile

        stepper.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_flops=True,
            record_shapes=False,
            profile_memory=False,
        ) as prof:
            stepper.step(batch)
        torch.cuda.synchronize()
        estimated = sum(float(getattr(e, "flops", 0) or 0) for e in prof.key_averages())
        return {"profiler_estimated_flops_per_step": estimated, "profiler_error": None}
    except Exception as exc:
        torch.cuda.empty_cache()
        return {
            "profiler_estimated_flops_per_step": None,
            "profiler_error": f"{type(exc).__name__}: {exc}",
        }


def run_setting(config: dict, setting: str) -> dict:
    validate_config(config)
    if setting not in SETTINGS:
        raise ValueError(setting)
    if not torch.cuda.is_available():
        raise RuntimeError("Fairness benchmark requires CUDA")
    output = _setting_dir(config, setting)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("step_metrics.csv", "profile.json", "failure.json"):
        if (output / name).exists():
            raise FileExistsError(f"Refuse to overwrite {output / name}")

    repo = _repo_root(config)
    before = protected_manifest(repo)
    atomic_json_dump(config, output / "resolved_config.json")
    seed_everything(int(config["seed"]))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = bool(config["benchmark"]["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(config["benchmark"]["tf32"])
    device = torch.device("cuda")
    phase = "initialization"
    try:
        loader = build_fairness_loader(config, setting)
        stepper = BenchmarkStepper(config, setting, device)
        initial_hash = _encoder_hash(stepper.encoder)
        params = stepper.parameter_summary()
        stream = _infinite(loader)
        source_digest = hashlib.sha256()
        warmup_steps = int(config["benchmark"]["warmup_steps"])
        measured_steps = int(config["benchmark"]["measured_steps"])
        batch_size = int(config["benchmark"]["batch_size"])

        phase = "warmup"
        for _ in range(warmup_steps):
            batch = next(stream)
            _update_source_hash(source_digest, batch)
            stepper.step(batch)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        measured = []
        phase = "measurement"
        for measured_step in range(1, measured_steps + 1):
            wall_start = time.perf_counter()
            batch = next(stream)
            fetched = time.perf_counter()
            _update_source_hash(source_digest, batch)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            state = stepper.step(batch)
            end_event.record()
            torch.cuda.synchronize()
            wall_end = time.perf_counter()
            measured.append(
                {
                    "step": measured_step,
                    "loss": state.loss,
                    "batch_fetch_seconds": fetched - wall_start,
                    "wall_seconds": wall_end - wall_start,
                    "gpu_milliseconds": float(start_event.elapsed_time(end_event)),
                    "source_images": batch_size,
                    "source_images_per_second": batch_size / (wall_end - wall_start),
                    **state.extras,
                }
            )

        workload = _workload_summary(setting, measured)
        peak_allocated = torch.cuda.max_memory_allocated() / 2**30
        peak_reserved = torch.cuda.max_memory_reserved() / 2**30
        phase = "profiler"
        profiler = _profile_one_step(stepper, next(stream))
        profile_value = {
            "status": "completed",
            "setting": setting,
            "seed": int(config["seed"]),
            "source_implementation": (
                "facebookresearch/ijepa" if setting == "ijepa_4layer"
                else "galilai-group/stable-pretraining LeJEPA"
            ),
            "warmup_steps": warmup_steps,
            "measured_steps": measured_steps,
            "batch_size": batch_size,
            "source_images_in_timed_steps": measured_steps * batch_size,
            "source_sequence_sha256_120_steps": source_digest.hexdigest(),
            "initial_encoder_sha256": initial_hash,
            "precision": "bf16",
            "tf32": bool(config["benchmark"]["tf32"]),
            "peak_gpu_memory_allocated_gib": peak_allocated,
            "peak_gpu_memory_reserved_gib": peak_reserved,
            **params,
            **workload,
            **_stats((r["wall_seconds"] for r in measured), "wall_seconds"),
            **_stats((r["gpu_milliseconds"] for r in measured), "gpu_milliseconds"),
            **_stats((r["batch_fetch_seconds"] for r in measured), "batch_fetch_seconds"),
            **_stats((r["source_images_per_second"] for r in measured), "source_images_per_second"),
            **profiler,
        }
        write_csv(measured, output / "step_metrics.csv")
        if protected_manifest(repo) != before:
            raise AssertionError("Protected existing pipeline changed")
        profile_value["protected_existing_pipeline_unchanged"] = True
        atomic_json_dump(profile_value, output / "profile.json")
        return profile_value
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        failure = {
            "status": "oom",
            "setting": setting,
            "phase": phase,
            "error": f"{type(exc).__name__}: {exc}",
            "protected_existing_pipeline_unchanged": protected_manifest(repo) == before,
        }
        atomic_json_dump(failure, output / "failure.json")
        return failure
    except Exception as exc:
        failure = {
            "status": "failed",
            "setting": setting,
            "phase": phase,
            "error": f"{type(exc).__name__}: {exc}",
            "protected_existing_pipeline_unchanged": protected_manifest(repo) == before,
        }
        atomic_json_dump(failure, output / "failure.json")
        raise


def smoke(config: dict) -> dict:
    validate_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("Smoke test requires CUDA")
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    repo = _repo_root(config)
    before = protected_manifest(repo)
    index = build_source_index(Path(config["data_root"]))
    rows = all_source_rows(index)
    assert len(rows) == 7901 and len({(r["fold"], r["sample_index"]) for r in rows}) == 7901
    cache = preload_images(rows[:2], index)
    raw = torch.stack(
        [
            torch.from_numpy(np.array(cache[int(r["fold"]), int(r["sample_index"])], copy=True)).permute(2, 0, 1)
            for r in rows[:2]
        ]
    )
    assert raw.dtype == torch.uint8 and raw.shape == (2, 3, 256, 256)
    device = torch.device("cuda")

    seed_everything(int(config["seed"]))
    student, teacher, predictor = build_ijepa_fair(config, device)
    ijepa_hash = _encoder_hash(student)
    assert len(predictor.transformer.layers) == 4
    assert predictor.transformer.layers[0].self_attn.num_heads == 12
    transform = make_ijepa_transform(config)
    images = torch.stack([transform(x) for x in raw]).to(device, dtype=torch.float32).div_(255.0)
    sampler = IJEPAOfficialMaskSampler(config, int(config["seed"]))
    context, targets = sampler.sample(2, device)
    for bi in range(2):
        context_set = set(context[bi].tolist())
        target_union = set(targets[bi].flatten().tolist())
        assert context_set.isdisjoint(target_union)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, prediction, target = objective(student, teacher, predictor, images, context, targets)
    assert torch.isfinite(loss) and prediction.shape == (*targets.shape, 768) and not target.requires_grad
    loss.backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert predictor.mask_token.grad is not None

    del student, teacher, predictor, loss, prediction, target
    gc.collect()
    torch.cuda.empty_cache()

    seed_everything(int(config["seed"]))
    lejepa = build_lejepa_fair(config, device)
    lejepa_hash = _encoder_hash(lejepa.encoder.base)
    assert ijepa_hash == lejepa_hash
    global_transform, local_transform = make_lejepa_transforms(config)
    globals_ = [torch.stack([global_transform(x) for x in raw]).to(device) for _ in range(2)]
    locals6 = [torch.stack([local_transform(x) for x in raw]).to(device) for _ in range(6)]
    assert all(v.shape == (2, 3, 256, 256) for v in globals_)
    assert all(v.shape == (2, 3, 96, 96) for v in locals6)
    lejepa.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        direct = lejepa.encoder.base(globals_[0])
        wrapped = lejepa.encoder(globals_[0])
        local_tokens = lejepa.encoder(locals6[0])
    assert torch.allclose(direct, wrapped, atol=0, rtol=0)
    assert direct.shape == (2, 64, 768) and local_tokens.shape == (2, 9, 768)
    lejepa.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss6, inv6, sig6, projected6 = lejepa(globals_, locals6)
    assert projected6.shape == (8, 2, 512)
    assert torch.isfinite(loss6) and torch.isfinite(inv6) and torch.isfinite(sig6)
    loss6.backward()
    assert next(lejepa.encoder.base.parameters()).grad is not None
    assert next(lejepa.projector.parameters()).grad is not None
    lejepa.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss4, _, _, projected4 = lejepa(globals_, locals6[:4])
    assert projected4.shape == (6, 2, 512) and torch.isfinite(loss4)

    result = {
        "passed": True,
        "dataset_unique_source_images": 7901,
        "labels_loaded": False,
        "downstream_validation_or_test_run": False,
        "shared_initial_encoder_sha256": ijepa_hash,
        "ijepa_source": "facebookresearch/ijepa",
        "ijepa_random_resized_crop_scale": [0.3, 1.0],
        "ijepa_predictor_depth": 4,
        "ijepa_predictor_heads_match_encoder": 12,
        "ijepa_target_blocks": 4,
        "ijepa_target_target_overlap_allowed": True,
        "ijepa_context_target_overlap": False,
        "lejepa_source": "galilai-group/stable-pretraining released LeJEPA",
        "lejepa_loss_form": "inv_loss + lambda * sigreg_loss",
        "lejepa_lambda": 0.02,
        "lejepa_2g4l_total_views": 6,
        "lejepa_2g6l_total_views": 8,
        "global_shape": [2, 3, 256, 256],
        "local_shape": [2, 3, 96, 96],
        "global_patch_tokens": 64,
        "local_patch_tokens": 9,
        "global_wrapper_exactly_matches_existing_encoder": True,
        "protected_existing_pipeline_unchanged": protected_manifest(repo) == before,
    }
    assert result["protected_existing_pipeline_unchanged"]
    atomic_json_dump(result, root / "smoke_test.json")
    return result


def require_smoke(config: dict) -> None:
    path = Path(config["output_dir"]) / "smoke_test.json"
    if not path.exists():
        raise FileNotFoundError(f"Run fairness smoke test first: {path}")
    value = json.loads(path.read_text())
    if not value.get("passed"):
        raise RuntimeError("Fairness smoke test did not pass")


def comparison(config: dict) -> dict:
    validate_config(config)
    root = Path(config["output_dir"])
    profiles = {}
    for setting in SETTINGS:
        directory = _setting_dir(config, setting)
        if (directory / "profile.json").exists():
            value = json.loads((directory / "profile.json").read_text())
        elif (directory / "failure.json").exists():
            value = json.loads((directory / "failure.json").read_text())
        else:
            value = {"status": "missing", "setting": setting}
        profiles[setting] = value

    successful = [p for p in profiles.values() if p.get("status") == "completed"]
    encoder_hashes = {p["initial_encoder_sha256"] for p in successful}
    source_hashes = {p["source_sequence_sha256_120_steps"] for p in successful}
    if len(encoder_hashes) > 1:
        raise AssertionError("Initial encoder hashes differ")
    if len(source_hashes) > 1:
        raise AssertionError("Source-image sequences differ")

    reference = profiles["ijepa_4layer"]
    rows = []
    for setting in SETTINGS:
        p = profiles[setting]
        row = {
            "setting": setting,
            "status": p.get("status"),
            "encoder_parameters": p.get("encoder_parameters"),
            "auxiliary_trainable_parameters": p.get("auxiliary_trainable_parameters"),
            "total_trainable_parameters": p.get("total_trainable_parameters"),
            "total_resident_parameters": p.get("total_resident_parameters"),
            "input_pixels_per_source": p.get("input_pixels_per_source"),
            "patch_tokens_per_source": p.get("patch_tokens_per_source"),
            "gpu_ms_mean": p.get("gpu_milliseconds_mean"),
            "wall_seconds_mean": p.get("wall_seconds_mean"),
            "source_images_per_second_mean": p.get("source_images_per_second_mean"),
            "peak_gpu_memory_allocated_gib": p.get("peak_gpu_memory_allocated_gib"),
            "profiler_estimated_flops_per_step": p.get("profiler_estimated_flops_per_step"),
        }
        if p.get("status") == "completed" and reference.get("status") == "completed":
            for out, key in (
                ("relative_gpu_time", "gpu_milliseconds_mean"),
                ("relative_wall_time", "wall_seconds_mean"),
                ("relative_pixels", "input_pixels_per_source"),
                ("relative_patch_tokens", "patch_tokens_per_source"),
                ("relative_peak_vram", "peak_gpu_memory_allocated_gib"),
                ("relative_profiled_flops", "profiler_estimated_flops_per_step"),
            ):
                num, den = p.get(key), reference.get(key)
                row[out] = float(num) / float(den) if num is not None and den not in (None, 0) else None
        rows.append(row)

    write_csv(rows, root / "comparison.csv")
    result = {
        "purpose": "source-faithful compute/workload diagnostic only",
        "successful_initial_encoder_hash_identical": len(encoder_hashes) <= 1,
        "successful_source_sequence_hash_identical": len(source_hashes) <= 1,
        "profiles": profiles,
        "rows": rows,
    }
    atomic_json_dump(result, root / "comparison.json")
    lines = [
        "# I-JEPA vs LeJEPA Source-Faithful Fairness Diagnostic",
        "",
        "I-JEPA follows facebookresearch/ijepa; LeJEPA follows the authors' released stable-pretraining 2G+6L implementation. Only shared-backbone/256-input and the chosen I-JEPA predictor depth=4 are deliberate adaptations.",
        "",
        "| Setting | Status | Encoder params | Aux params | Pixels/source | Patch tokens/source | GPU ms/step | Wall s/step | Images/s | Peak VRAM GiB | Relative GPU | Relative wall |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        def fmt(key, digits=3):
            value = r.get(key)
            return "—" if value is None else f"{float(value):.{digits}f}"
        lines.append(
            f"| {r['setting']} | {r['status']} | {r.get('encoder_parameters') or '—'} | "
            f"{r.get('auxiliary_trainable_parameters') or '—'} | {fmt('input_pixels_per_source',0)} | "
            f"{fmt('patch_tokens_per_source',0)} | {fmt('gpu_ms_mean')} | {fmt('wall_seconds_mean',4)} | "
            f"{fmt('source_images_per_second_mean',2)} | {fmt('peak_gpu_memory_allocated_gib',2)} | "
            f"{fmt('relative_gpu_time',2)} | {fmt('relative_wall_time',2)} |"
        )
    lines += [
        "",
        f"- Identical initial encoder weights: **{len(encoder_hashes) <= 1}**",
        f"- Identical 120-step source-image sequence: **{len(source_hashes) <= 1}**",
        "- 2G+4L is workload decomposition only; 2G+6L is the released LeJEPA view recipe.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result
