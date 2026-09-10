"""Isolated compute-fairness diagnostic for I-JEPA vs LeJEPA.

This module intentionally does not modify or depend on the existing I-JEPA
training/selection pipeline. It reuses only stable data/indexing/model helpers.
All benchmark outputs live under an output path containing "ijepa", so the
existing I-JEPA protected manifest ignores them by construction.
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
from .ijepa import BlockMasks, gather_tokens, objective, positional_grid
from .ijepa_training import protected_manifest
from .models import FreshPLIPVisionEncoder, make_teacher, normalize_clip, update_ema
from .parquet import build_source_index, preload_images
from .training import _make_optimizer
from .utils import atomic_json_dump, seed_everything, write_csv


SETTINGS = ("ijepa_4layer", "lejepa_2g4l", "lejepa_2g6l")


def validate_config(c: dict) -> None:
    assert c["seed"] == 20260903
    b = c["benchmark"]
    assert b["batch_size"] == 128
    assert b["warmup_steps"] == 20 and b["measured_steps"] == 100
    assert b["bf16"] and b["cache_in_ram"]
    i = c["ijepa"]
    assert i == {
        "image_size": 256,
        "patch_size": 32,
        "hidden_size": 768,
        "predictor_dim": 384,
        "predictor_depth": 4,
        "predictor_heads": 6,
        "predictor_mlp_ratio": 4,
        "dropout": 0.0,
        "target_count": 4,
        "ema_momentum": 0.996,
        "gradient_clip_norm": 5.0,
    }
    l = c["lejepa"]
    assert l["global_views"] == 2
    assert l["local_views"] == [4, 6]
    assert l["global_size"] == 256 and l["local_size"] == 96
    assert l["global_scale"] == [0.3, 1.0]
    assert l["local_scale"] == [0.05, 0.3]
    assert l["lambda"] == 0.05
    assert l["projector_dim"] == 512 and l["projector_hidden_dim"] == 2048
    assert l["sigreg_slices"] == 1024 and l["sigreg_points"] == 17
    assert float(l["sigreg_t_max"]) == 3.0


def _repo_root(config: dict) -> Path:
    # Match the existing experiment layout: <repo>/outputs/<run_name>.
    return Path(config["output_dir"]).resolve().parents[1]


def _setting_dir(config: dict, setting: str) -> Path:
    root = Path(config["output_dir"])
    mapping = {
        "ijepa_4layer": "ijepa_4layer",
        "lejepa_2g4l": "lejepa_2g4l",
        "lejepa_2g6l": "lejepa_2g6l",
    }
    return root / mapping[setting]


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


def _native_photometric(config: dict) -> list[nn.Module]:
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


def make_lejepa_transforms(config: dict):
    l = config["lejepa"]
    global_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop(
                size=(int(l["global_size"]), int(l["global_size"])),
                scale=tuple(float(x) for x in l["global_scale"]),
                antialias=True,
            ),
            *_native_photometric(config),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    local_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop(
                size=(int(l["local_size"]), int(l["local_size"])),
                scale=tuple(float(x) for x in l["local_scale"]),
                antialias=True,
            ),
            *_native_photometric(config),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    return global_transform, local_transform


class FairnessDataset(Dataset):
    """Unlabelled PanNuke source images with method-specific view construction."""

    def __init__(self, base: Dataset, setting: str, config: dict) -> None:
        if setting not in SETTINGS:
            raise ValueError(setting)
        self.base = base
        self.setting = setting
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
            raise RuntimeError(f"Unexpected PanNuke image: dtype={image.dtype}, shape={tuple(image.shape)}")
        if self.setting == "ijepa_4layer":
            return {
                "image": image,
                "fold": int(fold),
                "sample_index": int(sample_index),
            }
        assert self.global_transform is not None and self.local_transform is not None
        globals_ = torch.stack([self.global_transform(image) for _ in range(2)])
        locals_ = torch.stack([self.local_transform(image) for _ in range(self.n_local)])
        return {
            "global_views": globals_,
            "local_views": locals_,
            "fold": int(fold),
            "sample_index": int(sample_index),
        }


def build_fairness_loader(config: dict, setting: str) -> DataLoader:
    source_index = build_source_index(Path(config["data_root"]))
    rows = all_source_rows(source_index)
    if len(rows) != 7901 or len({(int(r["fold"]), int(r["sample_index"])) for r in rows}) != 7901:
        raise ValueError("Fairness benchmark requires exactly 7,901 unique PanNuke source images.")
    cache = preload_images(rows, source_index) if config["benchmark"]["cache_in_ram"] else None
    base = PanNukeImageDataset(
        rows,
        source_index,
        cache,
        include_label=False,
        include_key=True,
    )
    dataset = FairnessDataset(base, setting, config)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    workers = int(config["benchmark"]["num_workers"])
    kwargs = dict(
        batch_size=int(config["benchmark"]["batch_size"]),
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=True,
        generator=generator,
        worker_init_fn=_seed_worker,
    )
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def _infinite(loader: DataLoader) -> Iterator[dict]:
    while True:
        yield from loader


def _update_source_hash(digest: "hashlib._Hash", batch: dict) -> None:
    folds = batch["fold"].tolist()
    indexes = batch["sample_index"].tolist()
    for fold, index in zip(folds, indexes, strict=True):
        digest.update(f"{int(fold)}:{int(index)};".encode("ascii"))
    digest.update(b"|")


def _init_predictor_module(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class IJEPAFairPredictor(nn.Module):
    """Benchmark-only 4-layer predictor; existing I-JEPA code remains untouched."""

    def __init__(self) -> None:
        super().__init__()
        self.input_projection = nn.Linear(768, 384)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 384))
        self.register_buffer("position", positional_grid(384))
        layer = nn.TransformerEncoderLayer(
            d_model=384,
            nhead=6,
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
        for block in self.transformer.layers:
            nn.init.trunc_normal_(block.self_attn.in_proj_weight, std=0.02)
            if block.self_attn.in_proj_bias is not None:
                nn.init.zeros_(block.self_attn.in_proj_bias)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, tokens, context_indices, target_indices):
        b, blocks, target_length = target_indices.shape
        positions = self.position.expand(b, -1, -1)
        context = self.input_projection(tokens) + gather_tokens(positions, context_indices)
        context = (
            context[:, None]
            .expand(-1, blocks, -1, -1)
            .reshape(b * blocks, tokens.shape[1], 384)
        )
        queries = self.mask_token + gather_tokens(
            positions, target_indices.flatten(1)
        ).reshape(b * blocks, target_length, 384)
        predicted = self.transformer(torch.cat((context, queries), dim=1))[:, -target_length:]
        predicted = self.output_projection(self.final_norm(predicted))
        return predicted.reshape(b, blocks, target_length, 768)


def build_ijepa_fair(config: dict, device: torch.device):
    student = FreshPLIPVisionEncoder(config["plip_config_dir"], 256).to(device)
    if student.num_patches != 64 or student.hidden_size != 768 or student.patch_size != 32:
        raise RuntimeError("Unexpected shared PLIP/CLIP backbone architecture.")
    teacher = make_teacher(student)
    predictor = IJEPAFairPredictor().to(device)
    return student, teacher, predictor


class FlexiblePLIPVisionEncoder(nn.Module):
    """Same fresh PLIP/CLIP encoder, enabling its built-in local-view position interpolation."""

    def __init__(self, base: FreshPLIPVisionEncoder) -> None:
        super().__init__()
        self.base = base

    @property
    def hidden_size(self) -> int:
        return self.base.hidden_size

    @property
    def patch_size(self) -> int:
        return self.base.patch_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        h, w = images.shape[-2:]
        if h == self.base.image_size and w == self.base.image_size:
            # Exact original path for all 256x256 global views.
            return self.base(images)
        if h % self.patch_size or w % self.patch_size:
            raise ValueError("Local view dimensions must be divisible by the fixed patch size.")

        vision = self.base.model.vision_model
        # Transformers >=4.48 (the repo minimum) provides CLIP's official bicubic
        # interpolation path for learned patch positional embeddings.
        embedded = vision.embeddings(
            normalize_clip(images),
            interpolate_pos_encoding=True,
        )
        encoded = vision.encoder(
            inputs_embeds=vision.pre_layrnorm(embedded),
            return_dict=True,
        ).last_hidden_state
        patches = vision.post_layernorm(encoded[:, 1:])
        gh, gw = h // self.patch_size, w // self.patch_size
        expected = (images.shape[0], gh * gw, self.hidden_size)
        if patches.shape != expected:
            raise RuntimeError(f"Unexpected local token shape {tuple(patches.shape)} != {expected}")
        return patches


class SlicedEppsPulley(nn.Module):
    """Paper-style SIGReg with per-view sliced Epps-Pulley statistics."""

    def __init__(self, num_slices: int = 1024, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        if n_points % 2 != 1:
            raise ValueError("n_points must be odd.")
        self.num_slices = int(num_slices)
        t = torch.linspace(0.0, float(t_max), int(n_points), dtype=torch.float32)
        dt = float(t_max) / (int(n_points) - 1)
        weights = torch.full((int(n_points),), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi = torch.exp(-0.5 * t.square())
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

    def _one_view(self, x: torch.Tensor, seed: int) -> torch.Tensor:
        x = x.float()
        generator = torch.Generator(device=x.device).manual_seed(int(seed))
        directions = torch.randn(
            x.size(-1), self.num_slices, device=x.device, dtype=x.dtype, generator=generator
        )
        directions = directions / directions.norm(p=2, dim=0).clamp_min_(1e-12)
        projected = x @ directions
        x_t = projected.unsqueeze(-1) * self.t
        err = (
            (x_t.cos().mean(0) - self.phi).square()
            + x_t.sin().mean(0).square()
        )
        statistic = (err @ self.weights) * x.shape[0]
        return statistic.mean()

    def forward(self, per_view: torch.Tensor) -> torch.Tensor:
        if per_view.ndim != 3:
            raise ValueError("SIGReg expects [V, B, D] projected embeddings.")
        step = int(self.global_step.item())
        losses = [
            self._one_view(per_view[v], seed=step * 10007 + v)
            for v in range(per_view.shape[0])
        ]
        self.global_step.add_(1)
        return torch.stack(losses).mean()


class LeJEPAFair(nn.Module):
    """LeJEPA adapted to the exact shared PLIP/CLIP patch-token encoder."""

    def __init__(self, config: dict, base: FreshPLIPVisionEncoder) -> None:
        super().__init__()
        l = config["lejepa"]
        self.encoder = FlexiblePLIPVisionEncoder(base)
        hidden = int(l["projector_hidden_dim"])
        proj = int(l["projector_dim"])
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
        return self.encoder(images).mean(dim=1)

    def forward(self, global_views: list[torch.Tensor], local_views: list[torch.Tensor]):
        if len(global_views) != 2:
            raise ValueError("LeJEPA fairness benchmark requires exactly two global views.")
        if len(local_views) not in (4, 6):
            raise ValueError("LeJEPA fairness benchmark requires four or six local views.")
        batch = global_views[0].shape[0]

        global_features = self._features(torch.cat(global_views, dim=0))
        local_features = self._features(torch.cat(local_views, dim=0))
        features = torch.cat((global_features, local_features), dim=0)
        projected = self.projector(features)
        views = len(global_views) + len(local_views)
        projected = projected.reshape(views, batch, -1)

        center = projected[:2].mean(dim=0)
        inv_loss = (projected.float() - center.float().unsqueeze(0)).square().mean()
        sigreg_loss = self.sigreg(projected)
        loss = (1.0 - self.lamb) * inv_loss + self.lamb * sigreg_loss
        return loss, inv_loss, sigreg_loss, projected


def build_lejepa_fair(config: dict, device: torch.device) -> LeJEPAFair:
    base = FreshPLIPVisionEncoder(config["plip_config_dir"], 256).to(device)
    if base.num_patches != 64 or base.hidden_size != 768 or base.patch_size != 32:
        raise RuntimeError("Unexpected shared PLIP/CLIP backbone architecture.")
    return LeJEPAFair(config, base).to(device)


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
            self.optimizer = _make_optimizer(
                self.trainable,
                float(config["benchmark"]["learning_rate"]),
                float(config["benchmark"]["weight_decay"]),
            )
            self.masker = BlockMasks(int(config["seed"]))
        else:
            self.model = build_lejepa_fair(config, device)
            self.encoder = self.model.encoder.base
            self.trainable = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = _make_optimizer(
                self.trainable,
                float(config["benchmark"]["learning_rate"]),
                float(config["benchmark"]["weight_decay"]),
            )

    def train_mode(self) -> None:
        if self.setting == "ijepa_4layer":
            self.encoder.train()
            assert self.predictor is not None and self.teacher is not None
            self.predictor.train()
            self.teacher.eval()
        else:
            assert self.model is not None
            self.model.train()

    def step(self, batch: dict) -> StepState:
        self.train_mode()
        self.optimizer.zero_grad(set_to_none=True)
        autocast = torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=bool(self.config["benchmark"]["bf16"])
        )
        if self.setting == "ijepa_4layer":
            assert self.masker is not None and self.teacher is not None and self.predictor is not None
            images = batch["image"].to(
                self.device, dtype=torch.float32, non_blocking=True
            ).div_(255.0)
            context, targets = self.masker.sample(images.shape[0], self.device)
            with autocast:
                loss, _, _ = objective(
                    self.encoder, self.teacher, self.predictor, images, context, targets
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite I-JEPA loss.")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                self.trainable, float(self.config["ijepa"]["gradient_clip_norm"])
            )
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite I-JEPA gradient norm.")
            self.optimizer.step()
            update_ema(
                self.encoder,
                self.teacher,
                float(self.config["ijepa"]["ema_momentum"]),
            )
            return StepState(
                float(loss.detach()),
                {
                    "context_tokens": float(context.shape[1]),
                    "target_tokens_per_block": float(targets.shape[2]),
                    "gradient_norm": float(norm.detach()),
                },
            )

        assert self.model is not None
        global_batch = batch["global_views"].to(self.device, non_blocking=True)
        local_batch = batch["local_views"].to(self.device, non_blocking=True)
        global_views = [global_batch[:, i] for i in range(global_batch.shape[1])]
        local_views = [local_batch[:, i] for i in range(local_batch.shape[1])]
        with autocast:
            loss, inv, sigreg, _ = self.model(global_views, local_views)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite LeJEPA loss.")
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
            assert self.predictor is not None and self.teacher is not None
            auxiliary = _count_parameters(self.predictor)
            teacher = _count_parameters(self.teacher)
            total_trainable = _count_trainable_parameters(self.encoder) + _count_trainable_parameters(
                self.predictor
            )
            resident = encoder + auxiliary + teacher
        else:
            assert self.model is not None
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
        contexts = [float(r["context_tokens"]) for r in measured_rows]
        targets = [float(r["target_tokens_per_block"]) for r in measured_rows]
        c = statistics.mean(contexts)
        t = statistics.mean(targets)
        return {
            "logical_encoder_views_per_source": 2.0,
            "physical_encoder_calls_per_step": 2.0,
            "input_pixels_per_source": float(2 * 256 * 256),
            "patch_tokens_per_source": 128.0,
            "encoder_transformer_tokens_per_source": float((1 + c) + (1 + 64)),
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
    p90_index = min(len(ordered) - 1, math.ceil(0.90 * len(ordered)) - 1)
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
        estimated = sum(float(getattr(event, "flops", 0) or 0) for event in prof.key_averages())
        return {"profiler_estimated_flops_per_step": estimated, "profiler_error": None}
    except Exception as exc:  # profiler must never invalidate the timing benchmark
        if torch.cuda.is_available():
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
        raise RuntimeError("The fairness benchmark requires CUDA.")

    output = _setting_dir(config, setting)
    output.mkdir(parents=True, exist_ok=True)
    for protected_name in ("step_metrics.csv", "profile.json", "failure.json"):
        if (output / protected_name).exists():
            raise FileExistsError(
                f"Refuse to overwrite {output / protected_name}; use a fresh output directory."
            )

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
        if len(loader.dataset) != 7901:
            raise RuntimeError("Unexpected SSL dataset size.")
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
        measured: list[dict] = []
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

            row = {
                "step": measured_step,
                "loss": state.loss,
                "batch_fetch_seconds": fetched - wall_start,
                "wall_seconds": wall_end - wall_start,
                "gpu_milliseconds": float(start_event.elapsed_time(end_event)),
                "source_images": batch_size,
                "source_images_per_second": batch_size / (wall_end - wall_start),
                **state.extras,
            }
            measured.append(row)

        peak_allocated = torch.cuda.max_memory_allocated() / 2**30
        peak_reserved = torch.cuda.max_memory_reserved() / 2**30
        workload = _workload_summary(setting, measured)

        phase = "profiler"
        profiler_batch = next(stream)
        profiler = _profile_one_step(stepper, profiler_batch)

        wall_stats = _stats((r["wall_seconds"] for r in measured), "wall_seconds")
        gpu_stats = _stats((r["gpu_milliseconds"] for r in measured), "gpu_milliseconds")
        fetch_stats = _stats((r["batch_fetch_seconds"] for r in measured), "batch_fetch_seconds")
        throughput_stats = _stats(
            (r["source_images_per_second"] for r in measured), "source_images_per_second"
        )

        profile_value = {
            "status": "completed",
            "setting": setting,
            "seed": int(config["seed"]),
            "warmup_steps": warmup_steps,
            "measured_steps": measured_steps,
            "batch_size": batch_size,
            "source_images_in_timed_steps": measured_steps * batch_size,
            "source_sequence_sha256_120_steps": source_digest.hexdigest(),
            "initial_encoder_sha256": initial_hash,
            "precision": "bf16" if config["benchmark"]["bf16"] else "fp32",
            "tf32": bool(config["benchmark"]["tf32"]),
            "peak_gpu_memory_allocated_gib": peak_allocated,
            "peak_gpu_memory_reserved_gib": peak_reserved,
            **params,
            **workload,
            **wall_stats,
            **gpu_stats,
            **fetch_stats,
            **throughput_stats,
            **profiler,
        }
        write_csv(measured, output / "step_metrics.csv")
        after = protected_manifest(repo)
        if before != after:
            raise AssertionError("Protected existing pipeline/files changed during fairness benchmark.")
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
        raise RuntimeError("Smoke test requires CUDA.")
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    repo = _repo_root(config)
    before = protected_manifest(repo)

    source_index = build_source_index(Path(config["data_root"]))
    rows = all_source_rows(source_index)
    assert len(rows) == 7901
    assert len({(int(r["fold"]), int(r["sample_index"])) for r in rows}) == 7901
    selected = rows[:2]
    cache = preload_images(selected, source_index)
    raw = torch.stack(
        [
            torch.from_numpy(np.array(cache[int(r["fold"]), int(r["sample_index"])], copy=True))
            .permute(2, 0, 1)
            for r in selected
        ]
    )
    assert raw.dtype == torch.uint8 and raw.shape == (2, 3, 256, 256)

    device = torch.device("cuda")
    seed_everything(int(config["seed"]))
    student, teacher, predictor = build_ijepa_fair(config, device)
    ijepa_hash = _encoder_hash(student)
    assert len(predictor.transformer.layers) == 4
    images = raw.to(device, dtype=torch.float32).div(255.0)
    context, targets = BlockMasks(int(config["seed"])).sample(2, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, prediction, target = objective(student, teacher, predictor, images, context, targets)
    assert torch.isfinite(loss)
    assert prediction.shape == (*targets.shape, 768)
    assert not target.requires_grad
    loss.backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert predictor.input_projection.weight.grad is not None
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
    assert direct.shape == (2, 64, 768)
    assert local_tokens.shape == (2, 9, 768)

    lejepa.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss6, inv6, sig6, projected6 = lejepa(globals_, locals6)
    assert projected6.shape == (8, 2, 512)
    assert torch.isfinite(loss6) and torch.isfinite(inv6) and torch.isfinite(sig6)
    loss6.backward()
    assert next(lejepa.encoder.base.parameters()).grad is not None
    assert next(lejepa.projector.parameters()).grad is not None

    lejepa.zero_grad(set_to_none=True)
    locals4 = locals6[:4]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss4, _, _, projected4 = lejepa(globals_, locals4)
    assert projected4.shape == (6, 2, 512)
    assert torch.isfinite(loss4)

    result = {
        "passed": True,
        "dataset_unique_source_images": 7901,
        "labels_loaded": False,
        "balanced_metadata_loaded": False,
        "downstream_validation_or_test_run": False,
        "shared_initial_encoder_sha256": ijepa_hash,
        "ijepa_predictor_depth": 4,
        "ijepa_target_blocks": 4,
        "lejepa_2g4l_total_views": 6,
        "lejepa_2g6l_total_views": 8,
        "global_shape": [2, 3, 256, 256],
        "local_shape": [2, 3, 96, 96],
        "global_patch_tokens": 64,
        "local_patch_tokens": 9,
        "global_wrapper_exactly_matches_existing_encoder": True,
        "ijepa_teacher_stop_gradient": True,
        "ijepa_predictor_receives_gradient": True,
        "lejepa_encoder_and_projector_receive_gradient": True,
        "sigreg_finite": True,
        "protected_existing_pipeline_unchanged": protected_manifest(repo) == before,
    }
    assert result["protected_existing_pipeline_unchanged"]
    atomic_json_dump(result, root / "smoke_test.json")
    return result


def require_smoke(config: dict) -> None:
    path = Path(config["output_dir"]) / "smoke_test.json"
    if not path.exists():
        raise FileNotFoundError(f"Run the fairness smoke test first: {path}")
    value = json.loads(path.read_text())
    if not value.get("passed"):
        raise RuntimeError("Fairness smoke test did not pass.")


def comparison(config: dict) -> dict:
    validate_config(config)
    root = Path(config["output_dir"])
    records: list[dict] = []
    profiles: dict[str, dict] = {}
    for setting in SETTINGS:
        directory = _setting_dir(config, setting)
        profile_path = directory / "profile.json"
        failure_path = directory / "failure.json"
        if profile_path.exists():
            value = json.loads(profile_path.read_text())
        elif failure_path.exists():
            value = json.loads(failure_path.read_text())
        else:
            value = {"status": "missing", "setting": setting}
        profiles[setting] = value

    successful = [p for p in profiles.values() if p.get("status") == "completed"]
    encoder_hashes = {p["initial_encoder_sha256"] for p in successful}
    source_hashes = {p["source_sequence_sha256_120_steps"] for p in successful}
    if len(encoder_hashes) > 1:
        raise AssertionError("Successful runs did not start from identical encoder weights.")
    if len(source_hashes) > 1:
        raise AssertionError("Successful runs did not see the same 120-step source-image sequence.")

    reference = profiles["ijepa_4layer"]
    reference_ok = reference.get("status") == "completed"
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
        if p.get("status") == "completed" and reference_ok:
            for name, key in (
                ("relative_gpu_time", "gpu_milliseconds_mean"),
                ("relative_wall_time", "wall_seconds_mean"),
                ("relative_pixels", "input_pixels_per_source"),
                ("relative_patch_tokens", "patch_tokens_per_source"),
                ("relative_peak_vram", "peak_gpu_memory_allocated_gib"),
                ("relative_profiled_flops", "profiler_estimated_flops_per_step"),
            ):
                numerator = p.get(key)
                denominator = reference.get(key)
                row[name] = (
                    float(numerator) / float(denominator)
                    if numerator is not None and denominator not in (None, 0)
                    else None
                )
        records.append(row)

    if records:
        write_csv(records, root / "comparison.csv")
    result = {
        "purpose": "compute/workload fairness diagnostic only; no downstream performance used",
        "successful_initial_encoder_hash_identical": len(encoder_hashes) <= 1,
        "successful_source_sequence_hash_identical": len(source_hashes) <= 1,
        "profiles": profiles,
        "rows": records,
    }
    atomic_json_dump(result, root / "comparison.json")

    lines = [
        "# I-JEPA vs LeJEPA Fairness Diagnostic",
        "",
        "This diagnostic measures training workload only. It does not use downstream validation or test performance.",
        "",
        "| Setting | Status | Encoder params | Aux params | Pixels/source | Patch tokens/source | GPU ms/step | Wall s/step | Images/s | Peak VRAM GiB | Relative GPU | Relative wall |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        def f(key, digits=3):
            value = r.get(key)
            return "—" if value is None else f"{float(value):.{digits}f}"
        lines.append(
            f"| {r['setting']} | {r['status']} | "
            f"{r.get('encoder_parameters') or '—'} | {r.get('auxiliary_trainable_parameters') or '—'} | "
            f"{f('input_pixels_per_source',0)} | {f('patch_tokens_per_source',0)} | "
            f"{f('gpu_ms_mean')} | {f('wall_seconds_mean',4)} | {f('source_images_per_second_mean',2)} | "
            f"{f('peak_gpu_memory_allocated_gib',2)} | {f('relative_gpu_time',2)} | {f('relative_wall_time',2)} |"
        )
    lines += [
        "",
        f"- Identical initial encoder weights across successful runs: **{len(encoder_hashes) <= 1}**",
        f"- Identical 120-step PanNuke source sequence across successful runs: **{len(source_hashes) <= 1}**",
        "- LeJEPA 2G+4L is an intermediate workload diagnostic, not a claim that four local views match I-JEPA's four target blocks.",
        "- LeJEPA 2G+6L is the native-view main workload setting.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return result
