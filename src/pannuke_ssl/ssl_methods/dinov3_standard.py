from __future__ import annotations

import copy
import math
import random
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from pannuke_ssl.models import FreshPLIPVisionEncoder, update_ema
from .dinov3_core import (
    DINOHead,
    KoLeoLoss,
    MaskingGenerator,
    SharedPLIPDINOBackbone,
    cross_view_dino_loss,
    sinkhorn_knopp,
)
from .dinov3_data import DINOv3Views

class Step:
    def __init__(self, loss: torch.Tensor, metrics: dict[str, float]):
        self.loss = loss
        self.metrics = metrics


class StandardDINOv3(nn.Module):
    """DINOv3 base SSL objective adapted only to the experiment's shared PLIP backbone."""

    def __init__(self, c: dict[str, Any], device: torch.device) -> None:
        super().__init__()
        self.config = c
        self.device = device
        method = c["method"]
        objective = method["objective"]

        base = FreshPLIPVisionEncoder(c["backbone"]["config_dir"], 256).to(device)
        heads = int(base.model.config.num_attention_heads)
        if (base.num_patches, base.hidden_size, base.patch_size, heads) != (64, 768, 32, 12):
            raise RuntimeError("Unexpected shared encoder for DINOv3")
        self.student = SharedPLIPDINOBackbone(base).to(device)
        self.teacher = copy.deepcopy(self.student).to(device).eval()
        self.teacher.requires_grad_(False)

        self.student_dino_head = DINOHead(
            768,
            int(objective["dino_prototypes"]),
            hidden_dim=int(objective["head_hidden_dim"]),
            bottleneck_dim=int(objective["head_bottleneck_dim"]),
            nlayers=int(objective["head_layers"]),
        ).to(device)
        self.student_ibot_head = DINOHead(
            768,
            int(objective["ibot_prototypes"]),
            hidden_dim=int(objective["head_hidden_dim"]),
            bottleneck_dim=int(objective["head_bottleneck_dim"]),
            nlayers=int(objective["head_layers"]),
        ).to(device)
        self.teacher_dino_head = copy.deepcopy(self.student_dino_head).to(device).eval()
        self.teacher_ibot_head = copy.deepcopy(self.student_ibot_head).to(device).eval()
        self.teacher_dino_head.requires_grad_(False)
        self.teacher_ibot_head.requires_grad_(False)

        self.koleo = KoLeoLoss()
        self.mask_generator = MaskingGenerator((8, 8), min_num_patches=4, min_aspect=0.3)
        self._teacher_temp = float(method["teacher"]["temperature_start"])
        self._freeze_last_layer = False

    @property
    def encoder(self) -> nn.Module:
        # DINOv3 evaluates the EMA teacher; downstream still mean-pools final patch tokens.
        return self.teacher

    def wrap_dataset(self, base: Dataset) -> Dataset:
        return DINOv3Views(base, self.config)

    def train_mode(self) -> None:
        self.student.train()
        self.student_dino_head.train()
        self.student_ibot_head.train()
        self.teacher.eval()
        self.teacher_dino_head.eval()
        self.teacher_ibot_head.eval()

    def optimizer_parameters(self) -> list[nn.Parameter]:
        modules = (self.student, self.student_dino_head, self.student_ibot_head)
        return [p for module in modules for p in module.parameters() if p.requires_grad]

    def batch_size(self, batch: dict[str, torch.Tensor]) -> int:
        return int(batch["global_views"].shape[0])

    def _sample_masks(self, entries: int, tokens: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        cfg = self.config["method"]["objective"]
        probability = float(cfg["mask_sample_probability"])
        low, high = (float(x) for x in cfg["mask_ratio"])
        n_masked_samples = int(entries * probability)
        bounds = torch.linspace(low, high, n_masked_samples + 1).tolist() if n_masked_samples else []
        masks: list[torch.Tensor] = []
        for i in range(n_masked_samples):
            masks.append(self.mask_generator(int(tokens * bounds[i + 1])))
        masks.extend(torch.zeros(tokens, dtype=torch.bool) for _ in range(entries - n_masked_samples))
        random.shuffle(masks)
        stacked = torch.stack(masks).to(self.device, non_blocking=True)
        count = stacked.sum(dim=1).clamp_min(1)
        weights = (1.0 / count).unsqueeze(1).expand_as(stacked)[stacked]
        return stacked, weights, float(stacked.float().mean())

    def training_step(self, batch: dict[str, torch.Tensor], *, bf16: bool) -> Step:
        global_views = batch["global_views"].to(self.device, non_blocking=True)
        local_views = batch["local_views"].to(self.device, non_blocking=True)
        if global_views.shape[1] != 2 or local_views.shape[1] != 8:
            raise RuntimeError("Standard DINOv3 must remain 2 global + 8 local crops")
        b = global_views.shape[0]
        global_images = torch.cat([global_views[:, i] for i in range(2)], dim=0)
        local_images = torch.cat([local_views[:, i] for i in range(8)], dim=0)
        masks, mask_weights, mask_fraction = self._sample_masks(2 * b, 64)
        objective = self.config["method"]["objective"]
        student_temp = float(objective["student_temperature"])

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            with torch.no_grad():
                teacher_cls, teacher_patches = self.teacher.features(global_images)
                teacher_cls_logits = self.teacher_dino_head(teacher_cls).reshape(2, b, -1)
                teacher_cls_probs = sinkhorn_knopp(
                    teacher_cls_logits.reshape(2 * b, -1),
                    self._teacher_temp,
                ).reshape(2, b, -1)

            student_global_cls, student_global_patches = self.student.features(global_images, masks)
            student_local_cls, _ = self.student.features(local_images)
            student_global_logits = self.student_dino_head(student_global_cls).reshape(2, b, -1)
            student_local_logits = self.student_dino_head(student_local_cls).reshape(8, b, -1)

            dino_global = cross_view_dino_loss(
                student_global_logits,
                teacher_cls_probs,
                student_temperature=student_temp,
                ignore_diagonal=True,
            )
            dino_local = cross_view_dino_loss(
                student_local_logits,
                teacher_cls_probs,
                student_temperature=student_temp,
                ignore_diagonal=False,
            )

            # DINOv3 weights global/local terms by their number of valid crop pairs.
            global_terms = 2 * (2 - 1)
            local_terms = 2 * 8
            dino_global_scale = global_terms / (global_terms + local_terms)
            dino_local_scale = local_terms / (global_terms + local_terms)
            dino_loss = dino_global_scale * dino_global + dino_local_scale * dino_local

            masked_student = student_global_patches[masks]
            masked_teacher = teacher_patches[masks]
            if masked_student.numel() == 0:
                raise RuntimeError("DINOv3 iBOT produced no masked patches")
            student_patch_logits = self.student_ibot_head(masked_student)
            with torch.no_grad():
                teacher_patch_logits = self.teacher_ibot_head(masked_teacher)
                teacher_patch_probs = sinkhorn_knopp(teacher_patch_logits, self._teacher_temp)
            token_loss = -(teacher_patch_probs * F.log_softmax(student_patch_logits.float() / student_temp, dim=-1)).sum(dim=-1)
            ibot_loss = (token_loss * mask_weights).sum() / (2 * b)

            global_cls_by_view = student_global_cls.reshape(2, b, -1)
            koleo_loss = sum(self.koleo(global_cls_by_view[i]) for i in range(2)) / 2.0

            loss = (
                float(objective["dino_loss_weight"]) * dino_loss
                + float(objective["ibot_loss_weight"]) * ibot_loss
                + float(objective["koleo_loss_weight"]) * 2.0 * koleo_loss
            )

        return Step(
            loss,
            {
                "dino_global_loss": float(dino_global.detach()),
                "dino_local_loss": float(dino_local.detach()),
                "ibot_loss": float(ibot_loss.detach()),
                "koleo_loss": float(koleo_loss.detach()),
                "teacher_temperature": float(self._teacher_temp),
                "masked_patch_fraction": mask_fraction,
                "logical_views": 10.0,
            },
        )

    def build_optimizer(self) -> torch.optim.Optimizer:
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for module in (self.student, self.student_dino_head, self.student_ibot_head):
            for name, p in module.named_parameters():
                if not p.requires_grad:
                    continue
                if p.ndim == 1 or name.endswith("bias") or "mask_token" in name:
                    no_decay.append(p)
                else:
                    decay.append(p)
        optim = self.config["method"]["optimizer"]
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": float(optim["weight_decay"])},
                {"params": no_decay, "weight_decay": 0.0, "_no_weight_decay": True},
            ],
            lr=float(optim["peak_lr"]),
            betas=(0.9, 0.999),
        )

    def schedule(self, step: int, total: int) -> dict[str, float]:
        optim = self.config["method"]["optimizer"]
        teacher = self.config["method"]["teacher"]
        p = min(max(step / max(1, total - 1), 0.0), 1.0)

        warm = float(optim["warmup_fraction"])
        peak = float(optim["peak_lr"])
        minimum = float(optim["min_lr"])
        if p < warm:
            lr = peak * (p / max(warm, 1e-12))
        else:
            q = (p - warm) / max(1.0 - warm, 1e-12)
            lr = minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * q))

        wd0 = float(optim["weight_decay"])
        wd1 = float(optim["final_weight_decay"])
        wd = wd1 - 0.5 * (wd1 - wd0) * (1.0 + math.cos(math.pi * p))

        temp_warm = float(teacher["temperature_warmup_fraction"])
        t0 = float(teacher["temperature_start"])
        t1 = float(teacher["temperature_end"])
        self._teacher_temp = t0 + (t1 - t0) * min(p / max(temp_warm, 1e-12), 1.0)

        # DINOv3 freezes projection-head last layers briefly at the start.
        steps_per_epoch = math.ceil(int(self.config["data"]["expected_ssl_images"]) / int(self.config["training"]["batch_size"]))
        freeze_steps = int(optim["freeze_last_layer_epochs"]) * steps_per_epoch
        freeze = step < freeze_steps
        if freeze != self._freeze_last_layer:
            self.student_dino_head.last_layer.requires_grad_(not freeze)
            self.student_ibot_head.last_layer.requires_grad_(not freeze)
            self._freeze_last_layer = freeze
        return {"lr": lr, "weight_decay": wd}

    def after_optimizer_step(self, step: int, total: int) -> dict[str, float]:
        teacher_cfg = self.config["method"]["teacher"]
        p = min(max(step / max(1, total - 1), 0.0), 1.0)
        start = float(teacher_cfg["momentum_start"])
        end = float(teacher_cfg["momentum_end"])
        momentum = end - 0.5 * (end - start) * (1.0 + math.cos(math.pi * p))
        update_ema(self.student, self.teacher, momentum)
        update_ema(self.student_dino_head, self.teacher_dino_head, momentum)
        update_ema(self.student_ibot_head, self.teacher_ibot_head, momentum)
        return {"ema_momentum": momentum}
