"""MAE pretraining adapted to the project's *identical* randomly initialized CLIP ViT-B/32.

Core MAE algorithm follows facebookresearch/mae/models_mae.py: random per-image
75% masking after patch/position embedding; an encoder receives visible tokens
only; an 8-layer 512-dim decoder reconstructs RGB pixels; loss is on masked
patches alone. The encoder is deliberately the SAME FreshPLIPVisionEncoder
used by LeJEPA/Simplex; this is a matched-architecture MAE comparison, not an
unaltered official ViT-B/16/ImageNet pretraining reproduction.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from pannuke_ssl.models import FreshPLIPVisionEncoder, normalize_clip


def fixed_2d_sincos(grid_size: int, dim: int) -> torch.Tensor:
    """Original MAE 2D sin/cos coordinates (CLS position is all zero)."""
    if dim % 4:
        raise ValueError("Positional embedding dimension must be divisible by 4")
    axis = torch.arange(grid_size, dtype=torch.float32)
    gy, gx = torch.meshgrid(axis, axis, indexing="ij")
    half_axis = dim // 4
    freq = 1.0 / (10000 ** (torch.arange(half_axis, dtype=torch.float32) / half_axis))
    def embed(p):
        z = p.flatten().unsqueeze(-1) * freq
        return torch.cat((z.sin(), z.cos()), dim=-1)
    patches = torch.cat((embed(gx), embed(gy)), dim=-1)
    return torch.cat((torch.zeros(1, dim), patches), dim=0).unsqueeze(0)


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    b, c, h, w = images.shape
    if c != 3 or h != w or h % patch_size:
        raise ValueError("Expected square RGB images divisible by patch_size")
    g = h // patch_size
    return images.reshape(b, 3, g, patch_size, g, patch_size).permute(0, 2, 4, 3, 5, 1).reshape(b, g * g, patch_size**2 * 3)


def random_masking(batch: int, patches: int, ratio: float, device: torch.device):
    """Official MAE shuffling/mask indexing; zero means visible."""
    if not 0 < ratio < 1:
        raise ValueError("mask_ratio must lie strictly between 0 and 1")
    n_keep = int(patches * (1 - ratio))
    if n_keep < 1 or n_keep >= patches:
        raise ValueError("Mask ratio would leave zero or all patches visible")
    ids_shuffle = torch.rand(batch, patches, device=device).argsort(dim=1)
    ids_restore = ids_shuffle.argsort(dim=1)
    ids_keep = ids_shuffle[:, :n_keep]
    mask = torch.ones(batch, patches, device=device)
    mask[:, :n_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    return ids_keep, ids_restore, mask


class MAEDataset(Dataset):
    """Unlabeled, single-view MAE image augmentation: RRC + horizontal flip."""
    def __init__(self, base: Dataset, image_size: int, min_crop: float) -> None:
        self.base = base
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(
                image_size, scale=(min_crop, 1.0), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, fold, sample_index = self.base[index]
        if image.dtype != torch.uint8 or tuple(image.shape) != (3, 256, 256):
            raise RuntimeError(f"Unexpected PanNuke image {image.dtype}, {tuple(image.shape)}")
        return {"image": self.transform(image), "fold": int(fold), "sample_index": int(sample_index)}


class Step:
    def __init__(self, loss, metrics):
        self.loss, self.metrics = loss, metrics


class StandardMAE(nn.Module):
    def __init__(self, c: dict[str, Any], device: torch.device):
        super().__init__()
        self.config, self.device = c, device
        method = c["method"]
        self.mask_ratio = float(method["objective"]["mask_ratio"])
        self.norm_pix_loss = bool(method["objective"]["norm_pix_loss"])
        self.backbone = FreshPLIPVisionEncoder(c["backbone"]["config_dir"], 256).to(device)
        if (self.backbone.num_patches, self.backbone.hidden_size, self.backbone.patch_size) != (64, 768, 32):
            raise RuntimeError("MAE must use the project's matched fresh PLIP ViT-B/32")
        decoder = method["decoder"]
        d = int(decoder["embed_dim"])
        depth = int(decoder["depth"])
        heads = int(decoder["num_heads"])
        if (d, depth, heads) != (512, 8, 16):
            raise ValueError("Faithful MAE decoder requires 512 width / 8 blocks / 16 heads")
        self.decoder_embed = nn.Linear(768, d)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
        self.register_buffer("decoder_pos_embed", fixed_2d_sincos(8, d), persistent=True)
        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d, nhead=heads, dim_feedforward=d * 4,
                dropout=0.0, activation="gelu", layer_norm_eps=1e-6,
                batch_first=True, norm_first=True,
            )
            for _ in range(depth)
        ])
        self.decoder_norm = nn.LayerNorm(d, eps=1e-6)
        self.decoder_pred = nn.Linear(d, 3 * 32 * 32)
        self._init_decoder()
        self.to(device)

    def _init_decoder(self) -> None:
        nn.init.normal_(self.mask_token, std=0.02)
        for m in [self.decoder_embed, *self.decoder_blocks.modules(), self.decoder_norm, self.decoder_pred]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)

    @property
    def encoder(self) -> FreshPLIPVisionEncoder:
        # Full unmasked final patch tokens [B,64,768], exactly as LeJEPA.
        return self.backbone

    def wrap_dataset(self, base: Dataset):
        return MAEDataset(base, 256, float(self.config["method"]["augmentations"]["min_crop_scale"]))

    def train_mode(self):
        self.train()

    def optimizer_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def batch_size(self, batch):
        return int(batch["image"].shape[0])

    def encode_visible(self, images: torch.Tensor, ids_keep: torch.Tensor):
        """CLIP ViT patch and position embedding, then ONLY selected patches."""
        vision = self.backbone.vision_backbone
        emb = vision.embeddings
        x = emb.patch_embedding(normalize_clip(images)).flatten(2).transpose(1, 2)
        if x.shape[1:] != (64, 768):
            raise RuntimeError("Unexpected patch embedding grid")
        # Learned CLIP positions remain shared with the LeJEPA architecture.
        x = x + emb.position_embedding.weight[1:].unsqueeze(0)
        x = x.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        cls = emb.class_embedding.to(x.dtype).view(1, 1, -1)
        cls = (cls + emb.position_embedding.weight[:1].unsqueeze(0)).expand(images.shape[0], -1, -1)
        x = vision.pre_layrnorm(torch.cat([cls, x], dim=1))
        x = vision.encoder(inputs_embeds=x, return_dict=True).last_hidden_state
        return vision.post_layernorm(x)

    def decode(self, latent: torch.Tensor, ids_restore: torch.Tensor):
        x = self.decoder_embed(latent)
        b, l, d = x.shape
        num_mask = ids_restore.shape[1] + 1 - l
        if num_mask < 1:
            raise RuntimeError("Decoder was passed no masked patches")
        new_tokens = self.mask_token.expand(b, num_mask, -1)
        patches = torch.cat((x[:, 1:], new_tokens), dim=1)
        patches = patches.gather(1, ids_restore.unsqueeze(-1).expand(-1, -1, d))
        x = torch.cat((x[:, :1], patches), dim=1) + self.decoder_pos_embed.to(x.dtype)
        for layer in self.decoder_blocks:
            x = layer(x)
        return self.decoder_pred(self.decoder_norm(x))[:, 1:]

    def reconstruction_loss(self, images: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor):
        target = patchify(images, 32).float()
        if self.norm_pix_loss:
            mean = target.mean(-1, keepdim=True)
            var = target.var(-1, keepdim=True)  # default unbiased=True, per official implementation
            target = (target - mean) / torch.sqrt(var + 1e-6)
        mse = (pred.float() - target).square().mean(dim=-1)
        return (mse * mask).sum() / mask.sum()

    def forward(self, images: torch.Tensor):
        b = int(images.shape[0])
        ids_keep, ids_restore, mask = random_masking(b, 64, self.mask_ratio, images.device)
        latent = self.encode_visible(images, ids_keep)
        pred = self.decode(latent, ids_restore)
        loss = self.reconstruction_loss(images, pred, mask)
        return loss, pred, mask

    def training_step(self, batch, *, bf16: bool):
        images = batch["image"].to(self.device, dtype=torch.float32, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and self.device.type == "cuda"):
            loss, pred, mask = self(images)
        return Step(loss, {
            "masked_pixel_mse": float(loss.detach()),
            "mask_ratio_actual": float(mask.float().mean()),
            "reconstructed_patches": float(pred.shape[1]),
        })

    def build_optimizer(self):
        o = self.config["method"]["optimizer"]
        decay, no_decay = [], []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim <= 1 or name.endswith(".bias") or "position_embedding" in name or "mask_token" in name:
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        return torch.optim.AdamW(
            [{"params": decay}, {"params": no_decay, "weight_decay": 0., "_no_weight_decay": True}],
            lr=float(o["peak_lr"]), weight_decay=float(o["weight_decay"]),
            betas=(0.9, 0.95),
        )

    def schedule(self, step: int, total: int):
        o = self.config["method"]["optimizer"]
        p = max(0., min(1., step / max(1, total - 1)))
        warm = float(o["warmup_fraction"])
        peak = float(o["peak_lr"])
        final = peak * float(o["final_lr_ratio"])
        lr = peak * p / warm if p < warm else final + .5 * (peak - final) * (1 + math.cos(math.pi * (p - warm) / (1 - warm)))
        return {"lr": lr, "weight_decay": float(o["weight_decay"])}

    def after_optimizer_step(self, step, total):
        return {}

    def training_metadata(self):
        return {"mae_mask_ratio": self.mask_ratio, "mae_norm_pix_loss": self.norm_pix_loss,
                "mae_matched_backbone": "fresh CLIP ViT-B/32, 256px, learned CLIP positions",
                "mae_decoder": "512d 8 blocks 16 heads; discarded for downstream"}
