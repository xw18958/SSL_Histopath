from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

class DINOv3Views(Dataset):
    """Unlabelled PanNuke samples with DINOv3's 2-global/8-local augmentation layout."""

    def __init__(self, base: Dataset, config: dict[str, Any]) -> None:
        self.base = base
        method = config["method"]
        views = method["views"]
        aug = method["augmentations"]
        global_size = int(views["global_size"])
        local_size = int(views["local_size"])
        flip_p = float(aug["horizontal_flip_p"])

        self.global_geometry = v2.Compose(
            [
                v2.ToImage(),
                v2.RandomResizedCrop(
                    (global_size, global_size),
                    scale=tuple(float(x) for x in views["global_scale"]),
                    interpolation=v2.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                v2.RandomHorizontalFlip(p=flip_p),
            ]
        )
        self.local_geometry = v2.Compose(
            [
                v2.ToImage(),
                v2.RandomResizedCrop(
                    (local_size, local_size),
                    scale=tuple(float(x) for x in views["local_scale"]),
                    interpolation=v2.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                v2.RandomHorizontalFlip(p=flip_p),
            ]
        )
        jitter = lambda: v2.RandomApply(
            [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
            p=float(aug["color_jitter_p"]),
        )
        gray = lambda: v2.RandomGrayscale(p=float(aug["grayscale_p"]))
        self.global_one_photo = v2.Compose(
            [
                jitter(),
                gray(),
                v2.RandomApply([v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=float(aug["global1_blur_p"])),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        self.global_two_photo = v2.Compose(
            [
                jitter(),
                gray(),
                v2.RandomApply([v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=float(aug["global2_blur_p"])),
                v2.RandomSolarize(threshold=128, p=float(aug["global2_solarize_p"])),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        self.local_photo = v2.Compose(
            [
                jitter(),
                gray(),
                v2.RandomApply([v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=float(aug["local_blur_p"])),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        self.local_count = int(views["local_count"])

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image, fold, sample_index = self.base[index]
        if image.dtype != torch.uint8 or tuple(image.shape) != (3, 256, 256):
            raise RuntimeError(f"Unexpected PanNuke image: dtype={image.dtype}, shape={tuple(image.shape)}")
        g1 = self.global_one_photo(self.global_geometry(image))
        g2 = self.global_two_photo(self.global_geometry(image))
        local = torch.stack([self.local_photo(self.local_geometry(image)) for _ in range(self.local_count)])
        return {
            "global_views": torch.stack((g1, g2)),
            "local_views": local,
            "fold": int(fold),
            "sample_index": int(sample_index),
        }

