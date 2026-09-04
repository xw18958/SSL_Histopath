from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PanNukeImageDataset, loader_kwargs
from .degradations import defocus_blur, resolution_degrade
from .parquet import build_source_index, preload_images, read_metadata, verify_records
from .utils import atomic_json_dump, seed_everything


@dataclass(frozen=True)
class PiecewiseFit:
    breakpoint: float
    breakpoint_index: int
    slope_before: float
    slope_after: float
    intercept: float
    sse: float


def fit_piecewise_hinge(
    strengths: Iterable[float],
    values: Iterable[float],
    *,
    minimum_points_per_segment: int = 4,
    validate_shape: bool = True,
) -> PiecewiseFit:
    x = np.asarray(list(strengths), dtype=np.float64)
    y = np.asarray(list(values), dtype=np.float64)
    if x.ndim != 1 or x.size != y.size or np.any(np.diff(x) <= 0):
        raise ValueError("Strengths must be a strictly increasing 1-D sequence aligned with VIF")
    candidates = range(minimum_points_per_segment - 1, x.size - minimum_points_per_segment)
    fits: list[PiecewiseFit] = []
    for index in candidates:
        breakpoint = x[index]
        design = np.column_stack((np.ones_like(x), x, np.maximum(0.0, x - breakpoint)))
        coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ coefficients
        fits.append(
            PiecewiseFit(
                breakpoint=float(breakpoint),
                breakpoint_index=index,
                slope_before=float(coefficients[1]),
                slope_after=float(coefficients[1] + coefficients[2]),
                intercept=float(coefficients[0]),
                sse=float(residual @ residual),
            )
        )
    if not fits:
        raise ValueError("Sweep has too few points for the requested piecewise fit")
    fit = min(fits, key=lambda candidate: candidate.sse)
    if validate_shape and not (
        fit.slope_before < 0
        and fit.slope_after > fit.slope_before
        and abs(fit.slope_after) < abs(fit.slope_before)
    ):
        raise RuntimeError(
            "VIF curve has no valid flattening information-loss boundary: "
            f"before={fit.slope_before:.6g}, after={fit.slope_after:.6g}"
        )
    return fit


def _vif(clean: torch.Tensor, degraded: torch.Tensor) -> torch.Tensor:
    try:
        import piq
    except ImportError as error:
        raise RuntimeError("Calibration requires piq; install requirements.txt") from error
    return piq.vif_p(clean.float(), degraded.float(), data_range=1.0, reduction="none")


def _calibration_loader(config: dict) -> DataLoader:
    source_index = build_source_index(Path(config["data_root"]))
    rows = read_metadata(Path(config["metadata_csv"]))
    verify_records(rows, source_index)
    if len(rows) != 2546:
        raise ValueError(f"Calibration requires all 2,546 balanced samples, got {len(rows)}")
    cache = preload_images(rows, source_index) if config.get("cache_in_ram", True) else None
    dataset = PanNukeImageDataset(rows, source_index, cache, include_label=True, include_key=True)
    return DataLoader(
        dataset,
        **loader_kwargs(int(config["batch_size"]), int(config["num_workers"]), shuffle=False),
    )


def _measure_family(
    loader: DataLoader,
    *,
    family: str,
    levels: list[float],
    device: torch.device,
    csv_writer: csv.DictWriter,
) -> tuple[list[float], float]:
    started = time.perf_counter()
    medians: list[float] = []
    for level in levels:
        values: list[float] = []
        for images, _, folds, indices in loader:
            clean = images.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
            if family == "defocus":
                degraded = defocus_blur(clean, level)
                physical_strength = level
                target_side: int | str = ""
            else:
                target_side = int(level)
                physical_strength = 256.0 / target_side
                degraded = resolution_degrade(clean, physical_strength)
            scores = _vif(clean, degraded).detach().cpu().numpy().reshape(-1)
            values.extend(float(value) for value in scores)
            for fold, sample_index, score in zip(folds.tolist(), indices.tolist(), scores, strict=True):
                csv_writer.writerow(
                    {
                        "sample_id": f"fold{fold}_{sample_index:04d}",
                        "fold": fold,
                        "sample_index": sample_index,
                        "degradation": family,
                        "strength": physical_strength,
                        "target_side": target_side,
                        "vif": f"{float(score):.10g}",
                    }
                )
        if len(values) != 2546:
            raise RuntimeError(f"Expected 2,546 VIF scores at {family}={level}, got {len(values)}")
        medians.append(float(np.median(values)))
        print(f"{family:10s} level={level:g} median_vif={medians[-1]:.6f}", flush=True)
    return medians, time.perf_counter() - started


def run_calibration(config: dict) -> dict:
    seed_everything(int(config["seed"]))
    if config.get("device", "cuda") == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA calibration was requested but is unavailable")
    device = torch.device(config.get("device", "cuda"))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = _calibration_loader(config)
    csv_path = output_dir / "vif_measurements.csv"
    total_started = time.perf_counter()
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "fold", "sample_index", "degradation", "strength", "target_side", "vif"],
        )
        writer.writeheader()
        defocus_levels = [float(value) for value in config["defocus_radii"]]
        resolution_sides = [float(value) for value in config["resolution_sides"]]
        defocus_vif, defocus_seconds = _measure_family(
            loader, family="defocus", levels=defocus_levels, device=device, csv_writer=writer
        )
        resolution_vif, resolution_seconds = _measure_family(
            loader, family="resolution", levels=resolution_sides, device=device, csv_writer=writer
        )
    min_points = int(config.get("minimum_points_per_segment", 4))
    defocus_fit = fit_piecewise_hinge(defocus_levels, defocus_vif, minimum_points_per_segment=min_points)
    resolution_factors = [256.0 / side for side in resolution_sides]
    defocus_fit_dict = asdict(defocus_fit)
    resolution_fit = fit_piecewise_hinge(
        resolution_factors, resolution_vif, minimum_points_per_segment=min_points
    )
    resolution_fit_dict = asdict(resolution_fit)
    selected_side = int(round(256.0 / resolution_fit.breakpoint))
    result = {
        "seed": int(config["seed"]),
        "calibration_samples": 2546,
        "device": str(device),
        "vif": {"implementation": "piq.vif_p", "data_range": 1.0, "dtype": "float32", "reference": "clean"},
        "defocus": {
            "selected_radius": defocus_fit.breakpoint,
            "levels": defocus_levels,
            "median_vif": defocus_vif,
            "fit": defocus_fit_dict,
            "seconds": defocus_seconds,
        },
        "resolution": {
            "selected_factor": resolution_fit.breakpoint,
            "selected_target_side": selected_side,
            "target_sides": [int(side) for side in resolution_sides],
            "factors": resolution_factors,
            "median_vif": resolution_vif,
            "fit": resolution_fit_dict,
            "seconds": resolution_seconds,
        },
        "total_seconds": time.perf_counter() - total_started,
        "measurements_csv": str(csv_path),
    }
    atomic_json_dump(result, output_dir / "degradation_endpoints.json")
    return result
