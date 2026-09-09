"""Validation-only, B0-budget-equivalent Self-Flow hyperparameter tuning."""
from __future__ import annotations

import itertools
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PanNukeImageDataset, build_ssl_loader, loader_kwargs
from .models import update_ema
from .parquet import build_source_index, preload_images, read_metadata, verify_records
from .probe import _fit_probe
from .self_flow import self_flow_objective
from .self_flow_training import build_self_flow_models, load_self_flow_checkpoint
from .training import _learning_rate, _make_optimizer, _save_final_student, _weight_decay
from .utils import atomic_json_dump, seed_everything, write_csv


def _build_train_val_loaders(config: dict[str, Any]) -> dict[str, DataLoader]:
    """Build only train/validation loaders; the test split is never instantiated."""
    source_index = build_source_index(Path(config["data_root"]))
    rows = read_metadata(Path(config["metadata_csv"]))
    verify_records(rows, source_index)
    selected = {split: [row for row in rows if row["split"] == split] for split in ("train", "val")}
    expected = {"train": 2052, "val": 247}
    if {split: len(values) for split, values in selected.items()} != expected:
        raise ValueError("Unexpected train/validation split sizes for Self-Flow tuning")
    selected_rows = [row for values in selected.values() for row in values]
    cache = preload_images(selected_rows, source_index) if bool(config.get("cache_in_ram", True)) else None
    workers = int(config.get("num_workers", 8))
    return {
        split: DataLoader(
            PanNukeImageDataset(part, source_index, cache, include_label=True),
            **loader_kwargs(int(config["batch_size"]), workers, shuffle=(split == "train")),
        )
        for split, part in selected.items()
    }


def _cosine_ema(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return end
    progress = min(1.0, step / (total_steps - 1))
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


def _tune_pretrain(config: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Self-Flow tuning requires CUDA")
    seed_everything(int(config["seed"]))
    torch.set_num_threads(4)
    device = torch.device("cuda")
    train_config = config["train"]
    loader, _ = build_ssl_loader(
        config["data_root"],
        batch_size=int(train_config["batch_size"]),
        num_workers=int(train_config["num_workers"]),
        cache_in_ram=bool(train_config.get("cache_in_ram", True)),
    )
    if len(loader.dataset) != 7901 or getattr(loader.dataset, "include_label", True):
        raise RuntimeError("Self-Flow tuning SSL must see exactly 7,901 unlabeled images")
    student, teacher = build_self_flow_models(config, device)
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = _make_optimizer(trainable, float(train_config["learning_rate"]), float(train_config["weight_decay"]))
    accumulation_steps = int(train_config.get("effective_batch_size", 128)) // int(train_config["batch_size"])
    epochs = int(train_config["epochs"])
    steps_per_epoch = math.ceil(len(loader) / accumulation_steps)
    total_steps = epochs * steps_per_epoch
    global_step = 0
    history: list[dict[str, float]] = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, epochs + 1):
        student.train()
        teacher.eval()
        totals = {"total": 0.0, "flow": 0.0, "representation": 0.0}
        seen = 0
        for batch_index, uint8_images in enumerate(loader):
            images = uint8_images.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                total_loss, metrics = self_flow_objective(
                    student,
                    teacher,
                    images,
                    mask_ratio=float(config["model"]["mask_ratio"]),
                    representation_weight=float(config["model"]["representation_weight"]),
                    student_rep_layer=int(config["model"]["student_rep_layer"]),
                    teacher_rep_layer=int(config["model"]["teacher_rep_layer"]),
                )
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"Non-finite tuning loss at epoch={epoch}, batch={batch_index}")
            (total_loss / accumulation_steps).backward()
            is_update = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader)
            if is_update:
                norm = torch.nn.utils.clip_grad_norm_(trainable, float(train_config["gradient_clip_norm"]))
                if not torch.isfinite(norm):
                    raise FloatingPointError("Non-finite Self-Flow tuning gradient norm")
                lr = _learning_rate(train_config, global_step, total_steps)
                wd = _weight_decay(train_config, global_step, total_steps)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                    group["weight_decay"] = wd
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update_ema(
                    student,
                    teacher,
                    _cosine_ema(float(train_config["ema_start"]), 1.0, global_step, total_steps),
                )
                global_step += 1
            count = int(images.shape[0])
            seen += count
            for name in totals:
                totals[name] += float(metrics[name].detach()) * count
        if seen != 7901:
            raise RuntimeError(f"Expected 7,901 tuning samples, saw {seen}")
        history.append({"epoch": epoch, **{name: value / seen for name, value in totals.items()}})
    output_dir = Path(config["output_dir"])
    checkpoint = output_dir / "checkpoints" / "best.pt"
    _save_final_student(
        checkpoint,
        student=student,
        config=config,
        epoch=epochs,
        history=history,
        selection={"metric": "validation_macro_f1_probe", "test_split_used_for_selection": False},
    )
    return {"checkpoint": str(checkpoint), "history": history, "epochs": epochs}


def _probe_train_val(config: dict[str, Any], checkpoint: str, output_dir: Path, seed: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    loaders = _build_train_val_loaders(config)
    device = torch.device("cuda")
    encoder = load_self_flow_checkpoint(checkpoint, device)
    encoder.eval()
    features: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    with torch.inference_mode():
        for split in ("train", "val"):
            values: list[torch.Tensor] = []
            labels: list[torch.Tensor] = []
            for uint8_images, batch_labels in loaders[split]:
                images = uint8_images.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    values.append(encoder(images).mean(dim=1).float().cpu())
                labels.append(batch_labels.long())
            features[split] = (torch.cat(values), torch.cat(labels))
    np.savez_compressed(
        output_dir / "train_val_features.npz",
        train_features=features["train"][0].numpy(),
        train_labels=features["train"][1].numpy(),
        val_features=features["val"][0].numpy(),
        val_labels=features["val"][1].numpy(),
    )
    trials: list[dict[str, Any]] = []
    for index, (learning_rate, weight_decay) in enumerate(itertools.product((0.001, 0.003, 0.01), (0.0, 0.0001))):
        trial = _fit_probe(
            features,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            maximum_epochs=5,
            patience=5,
            seed=seed + index,
        )
        trials.append({key: value for key, value in trial.items() if key not in ("state", "history")})
    write_csv(trials, output_dir / "probe_leaderboard.csv")
    return max(trials, key=lambda row: float(row["val_macro_f1"]))


def _transfer_selected_hparams(config_path: Path, learning_rate: float, gamma: float) -> dict[str, Any]:
    text = config_path.read_text(encoding="utf-8")
    text, count_lr = re.subn(r"(?m)^  learning_rate: .*\n", f"  learning_rate: {learning_rate:.10g}\n", text, count=1)
    text, count_gamma = re.subn(r"(?m)^  representation_weight: .*\n", f"  representation_weight: {gamma:.10g}\n", text, count=1)
    if count_lr != 1 or count_gamma != 1:
        raise RuntimeError("Could not transfer exactly the selected Self-Flow hyperparameters")
    config_path.write_text(text, encoding="utf-8")
    from .config import load_yaml
    return load_yaml(config_path)


def run_self_flow_tuning(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["output_dir"])
    if (output_dir / "tuning_summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing Self-Flow tuning: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    from .config import load_yaml
    base = load_yaml(config["base_config"])
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    values = list(itertools.product(config["parameters"]["learning_rate"], config["parameters"]["gamma"]))
    for index, (learning_rate, gamma) in enumerate(values, start=1):
        trial_dir = output_dir / f"trial_{index:02d}"
        trial_config = load_yaml(config["base_config"])
        trial_config["seed"] = int(base["seed"]) + index
        trial_config["train"]["epochs"] = int(config["pretrain_epochs"])
        trial_config["train"]["learning_rate"] = float(learning_rate)
        trial_config["model"]["representation_weight"] = float(gamma)
        trial_config["output_dir"] = str(trial_dir / "pretrain")
        trial_started = time.perf_counter()
        row = {"trial": index, "learning_rate": float(learning_rate), "gamma": float(gamma), "seed": trial_config["seed"], "status": "running"}
        try:
            pretrain = _tune_pretrain(trial_config)
            best_probe = _probe_train_val(load_yaml(config["probe_config"]), pretrain["checkpoint"], trial_dir / "probe", trial_config["seed"])
            row.update({"status": "complete", "val_macro_f1": float(best_probe["val_macro_f1"]), "probe_learning_rate": best_probe["learning_rate"], "probe_weight_decay": best_probe["weight_decay"], "probe_best_epoch": best_probe["best_epoch"], "pretrain_checkpoint": pretrain["checkpoint"]})
        except (FloatingPointError, RuntimeError) as error:
            row.update({"status": "failed", "error": str(error), "val_macro_f1": -1.0})
        row["seconds"] = time.perf_counter() - trial_started
        rows.append(row)
        write_csv(rows, output_dir / "leaderboard.csv")
        print(json.dumps(row, sort_keys=True), flush=True)
    completed = [row for row in rows if row["status"] == "complete"]
    if not completed:
        raise RuntimeError("Every Self-Flow tuning trial failed")
    best = max(completed, key=lambda row: float(row["val_macro_f1"]))
    best_checkpoint = Path(best["pretrain_checkpoint"])
    for row in completed:
        checkpoint = Path(row["pretrain_checkpoint"])
        if checkpoint != best_checkpoint and checkpoint.exists():
            checkpoint.unlink()
    duration_config_path = Path(config["duration_config_path"])
    selected_config = _transfer_selected_hparams(duration_config_path, float(best["learning_rate"]), float(best["gamma"]))
    summary = {
        "method": config["method"],
        "selection_metric": config["selection_metric"],
        "best_trial": best,
        "trials": rows,
        "selected_learning_rate": best["learning_rate"],
        "selected_gamma": best["gamma"],
        "duration_config_path": str(duration_config_path),
        "final_config": selected_config,
        "seconds": time.perf_counter() - started,
    }
    atomic_json_dump(summary, output_dir / "tuning_summary.json")
    atomic_json_dump(selected_config, output_dir / "selected_self_flow_config.json")
    return summary
