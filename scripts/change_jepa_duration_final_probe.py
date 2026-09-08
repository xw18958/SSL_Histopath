"""B0-matched frozen clean-image linear probe for the selected Change-JEPA encoder."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pannuke_ssl.config import load_yaml
from pannuke_ssl.data import PanNukeImageDataset, loader_kwargs
from pannuke_ssl.parquet import build_source_index, preload_images, read_metadata, verify_records
from pannuke_ssl.probe import _evaluate, _fit_probe
from pannuke_ssl.training import load_student_checkpoint
from pannuke_ssl.utils import atomic_json_dump, seed_everything, write_csv

ROOT = Path("/raid1/xwan0900/SSL_proj")
ENCODER = ROOT / "outputs/change_jepa_duration_pilot/checkpoints/best.pt"
OUT = ROOT / "outputs/change_jepa_duration_pilot_final_probe"


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows_checked(config: dict) -> list[dict]:
    rows = read_metadata(Path(config["metadata_csv"]))
    if len(rows) != 2546 or len({(r["fold"], r["sample_index"]) for r in rows}) != 2546:
        raise AssertionError("Expected 2,546 unique balanced PanNuke probe records")
    if dict(Counter(r["split"] for r in rows)) != {"train": 2052, "val": 247, "test": 247}:
        raise AssertionError("Probe split counts differ from B0")
    counts = Counter((r["split"], r["class_id"]) for r in rows)
    for split, count in (("train", 108), ("val", 13), ("test", 13)):
        if [counts[(split, c)] for c in range(19)] != [count] * 19:
            raise AssertionError(f"{split} is not balanced exactly as B0")
    return rows


def setup(config: dict) -> None:
    if Path(config["encoder_checkpoint"]).resolve() != ENCODER:
        raise AssertionError("Unexpected encoder checkpoint")
    if Path(config["output_dir"]).resolve() != OUT:
        raise AssertionError("Unexpected probe output directory")
    if config["learning_rates"] != [0.001, 0.003, 0.01]:
        raise AssertionError("Learning-rate grid must match B0")
    if config["weight_decays"] != [0.0, 0.0001]:
        raise AssertionError("Weight-decay grid must match B0")
    if int(config["maximum_epochs"]) != 50 or int(config["early_stopping_patience"]) != 8:
        raise AssertionError("Probe epoch/patience settings must match B0")
    seed_everything(int(config["seed"]))


def extract_splits(config: dict, splits: tuple[str, ...]) -> dict[str, np.ndarray]:
    rows = rows_checked(config)
    selected = [r for r in rows if r["split"] in splits]
    index = build_source_index(Path(config["data_root"]))
    verify_records(rows, index)
    cache = preload_images(selected, index)
    encoder = load_student_checkpoint(config["plip_config_dir"], ENCODER, torch.device("cuda"))
    if encoder.training or any(p.requires_grad for p in encoder.parameters()):
        raise AssertionError("Encoder must be frozen for the final probe")

    values: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for split in splits:
            part = [r for r in selected if r["split"] == split]
            loader = DataLoader(
                PanNukeImageDataset(part, index, cache, include_label=True),
                **loader_kwargs(int(config["batch_size"]), int(config["num_workers"]), shuffle=False),
            )
            features, labels = [], []
            for images, y in loader:
                clean = images.to("cuda", dtype=torch.float32, non_blocking=True).div_(255.0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tokens = encoder(clean)
                if tokens.shape[1:] != (64, 768):
                    raise AssertionError(f"Unexpected token shape {tuple(tokens.shape)}")
                features.append(tokens.float().mean(dim=1).cpu())
                labels.append(y.long().cpu())
            values[f"{split}_features"] = torch.cat(features).numpy()
            values[f"{split}_labels"] = torch.cat(labels).numpy()
            values[f"{split}_keys"] = np.array([(r["fold"], r["sample_index"]) for r in part])
    return values


def extract_train_val(config: dict) -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing probe directory: {OUT}")
    OUT.mkdir(parents=True)
    values = extract_splits(config, ("train", "val"))
    np.savez_compressed(OUT / "train_val_features.npz", **values)
    checkpoint = torch.load(ENCODER, map_location="cpu", weights_only=False)
    atomic_json_dump(
        {
            "encoder_checkpoint_sha256": sha(ENCODER),
            "encoder_epoch": int(checkpoint["epoch"]),
            "clean_images": True,
            "pooling": "mean of 64 patch tokens",
            "test_accessed": False,
        },
        OUT / "encoder_lock.json",
    )
    atomic_json_dump(config, OUT / "resolved_config.json")
    print(json.dumps({"train": [2052, 768], "val": [247, 768]}, indent=2), flush=True)


def load_standardized_train_val(config: dict):
    lock = json.loads((OUT / "encoder_lock.json").read_text())
    if sha(ENCODER) != lock["encoder_checkpoint_sha256"]:
        raise AssertionError("Encoder checkpoint changed after feature extraction")
    with np.load(OUT / "train_val_features.npz") as data:
        train_x = torch.from_numpy(data["train_features"].copy())
        train_y = torch.from_numpy(data["train_labels"].copy())
        val_x = torch.from_numpy(data["val_features"].copy())
        val_y = torch.from_numpy(data["val_labels"].copy())
    mean = train_x.mean(0, keepdim=True)
    std = train_x.std(0, keepdim=True, unbiased=True).clamp_min(1e-6)
    features = {
        "train": ((train_x - mean) / std, train_y),
        "val": ((val_x - mean) / std, val_y),
    }
    return features, mean, std, lock


def tune(config: dict) -> None:
    if (OUT / "test_started.json").exists():
        raise FileExistsError("Test evaluation has already started")
    if (OUT / "selection.json").exists():
        raise FileExistsError("Probe has already been selected")
    features, mean, std, lock = load_standardized_train_val(config)
    best = None
    leaderboard, histories = [], []
    for lr in config["learning_rates"]:
        for wd in config["weight_decays"]:
            trial = _fit_probe(
                features,
                learning_rate=float(lr),
                weight_decay=float(wd),
                maximum_epochs=int(config["maximum_epochs"]),
                patience=int(config["early_stopping_patience"]),
                seed=int(config["seed"]),
            )
            row = {k: v for k, v in trial.items() if k not in ("state", "history")}
            leaderboard.append(row)
            histories.extend({"lr": lr, "weight_decay": wd, **r} for r in trial["history"])
            if best is None or trial["val_macro_f1"] > best["val_macro_f1"]:
                best = trial
    if best is None:
        raise RuntimeError("No linear-probe candidate completed")
    state = {
        "classifier": {k: v.cpu() for k, v in best["state"].items()},
        "feature_mean": mean,
        "feature_std": std,
    }
    torch.save(state, OUT / "best_linear_probe.pt")
    selection = {
        **{k: v for k, v in best.items() if k not in ("state", "history")},
        "selection_split": "val",
        "test_accessed": False,
        "encoder_checkpoint_sha256": lock["encoder_checkpoint_sha256"],
        "probe_checkpoint_sha256": sha(OUT / "best_linear_probe.pt"),
        "standardization_source": "train only",
    }
    write_csv(leaderboard, OUT / "probe_leaderboard.csv")
    write_csv(histories, OUT / "all_trial_metrics.csv")
    write_csv(best["history"], OUT / "selected_probe_metrics.csv")
    atomic_json_dump(selection, OUT / "selection.json")
    print(json.dumps(selection, indent=2), flush=True)


def test_once(config: dict) -> None:
    selection = json.loads((OUT / "selection.json").read_text())
    if sha(ENCODER) != selection["encoder_checkpoint_sha256"]:
        raise AssertionError("Encoder changed after probe selection")
    if sha(OUT / "best_linear_probe.pt") != selection["probe_checkpoint_sha256"]:
        raise AssertionError("Linear probe changed after selection")
    with (OUT / "test_started.json").open("x") as handle:
        json.dump({"test_evaluations": 1}, handle)

    values = extract_splits(config, ("test",))
    state = torch.load(OUT / "best_linear_probe.pt", map_location="cpu", weights_only=False)
    features = (torch.from_numpy(values["test_features"]) - state["feature_mean"]) / state["feature_std"]
    labels = torch.from_numpy(values["test_labels"])
    classifier = torch.nn.Linear(768, 19).cuda().eval()
    classifier.load_state_dict(state["classifier"])
    loss, metrics, predictions = _evaluate(classifier, features, labels, torch.device("cuda"))
    np.savez_compressed(
        OUT / "test_predictions.npz",
        labels=labels.numpy(), predictions=predictions, keys=values["test_keys"],
    )
    summary = {
        "evaluation_protocol": "B0-matched frozen clean-image linear probe",
        "test_evaluations": 1,
        "test_loss": loss,
        "test": metrics,
        "selection": selection,
        "correct_test_predictions": int((predictions == labels.numpy()).sum()),
        "test_samples": 247,
    }
    atomic_json_dump(summary, OUT / "linear_probe_summary.json")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("extract", "tune", "test"))
    parser.add_argument("--config", default=str(ROOT / "configs/change_jepa_duration_final_probe.yaml"))
    args = parser.parse_args()
    config = load_yaml(args.config)
    setup(config)
    {"extract": extract_train_val, "tune": tune, "test": test_once}[args.stage](config)


if __name__ == "__main__":
    main()
