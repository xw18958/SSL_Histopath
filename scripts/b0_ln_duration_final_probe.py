"""Isolated, guarded final probe for the selected B0-LN duration-pilot encoder."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader

from pannuke_ssl.b0_ln_training import validate_config as validate_pilot_config
from pannuke_ssl.b0_ln_training import verify_implementation_manifest, verify_protected_manifest
from pannuke_ssl.config import load_yaml
from pannuke_ssl.data import PanNukeImageDataset, loader_kwargs
from pannuke_ssl.parquet import build_source_index, preload_images, read_metadata, verify_records
from pannuke_ssl.probe import _evaluate, _fit_probe
from pannuke_ssl.training import load_student_checkpoint
from pannuke_ssl.utils import atomic_json_dump, seed_everything, write_csv

ROOT = Path("/raid1/xwan0900/SSL_proj")
PILOT = ROOT / "outputs/b0_ln_duration_pilot"
ENCODER = PILOT / "checkpoints/best.pt"
OUT = ROOT / "outputs/b0_ln_duration_pilot_final_probe"


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        digest.update(key.encode())
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def metadata(config: dict) -> list[dict]:
    rows = read_metadata(Path(config["metadata_csv"]))
    if len(rows) != 2546 or len({(row["fold"], row["sample_index"]) for row in rows}) != 2546:
        raise RuntimeError("B0-LN balanced metadata keys are invalid")
    if Counter(row["split"] for row in rows) != {"train": 2052, "val": 247, "test": 247}:
        raise RuntimeError("B0-LN balanced metadata split counts are invalid")
    counts = Counter((row["split"], row["class_id"]) for row in rows)
    for split, expected in (("train", 108), ("val", 13), ("test", 13)):
        if [counts[split, class_id] for class_id in range(19)] != [expected] * 19:
            raise RuntimeError("B0-LN balanced metadata per-class counts are invalid")
    return rows


def setup(config: dict) -> None:
    if Path(config["encoder_checkpoint"]).resolve() != ENCODER or Path(config["pilot_output_dir"]).resolve() != PILOT:
        raise ValueError("B0-LN final probe must use only its selected B0-LN encoder")
    if Path(config["output_dir"]).resolve() != OUT:
        raise ValueError("B0-LN final probe output directory is fixed and isolated")
    if config["learning_rates"] != [0.001, 0.003, 0.01] or config["weight_decays"] != [0.0, 0.0001]:
        raise ValueError("B0-LN final probe grid must exactly match B0")
    if config["maximum_epochs"] != 50 or config["early_stopping_patience"] != 8:
        raise ValueError("B0-LN final probe duration must exactly match B0")
    seed_everything(config["seed"])
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def verify_lock(config: dict) -> dict:
    lock = json.loads((OUT / "encoder_lock.json").read_text(encoding="utf-8"))
    if sha(ENCODER) != lock["checkpoint_sha256"] or sha(config["metadata_csv"]) != lock["metadata_sha256"]:
        raise RuntimeError("B0-LN final probe encoder or metadata lock changed")
    if sha(PILOT / "duration_selection.json") != lock["duration_selection_sha256"]:
        raise RuntimeError("B0-LN duration selection changed after probe lock")
    if sha(PILOT / "protected_manifest.json") != lock["protected_manifest_sha256"]:
        raise RuntimeError("B0-LN protected reference manifest changed after probe lock")
    if sha(PILOT / "implementation_manifest.json") != lock["implementation_manifest_sha256"]:
        raise RuntimeError("B0-LN implementation manifest changed after probe lock")
    if config != json.loads((OUT / "resolved_config.json").read_text(encoding="utf-8")):
        raise RuntimeError("B0-LN final probe configuration changed after lock")
    verify_protected_manifest(PILOT)
    verify_implementation_manifest(PILOT)
    return lock


def lock_encoder(config: dict) -> dict:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite B0-LN final-probe output: {OUT}")
    if not ENCODER.is_file() or not (PILOT / "duration_selection.json").is_file():
        raise FileNotFoundError("B0-LN selected checkpoint/duration selection is not ready")
    verify_protected_manifest(PILOT)
    verify_implementation_manifest(PILOT)
    duration = json.loads((PILOT / "duration_selection.json").read_text(encoding="utf-8"))
    if duration.get("selection_split") != "validation" or duration.get("test_split_used_for_selection"):
        raise RuntimeError("B0-LN checkpoint selection must be validation-only")
    checkpoint = torch.load(ENCODER, map_location="cpu", weights_only=False)
    pilot_config = checkpoint.get("config")
    validate_pilot_config(pilot_config)
    if checkpoint.get("epoch") != duration.get("selected_epoch") or "teacher" in checkpoint or "predictor" in checkpoint:
        raise RuntimeError("B0-LN checkpoint is not the frozen selected student-only artifact")
    state = {key.removeprefix("_orig_mod."): value for key, value in checkpoint["student"].items()}
    metadata(config)
    OUT.mkdir(exist_ok=False)
    atomic_json_dump(config, OUT / "resolved_config.json")
    lock = {
        "checkpoint_path": str(ENCODER),
        "checkpoint_sha256": sha(ENCODER),
        "student_state_sha256": tensor_sha(state),
        "selected_epoch": int(checkpoint["epoch"]),
        "configured_schedule_epochs": 300,
        "warmup_epochs": 30,
        "duration_selection_sha256": sha(PILOT / "duration_selection.json"),
        "metadata_sha256": sha(config["metadata_csv"]),
        "protected_manifest_sha256": sha(PILOT / "protected_manifest.json"),
        "implementation_manifest_sha256": sha(PILOT / "implementation_manifest.json"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "ssl_seed": checkpoint["config"]["seed"],
        "wording": f"epoch {checkpoint['epoch']} of the 300-epoch schedule with 30-epoch warmup; validation-only adaptive stop enabled",
    }
    atomic_json_dump(lock, OUT / "encoder_lock.json")
    return lock


def extract(config: dict, splits: list[str], lock: dict) -> tuple[dict[str, np.ndarray], dict]:
    rows = metadata(config)
    selected = [row for row in rows if row["split"] in splits]
    source_index = build_source_index(Path(config["data_root"]))
    verify_records(selected, source_index)
    cache = preload_images(selected, source_index)
    expected_keys = {(row["fold"], row["sample_index"]) for row in selected}
    if set(cache) != expected_keys:
        raise RuntimeError("B0-LN feature cache contains records outside requested splits")
    encoder = load_student_checkpoint(config["plip_config_dir"], ENCODER, torch.device("cuda"))
    if encoder.training or any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("B0-LN probe encoder must be frozen")
    before = tensor_sha(encoder.state_dict())
    if before != lock["student_state_sha256"]:
        raise RuntimeError("B0-LN loaded encoder differs from locked state")
    values: dict[str, np.ndarray] = {}
    repeated_equal = None
    with torch.inference_mode():
        for split in splits:
            part = [row for row in selected if row["split"] == split]
            loader = DataLoader(
                PanNukeImageDataset(part, source_index, cache, include_label=True),
                **loader_kwargs(config["batch_size"], config["num_workers"], shuffle=False),
            )
            vectors, labels = [], []
            for images, labels_batch in loader:
                images = images.to(device="cuda", dtype=torch.float32).div_(255.0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tokens = encoder(images)
                if tokens.shape[1:] != (64, 768):
                    raise RuntimeError("B0-LN feature extraction did not produce 64x768 tokens")
                pooled = tokens.float().mean(dim=1).cpu()
                if repeated_equal is None and split == "train":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        repeat = encoder(images).float().mean(dim=1).cpu()
                    repeated_equal = torch.equal(pooled, repeat)
                    if not repeated_equal:
                        raise RuntimeError("B0-LN frozen encoder is not deterministic during feature extraction")
                if not torch.isfinite(pooled).all():
                    raise FloatingPointError("B0-LN feature extraction produced non-finite features")
                vectors.append(pooled)
                labels.append(labels_batch)
            values[split + "_features"] = torch.cat(vectors).numpy()
            values[split + "_labels"] = torch.cat(labels).numpy()
            values[split + "_keys"] = np.asarray([(row["fold"], row["sample_index"]) for row in part])
            print(json.dumps({"extracted": split, "shape": list(values[split + "_features"].shape)}), flush=True)
    after = tensor_sha(encoder.state_dict())
    if after != before or any(parameter.grad is not None for parameter in encoder.parameters()):
        raise RuntimeError("B0-LN frozen encoder changed during feature extraction")
    if sha(ENCODER) != lock["checkpoint_sha256"]:
        raise RuntimeError("B0-LN selected checkpoint changed during feature extraction")
    return values, {
        "encoder_state_before": before,
        "encoder_state_after": after,
        "all_encoder_parameters_frozen": True,
        "all_encoder_gradients_none": True,
        "repeated_batch_exactly_equal": repeated_equal,
        "decoded_splits": splits,
        "token_shape": [64, 768],
        "pooling": "tokens.float().mean(dim=1)",
        "encoder_forward_precision": "bfloat16 autocast",
        "clean_images": True,
    }


def extract_trainval(config: dict) -> None:
    lock = lock_encoder(config)
    values, audit = extract(config, ["train", "val"], lock)
    np.savez_compressed(OUT / "train_val_features.npz", **values)
    audit["cache_sha256"] = sha(OUT / "train_val_features.npz")
    atomic_json_dump(audit, OUT / "feature_extraction_audit.json")
    verify_lock(config)


def load_trainval(config: dict) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor, str]:
    verify_lock(config)
    audit = json.loads((OUT / "feature_extraction_audit.json").read_text(encoding="utf-8"))
    if sha(OUT / "train_val_features.npz") != audit["cache_sha256"]:
        raise RuntimeError("B0-LN train/validation feature cache changed")
    with np.load(OUT / "train_val_features.npz") as data:
        expected = {split + "_" + value for split in ("train", "val") for value in ("features", "labels", "keys")}
        if set(data.files) != expected:
            raise RuntimeError("B0-LN train/validation feature cache keys are invalid")
        features = {
            split: (torch.from_numpy(data[split + "_features"].copy()), torch.from_numpy(data[split + "_labels"].copy()))
            for split in ("train", "val")
        }
    mean = features["train"][0].mean(0, keepdim=True)
    std = features["train"][0].std(0, keepdim=True, unbiased=True).clamp_min(1e-6)
    standardized = {split: ((values - mean) / std, labels) for split, (values, labels) in features.items()}
    if not torch.isfinite(std).all() or not torch.all(std > 0):
        raise FloatingPointError("B0-LN train-only feature standardization is invalid")
    return standardized, mean, std, audit["cache_sha256"]


def smoke(config: dict) -> None:
    features, _mean, _std, cache_sha = load_trainval(config)
    if features["train"][0].shape != (2052, 768) or features["val"][0].shape != (247, 768):
        raise RuntimeError("B0-LN final-probe train/validation feature shapes are invalid")
    if features["train"][0].mean(0).abs().max() >= 1e-4:
        raise RuntimeError("B0-LN train-only standardization does not center train features")
    before = tensor_sha({split + value: tensor for split, pair in features.items() for value, tensor in zip(("x", "y"), pair)})
    trial = _fit_probe(features, learning_rate=0.001, weight_decay=0.0, maximum_epochs=2, patience=2, seed=config["seed"])
    if len(trial["history"]) != 2 or not all(np.isfinite(row["train_loss"]) and np.isfinite(row["val_loss"]) for row in trial["history"]):
        raise RuntimeError("B0-LN final-probe smoke failed")
    after = tensor_sha({split + value: tensor for split, pair in features.items() for value, tensor in zip(("x", "y"), pair)})
    if before != after:
        raise RuntimeError("B0-LN final-probe smoke changed cached train/validation features")
    verify_lock(config)
    atomic_json_dump(
        {
            "passed": True,
            "smoke_epochs": 2,
            "test_images_accessed": False,
            "metadata_counts": {"train": 2052, "val": 247, "test": 247},
            "per_class_counts": [108, 13, 13],
            "cache_sha256": cache_sha,
            "standardized_tensor_sha256": before,
            "standardization": "training features only; sample std, floor 1e-6",
            "cached_features_unchanged_after_probe": True,
            "identical_cache_for_all_trials_enforced": True,
        },
        OUT / "smoke_test.json",
    )
    print("B0-LN FINAL-PROBE SMOKE PASSED", flush=True)


def tune(config: dict) -> None:
    if not json.loads((OUT / "smoke_test.json").read_text(encoding="utf-8")).get("passed"):
        raise RuntimeError("B0-LN final-probe tuning requires a passing train/validation smoke")
    if (OUT / "selection.json").exists() or (OUT / "best_linear_probe.pt").exists() or (OUT / "test_started.json").exists():
        raise FileExistsError("Refusing to overwrite B0-LN probe selection or tune after a test marker")
    features, mean, std, cache_sha = load_trainval(config)
    best, leaderboard, histories = None, [], []
    for learning_rate in config["learning_rates"]:
        for weight_decay in config["weight_decays"]:
            trial_sha = tensor_sha({split + value: tensor for split, pair in features.items() for value, tensor in zip(("x", "y"), pair)})
            if trial_sha != json.loads((OUT / "smoke_test.json").read_text(encoding="utf-8"))["standardized_tensor_sha256"]:
                raise RuntimeError("B0-LN probe trial did not receive the locked standardized cache")
            trial = _fit_probe(features, learning_rate=learning_rate, weight_decay=weight_decay,
                               maximum_epochs=50, patience=8, seed=config["seed"])
            row = {key: value for key, value in trial.items() if key not in ("state", "history")}
            row.update({"epochs_run": len(trial["history"]), "seed": config["seed"], "cache_sha256": cache_sha,
                        "standardized_tensor_sha256": trial_sha})
            leaderboard.append(row)
            histories.extend({"learning_rate": learning_rate, "weight_decay": weight_decay, **entry} for entry in trial["history"])
            print(json.dumps(row, sort_keys=True), flush=True)
            if best is None or trial["val_macro_f1"] > best["val_macro_f1"]:
                best = trial
    if best is None:
        raise RuntimeError("B0-LN final-probe grid produced no candidates")
    selection = {key: value for key, value in best.items() if key not in ("state", "history")}
    selection.update(
        {
            "seed": config["seed"],
            "selection_split": "val",
            "test_accessed": False,
            "cache_sha256": cache_sha,
            "encoder_checkpoint_sha256": sha(ENCODER),
            "tie_break": "earliest epoch within trial; first grid candidate across trials",
            "optimizer": "AdamW",
            "batch_size": 256,
            "lr_scheduler": "none",
            "maximum_epochs": 50,
            "patience": 8,
            "standardization_source": "train only",
            "std_correction": 1,
            "std_floor": 1e-6,
        }
    )
    state = {"classifier": {key: value.cpu() for key, value in best["state"].items()}, "feature_mean": mean, "feature_std": std,
             "selection": selection}
    torch.save(state, OUT / "best_linear_probe.pt")
    selection["probe_checkpoint_sha256"] = sha(OUT / "best_linear_probe.pt")
    write_csv(leaderboard, OUT / "probe_leaderboard.csv")
    write_csv(histories, OUT / "all_trial_metrics.csv")
    write_csv(best["history"], OUT / "selected_probe_metrics.csv")
    verify_lock(config)
    atomic_json_dump(selection, OUT / "selection.json")


def test_once(config: dict) -> None:
    lock = verify_lock(config)
    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    if sha(OUT / "best_linear_probe.pt") != selection["probe_checkpoint_sha256"]:
        raise RuntimeError("B0-LN selected probe checkpoint changed")
    # Claim the test before any test dataset/cache/loader construction.  The marker is never removed.
    with (OUT / "test_started.json").open("x", encoding="utf-8") as handle:
        json.dump({"selection_sha256": sha(OUT / "selection.json"), "time": time.time()}, handle)
    values, audit = extract(config, ["test"], lock)
    if values["test_features"].shape != (247, 768):
        raise RuntimeError("B0-LN one-time test did not extract exactly 247 features")
    np.savez_compressed(OUT / "test_features.npz", **values)
    audit["cache_sha256"] = sha(OUT / "test_features.npz")
    state = torch.load(OUT / "best_linear_probe.pt", map_location="cpu", weights_only=False)
    classifier = torch.nn.Linear(768, 19).cuda().eval()
    classifier.load_state_dict(state["classifier"])
    features = (torch.from_numpy(values["test_features"]) - state["feature_mean"]) / state["feature_std"]
    labels = torch.from_numpy(values["test_labels"])
    loss, metrics, predictions = _evaluate(classifier, features, labels, torch.device("cuda"))
    np.savez_compressed(OUT / "test_predictions.npz", labels=labels.numpy(), predictions=predictions, keys=values["test_keys"])
    from matplotlib import pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix

    names = {int(row["class_id"]): row["tissue_label"] for row in metadata(config)}
    report = classification_report(labels.numpy(), predictions, labels=list(range(19)), output_dict=True, zero_division=0)
    write_csv([{"class_id": index, "class_name": names[index], **report[str(index)]} for index in range(19)], OUT / "test_per_class_metrics.csv")
    matrix = confusion_matrix(labels.numpy(), predictions, labels=list(range(19)))
    np.savetxt(OUT / "test_confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(matrix, cmap="Blues", vmin=0)
    axis.set_xticks(range(19), [names[index] for index in range(19)], rotation=65, ha="right")
    axis.set_yticks(range(19), [names[index] for index in range(19)])
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("B0-LN encoder: single-seed transductive linear probe")
    for row in range(19):
        for column in range(19):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=7,
                      color="white" if matrix[row, column] > matrix.max() / 2 else "black")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(OUT / "test_confusion_matrix.png", dpi=180)
    plt.close(figure)
    summary = {
        "evaluation_protocol": "single-seed, transductive B0-LN linear-probe result",
        "encoder_description": lock["wording"],
        "ssl_unlabeled_images": 7901,
        "ssl_includes_downstream_validation_test": True,
        "test_evaluations": 1,
        "selection": selection,
        "test_loss": loss,
        "test": metrics,
        "correct_test_predictions": int((predictions == labels.numpy()).sum()),
        "test_samples": 247,
        "encoder_checkpoint_sha256": sha(ENCODER),
        "limitation": "Single-seed transductive result; test metrics are descriptive and do not establish significance or inductive generalization.",
    }
    verify_lock(config)
    atomic_json_dump(audit, OUT / "test_feature_extraction_audit.json")
    atomic_json_dump(summary, OUT / "linear_probe_summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def main(stage: str) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/b0_ln_duration_final_probe.yaml"))
    args = parser.parse_args()
    config = load_yaml(args.config)
    setup(config)
    {"extract": extract_trainval, "smoke": smoke, "tune": tune, "test": test_once}[stage](config)


if __name__ == "__main__":
    raise SystemExit("Use a stage wrapper: extract_b0_ln_duration_probe_features.py, smoke_b0_ln_duration_final_probe.py, tune_b0_ln_duration_final_probe.py, or test_b0_ln_duration_final_probe_once.py")
