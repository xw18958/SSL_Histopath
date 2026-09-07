"""Isolated, audited final probe for the selected B0 duration-pilot encoder."""
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

from pannuke_ssl.config import load_yaml
from pannuke_ssl.data import PanNukeImageDataset, loader_kwargs
from pannuke_ssl.parquet import build_source_index, preload_images, read_metadata, verify_records
from pannuke_ssl.probe import _fit_probe, _evaluate
from pannuke_ssl.training import load_student_checkpoint
from pannuke_ssl.utils import atomic_json_dump, seed_everything, write_csv

ROOT = Path("/raid1/xwan0900/SSL_proj")
ENCODER = ROOT / "outputs/b0_duration_pilot/checkpoints/best.pt"
OUT = ROOT / "outputs/b0_duration_pilot_final_probe"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha(values):
    h = hashlib.sha256()
    for key, value in sorted(values.items()):
        h.update(key.encode())
        a = value.detach().cpu().contiguous().numpy()
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def old_manifest():
    return {str(p.relative_to(ROOT)): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted((ROOT / "outputs/b0_final").rglob("*")) if p.is_file()}


def metadata(c):
    rows = read_metadata(Path(c["metadata_csv"]))
    assert len(rows) == 2546
    assert len({(r["fold"], r["sample_index"]) for r in rows}) == 2546
    expected = {"train": 2052, "val": 247, "test": 247}
    assert dict(Counter(r["split"] for r in rows)) == expected
    counts = Counter((r["split"], r["class_id"]) for r in rows)
    for split, count in {"train": 108, "val": 13, "test": 13}.items():
        assert [counts[(split, cls)] for cls in range(19)] == [count] * 19
    return rows


def setup(c):
    assert Path(c["encoder_checkpoint"]).resolve() == ENCODER
    assert Path(c["output_dir"]).resolve() == OUT
    assert c["learning_rates"] == [0.001, 0.003, 0.01]
    assert c["weight_decays"] == [0.0, 0.0001]
    assert c["maximum_epochs"] == 50 and c["early_stopping_patience"] == 8
    seed_everything(c["seed"])
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def verify_lock(c):
    lock = json.loads((OUT / "encoder_lock.json").read_text())
    assert sha(ENCODER) == lock["checkpoint_sha256"]
    assert sha(c["metadata_csv"]) == lock["metadata_sha256"]
    assert old_manifest() == lock["old_b0_final_manifest"]
    assert c == json.loads((OUT / "resolved_config.json").read_text())
    return lock


def lock_encoder(c):
    OUT.mkdir(exist_ok=False)
    checkpoint = torch.load(ENCODER, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 10
    assert checkpoint["config"]["train"]["epochs"] == 300
    assert checkpoint["config"]["train"]["warmup_fraction"] == 0.1
    assert checkpoint["config"]["output_dir"] == str(ROOT / "outputs/b0_duration_pilot")
    assert "student" in checkpoint and "teacher" not in checkpoint
    state = {k.removeprefix("_orig_mod."): v for k, v in checkpoint["student"].items()}
    lock = {"checkpoint_path": str(ENCODER), "checkpoint_sha256": sha(ENCODER),
            "student_state_sha256": tensor_sha(state), "epoch": 10,
            "schedule_epochs": 300, "warmup_epochs": 30,
            "wording": "epoch 10 of the 300-epoch schedule with 30-epoch warmup",
            "metadata_sha256": sha(c["metadata_csv"]), "old_b0_final_manifest": old_manifest(),
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "ssl_seed": checkpoint["config"]["seed"]}
    metadata(c)
    atomic_json_dump(c, OUT / "resolved_config.json")
    atomic_json_dump(lock, OUT / "encoder_lock.json")
    return lock


def extract(c, splits, lock):
    rows = metadata(c)
    selected = [r for r in rows if r["split"] in splits]
    index = build_source_index(Path(c["data_root"]))
    verify_records(rows, index)
    cache = preload_images(selected, index)
    assert set(cache) == {(r["fold"], r["sample_index"]) for r in selected}
    encoder = load_student_checkpoint(c["plip_config_dir"], ENCODER, torch.device("cuda"))
    assert not encoder.training and not any(p.requires_grad for p in encoder.parameters())
    before = tensor_sha(encoder.state_dict())
    assert before == lock["student_state_sha256"]
    values = {}
    duplicate_equal = None
    with torch.inference_mode():
        for split in splits:
            part = [r for r in selected if r["split"] == split]
            loader = DataLoader(PanNukeImageDataset(part, index, cache, include_label=True),
                                **loader_kwargs(c["batch_size"], c["num_workers"], shuffle=False))
            vectors, labels = [], []
            for images, y in loader:
                x = images.to(device="cuda", dtype=torch.float32).div_(255.0)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tokens = encoder(x)
                assert tokens.shape[1:] == (64, 768)
                pooled = tokens.float().mean(1).cpu()
                # Same exact batch is repeated only in the train/validation extraction smoke.
                if duplicate_equal is None and split == "train":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        repeated = encoder(x).float().mean(1).cpu()
                    duplicate_equal = torch.equal(pooled, repeated)
                    assert duplicate_equal
                assert torch.isfinite(pooled).all()
                vectors.append(pooled)
                labels.append(y)
            values[split + "_features"] = torch.cat(vectors).numpy()
            values[split + "_labels"] = torch.cat(labels).numpy()
            values[split + "_keys"] = np.array([(r["fold"], r["sample_index"]) for r in part])
            print(json.dumps({"extracted": split, "shape": list(values[split + "_features"].shape)}), flush=True)
    after = tensor_sha(encoder.state_dict())
    assert after == before
    assert all(p.grad is None for p in encoder.parameters())
    assert sha(ENCODER) == lock["checkpoint_sha256"]
    return values, {"encoder_state_before": before, "encoder_state_after": after,
                    "all_encoder_parameters_frozen": True, "all_encoder_gradients_none": True,
                    "repeated_batch_exactly_equal": duplicate_equal, "decoded_splits": splits,
                    "token_shape": [64, 768], "pooling": "tokens.float().mean(dim=1)",
                    "encoder_forward_precision": "bfloat16 autocast", "clean_images": True}


def extract_trainval(c):
    lock = lock_encoder(c)
    values, audit = extract(c, ["train", "val"], lock)
    np.savez_compressed(OUT / "train_val_features.npz", **values)
    audit["cache_sha256"] = sha(OUT / "train_val_features.npz")
    atomic_json_dump(audit, OUT / "feature_extraction_audit.json")
    verify_lock(c)


def load_trainval(c):
    verify_lock(c)
    audit = json.loads((OUT / "feature_extraction_audit.json").read_text())
    assert sha(OUT / "train_val_features.npz") == audit["cache_sha256"]
    with np.load(OUT / "train_val_features.npz") as data:
        assert set(data.files) == {s + "_" + k for s in ("train", "val") for k in ("features", "labels", "keys")}
        features = {s: (torch.from_numpy(data[s + "_features"].copy()),
                        torch.from_numpy(data[s + "_labels"].copy())) for s in ("train", "val")}
    mean = features["train"][0].mean(0, keepdim=True)
    std = features["train"][0].std(0, keepdim=True, unbiased=True).clamp_min(1e-6)
    standardized = {s: ((x - mean) / std, y) for s, (x, y) in features.items()}
    assert torch.isfinite(std).all() and torch.all(std > 0)
    return standardized, mean, std, audit["cache_sha256"]


def smoke(c):
    features, mean, std, cache_sha = load_trainval(c)
    assert features["train"][0].shape == (2052, 768)
    assert features["val"][0].shape == (247, 768)
    assert features["train"][0].mean(0).abs().max() < 1e-4
    h_before = tensor_sha({s + k: v for s, pair in features.items() for k, v in zip(("x", "y"), pair)})
    # Exercise the actual probe loop with a short, bounded trial, never saving its state.
    trial = _fit_probe(features, learning_rate=0.001, weight_decay=0,
                       maximum_epochs=2, patience=2, seed=c["seed"])
    assert len(trial["history"]) == 2
    assert all(np.isfinite(row["train_loss"]) and np.isfinite(row["val_loss"]) for row in trial["history"])
    h_after = tensor_sha({s + k: v for s, pair in features.items() for k, v in zip(("x", "y"), pair)})
    assert h_before == h_after
    verify_lock(c)
    atomic_json_dump({"passed": True, "smoke_epochs": 2, "test_images_accessed": False,
                      "metadata_counts": {"train": 2052, "val": 247, "test": 247},
                      "per_class_counts": [108, 13, 13], "cache_sha256": cache_sha,
                      "standardized_tensor_sha256": h_before,
                      "standardization": "training features only; sample std, floor 1e-6",
                      "cached_features_unchanged_after_probe": True,
                      "identical_cache_for_all_trials_enforced": True}, OUT / "smoke_test.json")
    print("SMOKE PASSED", flush=True)


def tune(c):
    assert json.loads((OUT / "smoke_test.json").read_text())["passed"]
    assert not (OUT / "selection.json").exists() and not (OUT / "best_linear_probe.pt").exists()
    assert not (OUT / "test_started.json").exists()
    features, mean, std, cache_sha = load_trainval(c)
    best, leaderboard, histories = None, [], []
    for lr in c["learning_rates"]:
        for wd in c["weight_decays"]:
            trial_sha = tensor_sha({s + k: v for s, pair in features.items() for k, v in zip(("x", "y"), pair)})
            assert trial_sha == json.loads((OUT / "smoke_test.json").read_text())["standardized_tensor_sha256"]
            trial = _fit_probe(features, learning_rate=lr, weight_decay=wd,
                               maximum_epochs=50, patience=8, seed=c["seed"])
            row = {k: v for k, v in trial.items() if k not in ("state", "history")}
            row.update({"epochs_run": len(trial["history"]), "seed": c["seed"],
                        "cache_sha256": cache_sha, "standardized_tensor_sha256": trial_sha})
            leaderboard.append(row)
            histories.extend({"lr": lr, "weight_decay": wd, **r} for r in trial["history"])
            print(json.dumps(row), flush=True)
            if best is None or trial["val_macro_f1"] > best["val_macro_f1"]:
                best = trial
    assert best is not None
    selection = {k: v for k, v in best.items() if k not in ("state", "history")}
    selection.update({"seed": c["seed"], "selection_split": "val", "test_accessed": False,
                      "cache_sha256": cache_sha, "encoder_checkpoint_sha256": sha(ENCODER),
                      "tie_break": "earliest epoch within trial; first grid candidate across trials",
                      "optimizer": "AdamW", "batch_size": 256, "lr_scheduler": "none",
                      "maximum_epochs": 50, "patience": 8, "standardization_source": "train only",
                      "std_correction": 1, "std_floor": 1e-6})
    state = {"classifier": {k: v.cpu() for k, v in best["state"].items()},
             "feature_mean": mean, "feature_std": std, "selection": selection}
    torch.save(state, OUT / "best_linear_probe.pt")
    selection["probe_checkpoint_sha256"] = sha(OUT / "best_linear_probe.pt")
    write_csv(leaderboard, OUT / "probe_leaderboard.csv")
    write_csv(histories, OUT / "all_trial_metrics.csv")
    write_csv(best["history"], OUT / "selected_probe_metrics.csv")
    verify_lock(c)
    atomic_json_dump(selection, OUT / "selection.json")


def test_once(c):
    lock = verify_lock(c)
    selection = json.loads((OUT / "selection.json").read_text())
    assert sha(OUT / "best_linear_probe.pt") == selection["probe_checkpoint_sha256"]
    # Exclusive marker prevents accidental re-evaluation, including after interruptions.
    with (OUT / "test_started.json").open("x") as f:
        json.dump({"selection_sha256": sha(OUT / "selection.json"), "time": time.time()}, f)
    values, audit = extract(c, ["test"], lock)
    np.savez_compressed(OUT / "test_features.npz", **values)
    audit["cache_sha256"] = sha(OUT / "test_features.npz")
    state = torch.load(OUT / "best_linear_probe.pt", map_location="cpu", weights_only=False)
    classifier = torch.nn.Linear(768, 19).cuda().eval()
    classifier.load_state_dict(state["classifier"])
    features = (torch.from_numpy(values["test_features"]) - state["feature_mean"]) / state["feature_std"]
    labels = torch.from_numpy(values["test_labels"])
    loss, metrics, predictions = _evaluate(classifier, features, labels, torch.device("cuda"))
    # Persist predictions immediately; subsequent reports operate on this one evaluation.
    np.savez_compressed(OUT / "test_predictions.npz", labels=labels.numpy(), predictions=predictions,
                        keys=values["test_keys"])
    from sklearn.metrics import classification_report, confusion_matrix
    names = {int(r["class_id"]): r["tissue_label"] for r in metadata(c)}
    report = classification_report(labels.numpy(), predictions, labels=list(range(19)),
                                   output_dict=True, zero_division=0)
    class_rows = [{"class_id": i, "class_name": names[i], **report[str(i)]} for i in range(19)]
    write_csv(class_rows, OUT / "test_per_class_metrics.csv")
    matrix = confusion_matrix(labels.numpy(), predictions, labels=list(range(19)))
    np.savetxt(OUT / "test_confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks(range(19), [names[i] for i in range(19)], rotation=65, ha="right")
    ax.set_yticks(range(19), [names[i] for i in range(19)])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("B0 duration-pilot encoder: single-seed transductive linear probe")
    for i in range(19):
        for j in range(19):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT / "test_confusion_matrix.png", dpi=180)
    plt.close(fig)
    summary = {"evaluation_protocol": "single-seed, transductive B0 linear-probe result",
               "encoder_description": lock["wording"], "ssl_unlabeled_images": 7901,
               "ssl_includes_downstream_validation_test": True, "test_evaluations": 1,
               "selection": selection, "test_loss": loss, "test": metrics,
               "correct_test_predictions": int((predictions == labels.numpy()).sum()), "test_samples": 247,
               "encoder_checkpoint_sha256": sha(ENCODER),
               "limitation": "Does not establish a universal optimal epoch; reproducibility requires fixed warmup steps and multiple SSL seeds."}
    verify_lock(c)
    atomic_json_dump(audit, OUT / "test_feature_extraction_audit.json")
    atomic_json_dump(summary, OUT / "linear_probe_summary.json")
    print(json.dumps(summary, indent=2), flush=True)


def main(stage):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs/b0_duration_final_probe.yaml"))
    args = p.parse_args()
    c = load_yaml(args.config)
    setup(c)
    {"extract": extract_trainval, "smoke": smoke, "tune": tune, "test": test_once}[stage](c)
