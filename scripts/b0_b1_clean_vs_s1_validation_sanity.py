"""Validation-only B0/B1 clean-versus-maximum-degradation diagnostic.

This runner intentionally has no test stage and never constructs a test image
loader.  It loads the existing hash-verified clean train/validation caches for
reporting only, creates new *degraded* train/validation feature caches in its
own exclusive output directory, and fits the locked final-probe grid there.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader

from pannuke_ssl.config import load_yaml
from pannuke_ssl.data import PanNukeImageDataset, loader_kwargs
from pannuke_ssl.degradations import DEFOCUS, RESOLUTION, DegradationEndpoints, degrade_batch
from pannuke_ssl.parquet import build_source_index, preload_images, verify_records
from pannuke_ssl.probe import _fit_probe
from pannuke_ssl.training import load_student_checkpoint
from pannuke_ssl.utils import atomic_json_dump, seed_everything, write_csv


ROOT = Path("/raid1/xwan0900/SSL_proj")
EXPECTED_OUTPUT = ROOT / "outputs/b0_b1_clean_vs_s1_validation_sanity"
EXPECTED_PROTECTED = ROOT / "outputs/b1_duration_pilot"
FIELDS = ("fold", "sample_index", "tissue_label", "class_id", "split")
SPLITS = ("train", "val")
ENCODERS = ("b0", "b1")


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def verify_protected_manifest(output_dir: str | Path) -> dict[str, Any]:
    """Verify the immutable B0/I-JEPA reference manifest without B1 imports."""
    output = Path(output_dir)
    manifest_path = output / "protected_manifest.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert value.get("schema_version") == 1 and isinstance(value.get("protected_paths"), list)
    root = output.parents[1]
    for entry in value["protected_paths"]:
        path = root / entry["path"]
        assert path.is_file()
        assert path.stat().st_size == int(entry["size"]) and sha(path) == entry["sha256"]
    return value


def no_test_artifacts(output: Path) -> None:
    """Guard against accidentally adding a held-out-test execution mode."""
    forbidden = ("test_started.json", "test_features.npz", "test_predictions.npz")
    assert not output.exists() or not any((output / name).exists() for name in forbidden)


def validate_config(config: dict[str, Any]) -> None:
    assert config["seed"] == 20260903
    assert Path(config["output_dir"]).resolve() == EXPECTED_OUTPUT
    assert Path(config["protected_manifest_dir"]).resolve() == EXPECTED_PROTECTED
    assert config["batch_size"] == 256 and config["num_workers"] == 4
    assert config["severity"] == 1.0
    assert config["action_assignment"] == "class_stratified_alternating_by_fold_sample_index"
    assert config["maximum_epochs"] == 50 and config["early_stopping_patience"] == 8
    assert config["learning_rates"] == [0.001, 0.003, 0.01]
    assert config["weight_decays"] == [0.0, 0.0001]
    assert tuple(config["encoders"]) == ENCODERS
    for name in ENCODERS:
        required = {"checkpoint", "encoder_lock", "clean_feature_cache", "clean_feature_audit", "clean_selection"}
        assert set(config["encoders"][name]) == required


def read_trainval_metadata(path: str | Path) -> list[dict[str, Any]]:
    """Store only train/validation rows; test rows are never materialized."""
    rows: list[dict[str, Any]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == FIELDS
        for raw in reader:
            if raw["split"] not in SPLITS:
                continue
            rows.append({
                "fold": int(raw["fold"]),
                "sample_index": int(raw["sample_index"]),
                "tissue_label": raw["tissue_label"],
                "class_id": int(raw["class_id"]),
                "split": raw["split"],
            })
    assert len(rows) == 2299
    assert len({(row["fold"], row["sample_index"]) for row in rows}) == 2299
    assert Counter(row["split"] for row in rows) == {"train": 2052, "val": 247}
    counts = Counter((row["split"], row["class_id"]) for row in rows)
    for split, count in (("train", 108), ("val", 13)):
        assert [counts[(split, class_id)] for class_id in range(19)] == [count] * 19
    return rows


def action_map(rows: list[dict[str, Any]]) -> dict[tuple[int, int], int]:
    """Balance actions within each class/split by sorted key and class parity."""
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), int(row["class_id"]))].append(row)
    mapping: dict[tuple[int, int], int] = {}
    for split, count in (("train", 108), ("val", 13)):
        for class_id in range(19):
            ordered = sorted(grouped[(split, class_id)], key=lambda row: (row["fold"], row["sample_index"]))
            assert len(ordered) == count
            first = DEFOCUS if class_id % 2 == 0 else RESOLUTION
            for offset, row in enumerate(ordered):
                mapping[(row["fold"], row["sample_index"])] = (first + offset) % 2
    assert len(mapping) == len(rows)
    by_split = Counter(
        (row["split"], mapping[(row["fold"], row["sample_index"])]) for row in rows
    )
    assert by_split["train", DEFOCUS] + by_split["train", RESOLUTION] == 2052
    assert by_split["val", DEFOCUS] + by_split["val", RESOLUTION] == 247
    assert abs(by_split["train", DEFOCUS] - by_split["train", RESOLUTION]) <= 1
    assert abs(by_split["val", DEFOCUS] - by_split["val", RESOLUTION]) <= 1
    return mapping


def load_endpoints(path: str | Path) -> DegradationEndpoints:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    endpoints = DegradationEndpoints(
        defocus_radius=float(value["defocus"]["selected_radius"]),
        resolution_factor=float(value["resolution"]["selected_factor"]),
    )
    assert endpoints.defocus_radius > 0 and endpoints.resolution_factor > 1
    return endpoints


def verify_encoder_inputs(config: dict[str, Any], name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    spec = config["encoders"][name]
    lock = json.loads(Path(spec["encoder_lock"]).read_text(encoding="utf-8"))
    selection = json.loads(Path(spec["clean_selection"]).read_text(encoding="utf-8"))
    audit = json.loads(Path(spec["clean_feature_audit"]).read_text(encoding="utf-8"))
    checkpoint = Path(spec["checkpoint"])
    cache = Path(spec["clean_feature_cache"])
    assert sha(checkpoint) == lock["checkpoint_sha256"] == selection["encoder_checkpoint_sha256"]
    assert sha(config["metadata_csv"]) == lock["metadata_sha256"]
    assert sha(cache) == audit["cache_sha256"] == selection["cache_sha256"]
    assert selection["selection_split"] == "val" and selection["test_accessed"] is False
    assert selection["seed"] == config["seed"]
    assert selection["learning_rate"] in config["learning_rates"]
    assert selection["weight_decay"] in config["weight_decays"]
    with np.load(cache, allow_pickle=False) as values:
        expected = {f"{split}_{item}" for split in SPLITS for item in ("features", "labels", "keys")}
        assert set(values.files) == expected
        for split, expected_shape in (("train", (2052, 768)), ("val", (247, 768))):
            part = [row for row in rows if row["split"] == split]
            np.testing.assert_array_equal(values[f"{split}_keys"], np.asarray([(r["fold"], r["sample_index"]) for r in part]))
            np.testing.assert_array_equal(values[f"{split}_labels"], np.asarray([r["class_id"] for r in part]))
            assert values[f"{split}_features"].shape == expected_shape
            assert np.isfinite(values[f"{split}_features"]).all()
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {key.removeprefix("_orig_mod."): value for key, value in raw["student"].items()}
    assert tensor_sha(state) == lock["student_state_sha256"]
    return {
        "checkpoint_sha256": sha(checkpoint),
        "student_state_sha256": tensor_sha(state),
        "clean_feature_cache_sha256": sha(cache),
        "clean_val_macro_f1": float(selection["val_macro_f1"]),
        "clean_selected_learning_rate": float(selection["learning_rate"]),
        "clean_selected_weight_decay": float(selection["weight_decay"]),
        "clean_selected_epoch": int(selection["best_epoch"]),
    }


def setup(config: dict[str, Any]) -> tuple[Path, list[dict[str, Any]], dict[tuple[int, int], int], dict[str, Any]]:
    validate_config(config)
    output = Path(config["output_dir"])
    no_test_artifacts(output)
    assert not output.exists(), f"Refusing to overwrite existing sanity-check output: {output}"
    seed_everything(int(config["seed"]))
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    protected = verify_protected_manifest(config["protected_manifest_dir"])
    assert len(protected["protected_paths"]) == 129
    rows = read_trainval_metadata(config["metadata_csv"])
    actions = action_map(rows)
    locks = {name: verify_encoder_inputs(config, name, rows) for name in ENCODERS}
    output.mkdir(exist_ok=False)
    atomic_json_dump(config, output / "resolved_config.json")
    atomic_json_dump({"protected_manifest": protected, "metadata_sha256": sha(config["metadata_csv"]),
                      "encoders": locks, "test_loader_constructed": False,
                      "test_images_decoded": False}, output / "input_lock.json")
    action_rows = []
    for row in rows:
        action = actions[(row["fold"], row["sample_index"])]
        action_rows.append({**row, "severity": 1.0, "action": action,
                            "action_name": "defocus" if action == DEFOCUS else "resolution"})
    write_csv(action_rows, output / "s1_action_map.csv")
    return output, rows, actions, {"protected": protected, "encoders": locks}


def model_for(config: dict[str, Any], name: str, expected_state_sha: str) -> torch.nn.Module:
    encoder = load_student_checkpoint(config["plip_config_dir"], config["encoders"][name]["checkpoint"], torch.device("cuda"))
    assert not encoder.training and not any(parameter.requires_grad for parameter in encoder.parameters())
    assert tensor_sha(encoder.state_dict()) == expected_state_sha
    return encoder


def extract_s1_features(
    config: dict[str, Any], name: str, rows: list[dict[str, Any]], actions: dict[tuple[int, int], int],
    expected_state_sha: str, endpoints: DegradationEndpoints, *, smoke_only: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Extract only train/validation maximum-degradation features; never test."""
    index = build_source_index(Path(config["data_root"]))
    verify_records(rows, index)
    cache = preload_images(rows, index)
    assert set(cache) == {(row["fold"], row["sample_index"]) for row in rows}
    encoder = model_for(config, name, expected_state_sha)
    before = tensor_sha(encoder.state_dict())
    result: dict[str, np.ndarray] = {}
    action_seen: Counter[int] = Counter()
    with torch.inference_mode():
        for split in SPLITS:
            part = [row for row in rows if row["split"] == split]
            loader = DataLoader(PanNukeImageDataset(part, index, cache, include_label=True, include_key=True),
                                **loader_kwargs(config["batch_size"], config["num_workers"], shuffle=False))
            vectors, labels, keys = [], [], []
            for batch_index, (uint8_images, y, folds, sample_indices) in enumerate(loader):
                batch_keys = list(zip(folds.tolist(), sample_indices.tolist()))
                batch_actions = torch.tensor([actions[key] for key in batch_keys], device="cuda", dtype=torch.long)
                assert set(batch_actions.tolist()) <= {DEFOCUS, RESOLUTION}
                severity = torch.ones(batch_actions.shape[0], device="cuda", dtype=torch.float32)
                images = uint8_images.to(device="cuda", dtype=torch.float32, non_blocking=True).div_(255.0)
                degraded = degrade_batch(images, batch_actions, severity, endpoints)
                assert torch.isfinite(degraded).all() and degraded.shape == images.shape
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tokens = encoder(degraded)
                assert tokens.shape[1:] == (64, 768)
                pooled = tokens.float().mean(1).cpu()
                assert torch.isfinite(pooled).all()
                vectors.append(pooled)
                labels.append(y.cpu())
                keys.extend(batch_keys)
                action_seen.update(batch_actions.cpu().tolist())
                if smoke_only:
                    break
            if smoke_only:
                break
            result[f"{split}_features"] = torch.cat(vectors).numpy()
            result[f"{split}_labels"] = torch.cat(labels).numpy()
            result[f"{split}_keys"] = np.asarray(keys)
            assert result[f"{split}_features"].shape == ((2052 if split == "train" else 247), 768)
            np.testing.assert_array_equal(result[f"{split}_keys"], np.asarray([(r["fold"], r["sample_index"]) for r in part]))
            np.testing.assert_array_equal(result[f"{split}_labels"], np.asarray([r["class_id"] for r in part]))
    after = tensor_sha(encoder.state_dict())
    assert after == before == expected_state_sha
    assert all(parameter.grad is None for parameter in encoder.parameters())
    audit = {"encoder_state_before": before, "encoder_state_after": after,
             "all_encoder_parameters_frozen": True, "all_encoder_gradients_none": True,
             "decoded_splits": list(SPLITS) if not smoke_only else ["train"],
             "test_loader_constructed": False, "test_images_decoded": False,
             "severity": 1.0, "token_shape": [64, 768],
             "pooling": "tokens.float().mean(dim=1)",
             "encoder_forward_precision": "bfloat16 autocast",
             "action_counts_observed": {"defocus": int(action_seen[DEFOCUS]), "resolution": int(action_seen[RESOLUTION])}}
    return result, audit


def smoke(config: dict[str, Any], output: Path, rows: list[dict[str, Any]], actions: dict[tuple[int, int], int], locks: dict[str, Any], endpoints: DegradationEndpoints) -> None:
    result = {}
    for name in ENCODERS:
        _, audit = extract_s1_features(config, name, rows, actions, locks[name]["student_state_sha256"], endpoints, smoke_only=True)
        result[name] = audit
    verify_protected_manifest(config["protected_manifest_dir"])
    atomic_json_dump({"passed": True, "encoders": result, "test_loader_constructed": False,
                      "test_images_decoded": False}, output / "smoke_test.json")


def standardize(values: dict[str, np.ndarray]) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor]:
    features = {split: (torch.from_numpy(values[f"{split}_features"].copy()),
                        torch.from_numpy(values[f"{split}_labels"].copy())) for split in SPLITS}
    mean = features["train"][0].mean(0, keepdim=True)
    std = features["train"][0].std(0, keepdim=True, unbiased=True).clamp_min(1e-6)
    standardized = {split: ((x - mean) / std, y) for split, (x, y) in features.items()}
    assert torch.isfinite(mean).all() and torch.isfinite(std).all() and torch.all(std > 0)
    return standardized, mean, std


def tune(config: dict[str, Any], output: Path, name: str, values: dict[str, np.ndarray], lock: dict[str, Any]) -> dict[str, Any]:
    features, mean, std = standardize(values)
    cache_path = output / f"{name}_s1_train_val_features.npz"
    cache_sha = sha(cache_path)
    best: dict[str, Any] | None = None
    leaderboard: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    for learning_rate in config["learning_rates"]:
        for weight_decay in config["weight_decays"]:
            trial = _fit_probe(features, learning_rate=learning_rate, weight_decay=weight_decay,
                               maximum_epochs=config["maximum_epochs"], patience=config["early_stopping_patience"],
                               seed=config["seed"])
            row = {key: value for key, value in trial.items() if key not in ("state", "history")}
            row.update({"encoder": name, "epochs_run": len(trial["history"]), "seed": config["seed"],
                        "cache_sha256": cache_sha, "selection_split": "val"})
            leaderboard.append(row)
            histories.extend({"encoder": name, "learning_rate": learning_rate, "weight_decay": weight_decay, **item}
                             for item in trial["history"])
            if best is None or trial["val_macro_f1"] > best["val_macro_f1"]:
                best = trial
    assert best is not None
    selection = {key: value for key, value in best.items() if key not in ("state", "history")}
    selection.update({"encoder": name, "condition": "maximum_degradation_s1_mixed_actions", "selection_split": "val",
                      "test_accessed": False, "cache_sha256": cache_sha, "encoder_checkpoint_sha256": lock["checkpoint_sha256"],
                      "tie_break": "earliest epoch within trial; first grid candidate across trials",
                      "optimizer": "AdamW", "batch_size": 256, "lr_scheduler": "none", "maximum_epochs": 50,
                      "patience": 8, "standardization_source": "train only", "std_correction": 1, "std_floor": 1e-6})
    state = {"classifier": {key: value.cpu() for key, value in best["state"].items()},
             "feature_mean": mean, "feature_std": std, "selection": selection}
    probe_path = output / f"{name}_s1_best_linear_probe.pt"
    torch.save(state, probe_path)
    selection["probe_checkpoint_sha256"] = sha(probe_path)
    write_csv(leaderboard, output / f"{name}_s1_probe_leaderboard.csv")
    write_csv(histories, output / f"{name}_s1_all_trial_metrics.csv")
    write_csv(best["history"], output / f"{name}_s1_selected_probe_metrics.csv")
    atomic_json_dump(selection, output / f"{name}_s1_selection.json")
    return selection


def conclusion(rows: list[dict[str, Any]]) -> str:
    deltas = [float(row["degraded_minus_clean_macro_f1_pp"]) for row in rows]
    if all(delta >= 5.0 for delta in deltas):
        return "Both encoders improve by at least 5 pp at s=1, supporting clean-domain mismatch as an important contributor."
    if all(abs(delta) <= 2.0 for delta in deltas):
        return "Both encoders are within 2 pp across conditions, so this check does not support clean-domain mismatch as the main explanation."
    return "The condition effect is not consistently large for both encoders; this diagnostic alone does not establish clean-domain mismatch as the main explanation."


def report(output: Path, summary_rows: list[dict[str, Any]], endpoints: DegradationEndpoints) -> None:
    headers = "| Encoder | Clean validation macro-F1 | Mixed s=1 validation macro-F1 | Degraded − clean |\n|---|---:|---:|---:|"
    table_rows = "\n".join(
        f"| {row['encoder'].upper()} | {100 * row['clean_val_macro_f1']:.2f}% | {100 * row['s1_val_macro_f1']:.2f}% | {row['degraded_minus_clean_macro_f1_pp']:+.2f} pp |"
        for row in summary_rows
    )
    text = "\n".join([
        "# B0/B1 clean vs maximum-degradation validation sanity check", "", headers, table_rows, "",
        f"Maximum degradation used calibrated defocus radius {endpoints.defocus_radius:g} or resolution factor {endpoints.resolution_factor:g} at s=1.0, assigned once by deterministic class-stratified alternation.",
        "", "## Conclusion", "", conclusion(summary_rows), "",
        "This is a single-seed validation-only diagnostic. Clean values are hash-verified existing final-probe selections; new features and probes use train/validation only. No held-out-test loader, marker, features, predictions, or metrics were created.",
    ])
    (output / "REPORT.md").write_text(text + "\n", encoding="utf-8")


def run(config: dict[str, Any]) -> None:
    output, rows, actions, lock_data = setup(config)
    endpoints = load_endpoints(config["endpoints_json"])
    atomic_json_dump({"endpoints_json_sha256": sha(config["endpoints_json"]), "severity": 1.0,
                      "defocus_radius": endpoints.defocus_radius, "resolution_factor": endpoints.resolution_factor},
                     output / "degradation_audit.json")
    smoke(config, output, rows, actions, lock_data["encoders"], endpoints)
    assert json.loads((output / "smoke_test.json").read_text(encoding="utf-8"))["passed"]
    summary_rows = []
    for name in ENCODERS:
        values, audit = extract_s1_features(config, name, rows, actions, lock_data["encoders"][name]["student_state_sha256"], endpoints, smoke_only=False)
        cache_path = output / f"{name}_s1_train_val_features.npz"
        np.savez_compressed(cache_path, **values)
        audit["cache_sha256"] = sha(cache_path)
        atomic_json_dump(audit, output / f"{name}_s1_feature_extraction_audit.json")
        selection = tune(config, output, name, values, lock_data["encoders"][name])
        clean = lock_data["encoders"][name]["clean_val_macro_f1"]
        degraded = float(selection["val_macro_f1"])
        summary_rows.append({"encoder": name, "clean_val_macro_f1": clean, "s1_val_macro_f1": degraded,
                             "degraded_minus_clean_macro_f1_pp": 100.0 * (degraded - clean),
                             "s1_learning_rate": selection["learning_rate"], "s1_weight_decay": selection["weight_decay"],
                             "s1_best_epoch": selection["best_epoch"]})
    verify_protected_manifest(config["protected_manifest_dir"])
    locks_after = {name: verify_encoder_inputs(config, name, rows) for name in ENCODERS}
    assert locks_after == lock_data["encoders"]
    no_test_artifacts(output)
    write_csv(summary_rows, output / "summary.csv")
    atomic_json_dump({"rows": summary_rows, "conclusion": conclusion(summary_rows), "test_loader_constructed": False,
                      "test_images_decoded": False, "protected_manifest_entries": 129,
                      "protected_artifacts_unchanged": True, "encoder_inputs_unchanged": True}, output / "summary.json")
    report(output, summary_rows, endpoints)
    print(json.dumps(json.loads((output / "summary.json").read_text(encoding="utf-8")), indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/b0_b1_clean_vs_s1_validation_sanity.yaml"))
    args = parser.parse_args()
    run(load_yaml(args.config))


if __name__ == "__main__":
    main()
