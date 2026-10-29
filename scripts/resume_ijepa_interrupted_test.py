#!/usr/bin/env python3
"""Complete the single interrupted I-JEPA classification test, no refitting.

The probe and hyperparameters were frozen before the previous test extraction
failed on operating-system file descriptor exhaustion. This recovers that
interrupted *first* evaluation, and refuses any overwrite/repeat.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from pannuke_ssl.probe import _evaluate
from pannuke_ssl.ssl_framework.downstream import _extract, _split_rows
from pannuke_ssl.ssl_framework.external_datasets import load_external_manifest
from pannuke_ssl.ssl_framework.trainer import load_checkpoint
from pannuke_ssl.ssl_methods.registry import build_method
from pannuke_ssl.utils import atomic_json_dump, seed_everything, write_csv

DATASET = "gashissdb_binary"
EPOCH = 200


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    out = args.output
    if out.name != "epoch_200" or out.parent.name != DATASET:
        raise ValueError("Recovery is restricted to the interrupted GaHISSDB epoch-200 result")
    marker_path = out / "test_started.json"
    probe_path = out / "best_linear_probe.pt"
    selection_path = out / "probe_selection.json"
    provenance_path = out / "dataset_provenance.json"
    for path in (marker_path, probe_path, selection_path, provenance_path):
        if not path.is_file():
            raise RuntimeError(f"Missing required preselected audit artifact: {path}")
    forbidden = (
        "test_metrics.json", "test_predictions.npz", "test_per_class_metrics.csv",
        "test_confusion_matrix.csv", "test_prediction_records.csv",
    )
    if any((out / name).exists() for name in forbidden):
        raise RuntimeError("Test outputs exist; refusing repeat or overwrite")
    if (out / "resume_claim.json").exists():
        raise RuntimeError("This interrupted test has already been claimed for recovery")

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (raw.get("method_name"), raw.get("epoch")) != ("ijepa", EPOCH):
        raise ValueError("Not the approved I-JEPA epoch-200 checkpoint")
    c = raw["config"]
    assert c["seed"] == 20260903 and c["representation"]["pooling"] == "mean_patch_tokens"
    assert c["downstream"]["test_once"] is True
    assert c["downstream"]["probe"]["search"] == "sequential_greedy"
    c["backbone"]["config_dir"] = str(args.model_root / "plip_model")
    c["manifests"]["root"] = str(args.manifest_root)
    marker = json.loads(marker_path.read_text())
    if any((marker.get(key) != value) for key, value in {
        "dataset": DATASET, "encoder_epoch": EPOCH, "probe_selected": True,
        "num_classes": 2,
    }.items()):
        raise ValueError("Interrupted test marker mismatch")
    selection = json.loads(selection_path.read_text())
    if selection.get("test_used") is not False:
        raise RuntimeError("Probe selection was test-influenced")
    state = torch.load(probe_path, map_location="cpu", weights_only=False)
    if selection != state["selection"]:
        raise RuntimeError("Saved probe and selected hyperparameters disagree")

    ds = load_external_manifest(DATASET, args.manifest_root)
    provenance = json.loads(provenance_path.read_text())
    if provenance["manifest_sha256"] != ds.manifest_sha256:
        raise RuntimeError("Dataset changed since probe selection")
    audit = {
        "recovery": "finish_interrupted_first_test_only",
        "reason": "OSError [Errno 24] before test features/predictions were produced",
        "original_marker_sha256": sha(marker_path),
        "selected_probe_sha256": sha(probe_path),
        "selection_sha256": sha(selection_path),
        "manifest_sha256": ds.manifest_sha256,
        "checkpoint_sha256": sha(args.checkpoint),
        "dataset": DATASET,
        "epoch": EPOCH,
        "probe_refitted": False,
        "hyperparameters_changed": False,
        "any_prior_test_outputs": False,
    }
    if args.preflight_only:
        print(json.dumps({"preflight_passed": True, **audit}, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    # Exclusive claim prevents double recovery, even in two independent processes.
    with (out / "resume_claim.json").open("x") as handle:
        json.dump(audit, handle, indent=2)
    seed_everything(int(c["seed"]))
    device = torch.device("cuda")
    method = build_method(c, device)
    loaded = load_checkpoint(method, args.checkpoint, device)
    if int(loaded["epoch"]) != EPOCH:
        raise RuntimeError("Wrong restored checkpoint")
    encoder = method.encoder.eval().to(device)
    encoder.requires_grad_(False)
    rows = _split_rows(c, ds)["test"]
    # Exact original downstream._extract: frozen features, identical 256px
    # preprocessing, BF16, batch=256, workers=8; NO train/val re-evaluation.
    x, y, keys = _extract(encoder, rows, c, device, ds)
    mean, std = state["feature_mean"], state["feature_std"]
    classifier = torch.nn.Linear(mean.shape[1], len(ds.class_names)).to(device).eval()
    classifier.load_state_dict(state["classifier"])
    loss, metrics, preds = _evaluate(classifier, (x - mean) / std, y, device)
    np.savez_compressed(out / "test_predictions.npz",
                        labels=y.numpy(), predictions=preds, keys=keys)
    classes = list(range(len(ds.class_names)))
    report = classification_report(y.numpy(), preds, labels=classes,
                                   target_names=list(ds.class_names),
                                   output_dict=True, zero_division=0)
    write_csv([
        {"class_id": i, "class_name": ds.class_names[i], **report[ds.class_names[i]]}
        for i in classes
    ], out / "test_per_class_metrics.csv")
    np.savetxt(out / "test_confusion_matrix.csv",
               confusion_matrix(y.numpy(), preds, labels=classes),
               delimiter=",", fmt="%d")
    lookup = {int(row["record_index"]): row for row in rows}
    def ref(row):
        if row.get("relative_path"):
            return str(row["relative_path"])
        if row.get("storage") == "parquet":
            return f"{row.get('parquet_file','')}#row_group={row.get('row_group','')}#row={row.get('row_in_group','')}"
        if row.get("storage") == "hdf5":
            return f"{row.get('hdf5_file','')}#index={row.get('hdf5_index','')}"
        return str(row.get("source_name", row.get("record_index", "")))
    write_csv([
        {"record_index": int(k), "sample_ref": ref(lookup[int(k)]),
         "relative_path": lookup[int(k)].get("relative_path", ""),
         "class_id": int(label), "class_name": ds.class_names[int(label)],
         "prediction_id": int(pred), "prediction_name": ds.class_names[int(pred)]}
        for k, label, pred in zip(keys, y.numpy(), preds)
    ], out / "test_prediction_records.csv")
    result = {
        "method": "ijepa", "dataset": DATASET, "encoder_epoch": EPOCH,
        "feature_dim": int(mean.shape[1]), "num_classes": len(ds.class_names),
        "class_names": list(ds.class_names),
        "probe_validation_macro_f1": float(selection["validation_macro_f1"]),
        "test_loss": float(loss), "test": metrics, "test_images": int(y.numel()),
        "test_evaluated_once": True, "dataset_metadata": provenance,
    }
    atomic_json_dump(result, out / "test_metrics.json")
    print(json.dumps({"recovery_complete": True, "test": result["test"],
                      "test_images": result["test_images"]}, indent=2))


if __name__ == "__main__":
    main()
