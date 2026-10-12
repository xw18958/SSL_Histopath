#!/usr/bin/env python3
"""Audited three-model result comparison at common K=8 epochs.

LeJEPA/K64: five checkpoints (100,150,200,250,300)
K8: original four checkpoints (100,150,200,250); 300 is unavailable.
No model selection, evaluation recomputation from test labels, or pretraining.
"""
from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

import numpy as np

from patch_benchmark import (
    EXTERNAL_DATASETS, EPOCHS, DEFAULT_OUT, dump_json, sha_json
)

MODELS = ("lejepa", "simplex_k64", "simplex_k8")
K8_EPOCHS = (100, 150, 200, 250)
RETRIEVAL = ("mAP", "macro_mAP", "Acc@1", "Acc@3", "Acc@5", "majority_vote_Acc@5")
FEWSHOT = ("macro_f1", "balanced_accuracy", "accuracy")
FAIRNESS_FIELDS = (
    "manifest_sha256", "row_order_sha256", "test_indices_sha256",
    "train_indices_sha256", "support_subsets_sha256", "feature_dim",
    "split_variant",
)


def read_result(out, model, epoch, dataset):
    path = out / "results/full/main" / model / f"epoch_{epoch}" / f"{dataset}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing full benchmark result {path}")
    r = json.loads(path.read_text())
    if (r["method"], r["epoch"], r["dataset"], r["split_variant"],
        r["smoke_per_class"]) != (model, epoch, dataset, "main", 0):
        raise ValueError("Result identity/protocol mismatch " + str(path))
    if not all(r[k] for k in FAIRNESS_FIELDS):
        raise RuntimeError("Missing input/provenance hash " + str(path))
    if r.get("no_test_selection") is not True or r.get("gallery_train_only") is not True:
        raise RuntimeError("Leaky train/test protocol flag " + str(path))
    if len(r["support_subsets"]) != 3 or any(
            len(r["support_subsets"][str(shot)]) != 100 for shot in (1,5,10)):
        raise RuntimeError("Not 100 predetermined supports per shot")
    if sha_json(r["support_subsets"]) != r["support_subsets_sha256"]:
        raise RuntimeError("Support subset fingerprint mismatch")
    return r


def check_same_inputs(rows, epoch, dataset):
    base = rows["lejepa"]
    for method, result in rows.items():
        for field in FAIRNESS_FIELDS:
            if result[field] != base[field]:
                raise RuntimeError(
                    f"UNFAIR: {dataset} epoch {epoch}, {method}: {field}")
        if result["support_subsets"] != base["support_subsets"]:
            raise RuntimeError(f"Support image IDs differ: {dataset}/{epoch}/{method}")
        if result["retrieval"]["num_gallery"] != base["retrieval"]["num_gallery"]:
            raise RuntimeError("Different retrieval galleries")
        if result["retrieval"]["num_queries"] != base["retrieval"]["num_queries"]:
            raise RuntimeError("Different retrieval test sets")
        for shot in ("1", "5", "10"):
            for metric in FEWSHOT:
                if len(result["fewshot"][shot]["per_subset"][metric]) != 100:
                    raise RuntimeError("Incorrect few-shot score count")
                mean = float(np.mean(result["fewshot"][shot]["per_subset"][metric]))
                if not np.isclose(mean,
                                  result["fewshot"][shot]["metrics"][metric]["mean"],
                                  rtol=0, atol=1e-12):
                    raise RuntimeError("Per-subset score mean differs from summary")


def generate_threeway(out=DEFAULT_OUT):
    out = Path(out)
    baseline_check = out / "reports/full/main/verification.json"
    if not baseline_check.is_file():
        raise FileNotFoundError("Missing verified 150-job LeJEPA/K64 report")
    baseline_status = json.loads(baseline_check.read_text())
    if not baseline_status["final_complete"] or baseline_status["completed_jobs"] != 150:
        raise RuntimeError("Original LeJEPA/K64 run incomplete")
    all_results = {}
    per_dataset = []

    for epoch in EPOCHS:
        methods = MODELS if epoch in K8_EPOCHS else MODELS[:2]
        for dataset in EXTERNAL_DATASETS:
            rows = {method: read_result(out, method, epoch, dataset) for method in methods}
            check_same_inputs(rows, epoch, dataset)
            all_results[(epoch, dataset)] = rows
            for task, shot, metric in (
                [("retrieval", "all", m) for m in RETRIEVAL] +
                [("fewshot", f"{s}-shot", m) for s in (1, 5, 10) for m in FEWSHOT]
            ):
                record = {
                    "dataset": dataset, "epoch": epoch,
                    "task": task, "setting": shot, "metric": metric,
                    "checkpoint_300_K8": "not_saved" if epoch == 300 else "available",
                }
                for model in MODELS:
                    row = rows.get(model)
                    if row is None:
                        record[model] = None
                        if task == "fewshot":
                            record[model+"_std"] = None
                            record[model+"_ci95_low"] = None
                            record[model+"_ci95_high"] = None
                        continue
                    if task == "retrieval":
                        record[model] = float(row["retrieval"][metric])
                    else:
                        summary = row["fewshot"][shot.split("-")[0]]["metrics"][metric]
                        record[model] = float(summary["mean"])
                        record[model+"_std"] = float(summary["std"])
                        record[model+"_ci95_low"] = float(summary["ci95_lower"])
                        record[model+"_ci95_high"] = float(summary["ci95_upper"])
                for left, right, field in (
                    ("simplex_k64", "lejepa", "k64_minus_lejepa_pp"),
                    ("simplex_k8", "lejepa", "k8_minus_lejepa_pp"),
                    ("simplex_k8", "simplex_k64", "k8_minus_k64_pp"),
                ):
                    record[field] = (
                        100*(record[left]-record[right]) if
                        record[left] is not None and record[right] is not None
                        else None
                    )
                per_dataset.append(record)

    # Enforce exact coverage (including original-only epoch300).
    assert len(all_results) == 5 * 15
    assert sum("simplex_k8" in rows for rows in all_results.values()) == 4 * 15
    assert len(per_dataset) == 5 * 15 * (6 + 9)

    groups = collections.defaultdict(list)
    for row in per_dataset:
        groups[(row["epoch"], row["task"], row["setting"], row["metric"])].append(row)
    summary = []
    for (epoch, task, setting, metric), records in sorted(groups.items()):
        if len(records) != 15:
            raise RuntimeError("Aggregate does not cover all 15 datasets")
        values = {
            model: float(np.mean([r[model] for r in records]))
            if all(r[model] is not None for r in records) else None
            for model in MODELS
        }
        row = {
            "epoch": epoch, "task": task, "setting": setting, "metric": metric,
            "dataset_count": len(records), **values
        }
        for left, right, field in (
            ("simplex_k64", "lejepa", "k64_minus_lejepa_pp"),
            ("simplex_k8", "lejepa", "k8_minus_lejepa_pp"),
            ("simplex_k8", "simplex_k64", "k8_minus_k64_pp"),
        ):
            row[field] = (
                100*(values[left]-values[right])
                if values[left] is not None and values[right] is not None
                else None
            )
            if row[field] is not None and not np.isclose(
                row[field],
                np.mean([r[field] for r in records]),
                atol=1e-9, rtol=0):
                raise RuntimeError("Independent aggregate discrepancy")
        summary.append(row)
    audit = json.loads((out / "audit.json").read_text())
    if len(audit) != 15 or any(
            a["pixel_duplicate_cross_split_count"] != 0 for a in audit):
        raise RuntimeError("Audit is incomplete")
    for row in per_dataset:
        x = next(a for a in audit if a["dataset"] == row["dataset"])
        row["source_independence"] = x["source_independence"]

    folder = out / "reports/full/main/three_methods"
    folder.mkdir(parents=True, exist_ok=True)
    dump_json(folder/"per_dataset.json", per_dataset)
    dump_json(folder/"summary.json", summary)
    verification = {
        "all_input_identity_checked": True,
        "all_support_selections_identical": True,
        "frozen_backbone_no_retraining": True,
        "dataset_count": 15,
        "methods": list(MODELS),
        "shared_epochs": list(K8_EPOCHS),
        "lejepa_k64_epochs": list(EPOCHS),
        "k8_epoch_300_status": "unavailable_not_trained_no_interpolation",
        "baseline_completed_jobs": 150,
        "new_k8_completed_jobs": 60,
        "shared_epoch_three_model_results": 180,
        "total_method_checkpoint_dataset_results": 210,
        "per_dataset_metric_rows": len(per_dataset),
        "aggregate_metric_rows": len(summary),
        "checkpoint_300_k8_is_missing": True,
        "full_three_method_common_epoch_comparison": True,
        "aggregate_independently_verified": True,
    }
    dump_json(folder/"verification.json", verification)
    for filename, data in (("per_dataset.csv", per_dataset), ("summary.csv", summary)):
        with (folder/filename).open("w", newline="") as h:
            fieldnames = list(dict.fromkeys(key for row in data for key in row))
            writer = csv.DictWriter(h, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps({"three_way_rows":len(per_dataset),
                      "mean_rows":len(summary),
                      "verified_jobs":210,
                      "report_directory":str(folder)}, indent=2),flush=True)
    return summary


if __name__ == "__main__":
    generate_threeway()
