#!/usr/bin/env python3
"""Export matched I-JEPA comparison from frozen source JSONs, never test reruns."""
from pathlib import Path
import csv
import hashlib
import json
import collections
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_reports/ijepa_cvpr_20261029"
BASE = ROOT / "benchmark_reports/mae_cvpr_20261027"
SRC = OUT / "source/ijepa"
METHODS = ("lejepa", "simplex_k8", "mae", "ijepa")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ijepa_path(task, dataset, epoch):
    epoch = "epoch_" + str(epoch)
    if task == "classification":
        return SRC / "classification" / dataset / epoch / "test_metrics.json"
    if task == "retrieval":
        return SRC / "retrieval" / dataset / epoch / "test_retrieval_metrics.json"
    if task == "patch":
        return SRC / "patch" / epoch / (dataset + ".json")
    if task == "fewshot":
        return SRC / "fewshot" / epoch / (dataset + ".json")
    if task == "semantic":
        return SRC / "semantic" / dataset / epoch / "result.json"
    if task in ("instance", "instance_f1"):
        return SRC / "instance" / epoch / "result.json"
    raise ValueError(task)


def percent(raw, task, metric):
    if task == "classification":
        return 100 * raw["test"]["accuracy"]
    if task == "retrieval":
        return 100 * raw["test"]["overall_mean_recall"]
    if task == "patch":
        return 100 * raw["retrieval"]["mAP"]
    if task == "fewshot":
        shot = metric.split("-")[0]
        return 100 * raw["results"][shot]["metrics"]["accuracy"]["mean"]
    if task == "semantic":
        test = raw["test"]
        if "testa" in test and "testb" in test:
            return 50 * (test["testa"]["miou"] + test["testb"]["miou"])
        return 100 * test["miou"]
    if task == "instance":
        return 50 * (raw["testA"]["object_dice"] + raw["testB"]["object_dice"])
    if task == "instance_f1":
        return 50 * (raw["testA"]["object_f1"] + raw["testB"]["object_f1"])
    raise ValueError(task)


def export():
    baseline_metrics = list(csv.DictReader((BASE / "comparison_all_metrics.csv").open()))
    audit = json.loads((BASE / "comparison_audit.json").read_text())
    baselines = {(x["task"], x["dataset"], x["epoch"], x["metric"]): x["inputs"]
                 for x in audit["entries"]}
    assert len(baseline_metrics) == len(baselines) == 124
    result_rows = []
    verifications = []
    parity = collections.Counter()
    for row in baseline_metrics:
        task, dataset, epoch, metric = row["task"], row["dataset"], int(row["epoch"]), row["metric"]
        key = (task, dataset, epoch, metric)
        sources = {}
        documents = {}
        for method in METHODS:
            path = ijepa_path(task, dataset, epoch) if method == "ijepa" else ROOT / baselines[key][method]["path"]
            assert path.is_file(), path
            hash_actual = sha256(path)
            if method != "ijepa":
                assert hash_actual == baselines[key][method]["sha256"], (key, method, "source hash")
            sources[method] = {"path": str(path.relative_to(ROOT)), "sha256": hash_actual}
            doc = json.loads(path.read_text())
            documents[method] = doc
            if method != "ijepa":
                source_score = percent(doc, task, metric)
                assert abs(source_score - float(row[method])) < 0.00002, (key, method, source_score, row[method])
        ijepa = documents["ijepa"]
        if task != "instance_f1":
            for method in METHODS[:3]:
                comp = documents[method]
                if task == "classification":
                    for field in ("num_classes", "class_names", "test_images"):
                        assert ijepa[field] == comp[field], (key, method, field)
                    assert ijepa["dataset_metadata"]["manifest_sha256"] == comp["dataset_metadata"]["manifest_sha256"], (key, method, "manifest")
                elif task == "retrieval":
                    for field in ("manifest_sha256", "input_image_files_sha256", "evaluation_protocol_version", "pairs", "protocol_sha256"):
                        assert ijepa[field] == comp[field], (key, method, field)
                elif task == "patch":
                    for field in ("manifest_sha256", "row_order_sha256", "test_indices_sha256", "train_indices_sha256", "support_subsets_sha256", "split_variant"):
                        assert ijepa[field] == comp[field], (key, method, field)
                elif task == "fewshot":
                    for field in ("manifest_sha256", "row_order_sha256", "selected_support_sha256", "shots", "repeats", "protocol"):
                        assert ijepa[field] == comp[field], (key, method, field)
                elif task == "semantic":
                    for field in ("manifest_sha256", "train_images", "val_images", "test_images"):
                        assert ijepa[field] == comp[field], (key, method, field)
                elif task == "instance":
                    for field in ("source_manifest_sha256", "source_splits"):
                        assert ijepa[field] == comp[field], (key, method, field)
            if task == "classification":
                assert ijepa["test_evaluated_once"] is True
            if task == "retrieval":
                assert ijepa["test_evaluated_once_within_protocol"] is True
            if task == "patch":
                assert ijepa["no_test_selection"] is True
            if task == "fewshot":
                assert ijepa["encoder_frozen"] is True
            if task in ("semantic", "instance"):
                assert ijepa["test_used_for_tuning"] is False
            parity[task] += 1
        value = percent(ijepa, task, metric)
        record = {k: row[k] for k in ("task", "dataset", "label", "epoch", "metric", "lejepa", "simplex_k8", "mae")}
        record["ijepa"] = f"{value:.9f}"
        for method in METHODS[:3]:
            record["ijepa_minus_" + method + "_pp"] = f"{(value - float(row[method])):.9f}"
        result_rows.append(record)
        verifications.append({"task": task, "dataset": dataset, "epoch": epoch, "metric": metric, "sources": sources})
    assert len(result_rows) == 124
    assert dict(parity) == {"classification":22, "patch":22, "fewshot":66, "retrieval":4, "semantic":6, "instance":2}, parity
    result_counts = (
        len(list(SRC.glob("classification/*/epoch_*/test_metrics.json"))),
        len(list(SRC.glob("retrieval/*/epoch_*/test_retrieval_metrics.json"))),
        len(list(SRC.glob("patch/epoch_*/*.json"))),
        len(list(SRC.glob("fewshot/epoch_*/*.json"))),
        len(list(SRC.glob("semantic/*/epoch_*/result.json"))),
        len(list(SRC.glob("instance/epoch_*/result.json"))),
    )
    assert result_counts == (22,4,22,22,6,2), result_counts
    pretrain = json.loads((OUT / "experiment_metadata/pretraining_summary.json").read_text())
    assert pretrain["epochs_completed"] == 250 and pretrain["labels_loaded_for_ssl"] is False
    assert pretrain["saved_checkpoint_epochs"] == [100,150,200,250]
    groups = collections.defaultdict(list)
    for row in result_rows:
        groups[(row["task"], row["metric"], int(row["epoch"]))].append(row)
    summary = []
    for (task, metric, epoch), group in sorted(groups.items()):
        item = {"task": task, "metric": metric, "epoch": epoch, "n_datasets": len(group)}
        for method in METHODS:
            item[method] = round(statistics.mean(float(x[method]) for x in group), 6)
        for method in METHODS[:3]:
            item["ijepa_wins_over_" + method] = sum(float(x["ijepa"]) > float(x[method]) for x in group)
        summary.append(item)

    def write_csv(path, rows):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    write_csv(OUT / "comparison_all_metrics.csv", result_rows)
    write_csv(OUT / "comparison_summary.csv", summary)
    (OUT / "comparison_audit.json").write_text(json.dumps({
        "methods": list(METHODS), "epochs": [100,200], "metric_rows": 124,
        "primary_ijepa_results": 78, "parity_validated": dict(parity),
        "sources_and_sha256": verifications, "pretrain_epochs": 250,
        "pretraining_no_labels": True,
        "pretrain_summary_sha256": sha256(OUT/"experiment_metadata/pretraining_summary.json"),
        "tuning_summary_sha256": sha256(OUT/"experiment_metadata/tuning_summary.json"),
    }, indent=2) + "\n")

    title = {"classification": "Classification (accuracy)",
             "retrieval": "Image-text retrieval (mean recall)",
             "patch": "Patch retrieval (mAP)", "fewshot": "Trainable few-shot",
             "semantic": "Semantic segmentation (mIoU)",
             "instance": "GlaS instance (object Dice)",
             "instance_f1": "GlaS instance (object F1)"}
    lines = ["# I-JEPA 100/200-epoch benchmark results", "",
             "Matched comparison: LeJEPA, Simplex-LeJEPA K=8, MAE and I-JEPA.",
             "All scores are percentages. Classification uses test accuracy; image-text retrieval symmetric mean recall; patch retrieval mAP; trainable 1/5/10-shot accuracy (100 fixed support sets); semantic segmentation mIoU (GlaS TestA/TestB averaged equally); GlaS instance segmentation object Dice and F1 (TestA/TestB averaged equally).",
             "",
             "Matched: fresh ViT-B/32 encoder (256px), PanNuke, batch 128, 100/200 checkpoint epochs, frozen downstream encoders and evaluation protocols. I-JEPA predictor depth remains 4. Different native SSL objectives and teacher/masking schemes mean compute/FLOPs are not matched.",
             "",
             "## Unweighted means across datasets", "",
             "| Task | Epoch | N | LeJEPA | Simplex K8 | MAE | I-JEPA | I-JEPA wins vs MAE |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        label = title[s["task"]]
        if s["task"] == "fewshot":
            label += f" ({s['metric'].split('-')[0]} shot)"
        lines.append(f"| {label} | {s['epoch']} | {s['n_datasets']} | {s['lejepa']:.2f} | {s['simplex_k8']:.2f} | {s['mae']:.2f} | {s['ijepa']:.2f} | {s['ijepa_wins_over_mae']}/{s['n_datasets']} |")
    for task in ("classification","retrieval","patch","fewshot","semantic","instance","instance_f1"):
        subset = [r for r in result_rows if r["task"] == task]
        lines += ["", "## " + title[task], "",
                  "| Dataset | LeJEPA 100 | Simplex K8 100 | MAE 100 | I-JEPA 100 | LeJEPA 200 | Simplex K8 200 | MAE 200 | I-JEPA 200 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        unique = sorted({(x["dataset"], x["label"], x["metric"]) for x in subset})
        for ds, label, metric in unique:
            epochs = {int(x["epoch"]): x for x in subset if x["dataset"] == ds and x["metric"] == metric}
            values = [f"{float(epochs[e][m]):.2f}" for e in (100,200) for m in METHODS]
            lines.append("| " + label + (f" ({metric})" if task == "fewshot" else "") + " | " + " | ".join(values) + " |")
    lines += ["", "## Data and audit trail", "",
              "- comparison_all_metrics.csv: 124 individual metric records, plus I-JEPA percentage-point differences.",
              "- comparison_summary.csv: task- and epoch-level dataset means.",
              "- comparison_audit.json: checksums and exact cross-method split, support selection and protocol checks.",
              "- source/ijepa/: all 78 primary I-JEPA JSON results, selection details and dataset provenance.",
              "- experiment_metadata/: tuning, 250-epoch final pretraining, configuration and checkpoint hashes; no weight files.",
              "- Reference results: benchmark_reports/mae_cvpr_20261027/source/.",
              "",
              "This is a single-seed experiment; do not claim multiple-seed evidence."]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print("REPORT_OK",len(result_rows),"PRIMARY_COUNTS",result_counts,"PARITY",dict(parity))
    for s in summary:
        print("SUMMARY",s["task"],s["metric"],s["epoch"],s["ijepa"],s["mae"])


if __name__ == "__main__":
    export()
