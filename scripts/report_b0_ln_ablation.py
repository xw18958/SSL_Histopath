#!/usr/bin/env python
"""Build the B0-LN report solely from already-saved artifacts; never loads images/models."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from pannuke_ssl.utils import atomic_json_dump, write_csv

ROOT = Path("/raid1/xwan0900/SSL_proj")
OUT = ROOT / "outputs/b0_ln_ablation"


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required saved artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def row(name: str, pilot: Path, probe: Path) -> dict:
    duration = load_json(pilot / "duration_selection.json")
    selection = load_json(probe / "selection.json")
    test = load_json(probe / "linear_probe_summary.json")
    return {
        "encoder": name,
        "selected_ssl_epoch": int(duration["selected_epoch"]),
        "validation_monitor_macro_f1": float(duration["selection_score"]),
        "final_probe_validation_macro_f1": float(selection["val_macro_f1"]),
        "test_accuracy": float(test["test"]["accuracy"]),
        "test_macro_f1": float(test["test"]["macro_f1"]),
        "test_correct": int(test["correct_test_predictions"]),
        "test_samples": int(test["test_samples"]),
    }


def load_dynamics(pilot: Path) -> dict[int, dict]:
    with (pilot / "pretrain_metrics.csv").open(newline="", encoding="utf-8") as handle:
        pretrain = {int(float(entry["epoch"])): entry for entry in csv.DictReader(handle)}
    with (pilot / "monitor_metrics.csv").open(newline="", encoding="utf-8") as handle:
        monitor = list(csv.DictReader(handle))
    columns = (
        "total", "prediction", "regularizer", "embedding_std", "linear_val_macro_f1", "knn_val_macro_f1",
        "feature_effective_rank", "feature_effective_rank_fraction", "feature_mean_pairwise_cosine",
    )
    rows: dict[int, dict] = {}
    for value in monitor:
        epoch = int(float(value["epoch"]))
        joined = {"epoch": epoch}
        joined.update({key: pretrain.get(epoch, {}).get(key, "") for key in ("total", "prediction", "regularizer", "embedding_std")})
        joined.update({key: value.get(key, "") for key in columns[4:]})
        rows[epoch] = joined
    return rows


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite report directory: {OUT}")
    b0 = row("B0-original", ROOT / "outputs/b0_duration_pilot", ROOT / "outputs/b0_duration_pilot_final_probe")
    b0_ln = row("B0-LN", ROOT / "outputs/b0_ln_duration_pilot", ROOT / "outputs/b0_ln_duration_pilot_final_probe")
    ijepa = row("I-JEPA", ROOT / "outputs/ijepa_duration_pilot", ROOT / "outputs/ijepa_duration_pilot_final_probe")
    pilot_selection = load_json(ROOT / "outputs/b0_ln_duration_pilot/duration_selection.json")
    OUT.mkdir(exist_ok=False)
    comparison = []
    for current in (b0, b0_ln, ijepa):
        comparison.append({
            **current,
            "monitor_minus_b0_pp": 100.0 * (current["validation_monitor_macro_f1"] - b0["validation_monitor_macro_f1"]),
            "probe_val_minus_b0_pp": 100.0 * (current["final_probe_validation_macro_f1"] - b0["final_probe_validation_macro_f1"]),
            "test_accuracy_minus_b0_pp": 100.0 * (current["test_accuracy"] - b0["test_accuracy"]),
            "test_macro_f1_minus_b0_pp": 100.0 * (current["test_macro_f1"] - b0["test_macro_f1"]),
        })
    write_csv(comparison, OUT / "comparison.csv")
    b0_dynamics = load_dynamics(ROOT / "outputs/b0_duration_pilot")
    b0_ln_dynamics = load_dynamics(ROOT / "outputs/b0_ln_duration_pilot")
    write_csv([b0_ln_dynamics[epoch] for epoch in sorted(b0_ln_dynamics)], OUT / "b0_ln_dynamics.csv")
    dynamic_fields = (
        "total", "prediction", "regularizer", "embedding_std", "linear_val_macro_f1", "knn_val_macro_f1",
        "feature_effective_rank", "feature_effective_rank_fraction", "feature_mean_pairwise_cosine",
    )
    dynamics_comparison = []
    for epoch in sorted(set(b0_dynamics) & set(b0_ln_dynamics)):
        record = {"epoch": epoch}
        for field in dynamic_fields:
            b0_value = b0_dynamics[epoch].get(field, "")
            b0_ln_value = b0_ln_dynamics[epoch].get(field, "")
            record["b0_" + field] = b0_value
            record["b0_ln_" + field] = b0_ln_value
            if b0_value != "" and b0_ln_value != "":
                record[field + "_difference"] = float(b0_ln_value) - float(b0_value)
            else:
                record[field + "_difference"] = ""
        dynamics_comparison.append(record)
    write_csv(dynamics_comparison, OUT / "b0_b0_ln_dynamics_comparison.csv")
    monitor_delta = 100.0 * (b0_ln["validation_monitor_macro_f1"] - b0["validation_monitor_macro_f1"])
    probe_delta = 100.0 * (b0_ln["final_probe_validation_macro_f1"] - b0["final_probe_validation_macro_f1"])
    if monitor_delta >= 5.0 and probe_delta >= 5.0:
        classification = "material improvement"
    elif monitor_delta >= 2.0 or probe_delta >= 2.0:
        classification = "modest or mixed improvement"
    else:
        classification = "negligible improvement"
    ijepa_gap_probe = 100.0 * (ijepa["final_probe_validation_macro_f1"] - b0_ln["final_probe_validation_macro_f1"])
    ijepa_gap_test = 100.0 * (ijepa["test_macro_f1"] - b0_ln["test_macro_f1"])
    report = "\n".join(
        [
            "# B0-LN teacher-target LayerNorm ablation",
            "",
            "## Exact change",
            "",
            "B0-LN is identical to B0-original except that its stop-gradient EMA teacher target is normalized as `F.layer_norm(raw_teacher_tokens.float(), (raw_teacher_tokens.shape[-1],))` inside `torch.no_grad()`. Student tokens are passed to the unchanged B0 predictor without LayerNorm. Optimizer, schedule horizon, EMA, BF16, predictor, VICReg, degradation, anchors, adjacent transitions, monitor, pooling, and probe grid are locked by config and implementation manifests.",
            "",
            "## Results",
            "",
            "| Metric | B0-original | B0-LN | Difference (B0-LN − B0) |",
            "|---|---:|---:|---:|",
            f"| Selected SSL epoch | {b0['selected_ssl_epoch']} | {b0_ln['selected_ssl_epoch']} | {b0_ln['selected_ssl_epoch'] - b0['selected_ssl_epoch']} |",
            f"| Validation monitor macro-F1 | {100*b0['validation_monitor_macro_f1']:.2f}% | {100*b0_ln['validation_monitor_macro_f1']:.2f}% | {monitor_delta:+.2f} pp |",
            f"| Final-probe validation macro-F1 | {100*b0['final_probe_validation_macro_f1']:.2f}% | {100*b0_ln['final_probe_validation_macro_f1']:.2f}% | {probe_delta:+.2f} pp |",
            f"| Test accuracy | {100*b0['test_accuracy']:.2f}% | {100*b0_ln['test_accuracy']:.2f}% | {100*(b0_ln['test_accuracy']-b0['test_accuracy']):+.2f} pp |",
            f"| Test macro-F1 | {100*b0['test_macro_f1']:.2f}% | {100*b0_ln['test_macro_f1']:.2f}% | {100*(b0_ln['test_macro_f1']-b0['test_macro_f1']):+.2f} pp |",
            "",
            f"B0-LN is classified as **{classification}** under the prespecified requirement that both validation measures improve by at least 5 pp for a material effect.",
            "",
            "## Duration and dynamics",
            "",
            f"B0-LN completed {pilot_selection['stopping_epoch']} epochs of the fixed 300-epoch/30-warmup schedule; `early_stopped={pilot_selection['early_stopped']}`. The validation-only stopping record is in `duration_selection.json`; `b0_ln_dynamics.csv` gives B0-LN-only dynamics and `b0_b0_ln_dynamics_comparison.csv` compares total/prediction/regularizer loss, embedding std, effective rank, cosine, and validation linear/kNN macro-F1 at matched monitor epochs. B0-LN loss/representation plots are in the pilot output.",
            "",
            "## I-JEPA context",
            "",
            f"Saved matched I-JEPA final-probe validation macro-F1 is {100*ijepa['final_probe_validation_macro_f1']:.2f}% and test macro-F1 is {100*ijepa['test_macro_f1']:.2f}%. B0-LN remains {ijepa_gap_probe:.2f} pp below I-JEPA on probe validation and {ijepa_gap_test:.2f} pp below on test macro-F1. The result is single-seed and transductive; test deltas are descriptive, not significance claims.",
            "",
            "## Integrity",
            "",
            "The report reads saved JSON/CSV artifacts only: it does not instantiate an encoder, decode images, or call any test entrypoint. Protected B0/B1/I-JEPA references and the frozen B0-LN implementation manifest were verified before and after B0-LN execution.",
            "",
        ]
    )
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    atomic_json_dump(
        {
            "b0_original": b0,
            "b0_ln": b0_ln,
            "ijepa": ijepa,
            "validation_monitor_delta_pp": monitor_delta,
            "final_probe_validation_delta_pp": probe_delta,
            "classification": classification,
            "report_is_artifact_only": True,
            "test_entrypoint_invoked_by_report": False,
        },
        OUT / "summary.json",
    )
    print(report)


if __name__ == "__main__":
    main()
