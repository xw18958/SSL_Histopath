from __future__ import annotations
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def write_final_report(config: dict[str, Any]) -> Path:
    root=Path(config["output"]["root"])/config["method"]["name"]
    tune=_read(root/"tuning/tuning_summary.json")
    pre=_read(root/"pretrain/run_summary.json")
    down=_read(root/"downstream/test_metrics.json")
    lines=[f"# Standard SSL Report — {config['method']['name']}","",f"- SSL images: **{config['data']['expected_ssl_images']}**, unlabeled",f"- Representation: **{config['representation']['pooling']}**",f"- Validation selection: **{config['validation']['selection_metric']}**",f"- Test protocol: **one evaluation after encoder/probe selection**",""]
    if tune:
        lines += ["## Tuning", "", f"- Selected learning rate: **{tune['selected_value']}**", f"- Source/recommended LR included: **{tune['source_value_included']}** (`{tune['source_value']}`)", f"- Best tuning validation macro-F1: **{tune['selected_validation_linear_macro_f1']:.6f}**", ""]
    if pre:
        lines += ["## SSL pretraining", "", f"- Epochs completed: **{pre['epochs_completed']} / {pre['epochs_planned']}**", f"- Stop reason: **{pre['stop_reason']}**", f"- Selected epoch: **{pre['best_epoch']}**", f"- Best validation linear macro-F1: **{pre['best_validation_linear_macro_f1']}**", ""]
    if down:
        t=down["test"]
        lines += ["## Downstream test", "", f"- Accuracy: **{t['accuracy']:.6f}**", f"- Balanced accuracy: **{t['balanced_accuracy']:.6f}**", f"- Macro-F1: **{t['macro_f1']:.6f}**", f"- Weighted F1: **{t['weighted_f1']:.6f}**", f"- Test images: **{down['test_images']}**", ""]
    lines += ["## Method source metadata", "", "```json", json.dumps(config["method"]["source_metadata"],indent=2,sort_keys=True), "```", ""]
    path=root/"FINAL_REPORT.md"; path.parent.mkdir(parents=True,exist_ok=True); path.write_text("\n".join(lines),encoding="utf-8"); return path
