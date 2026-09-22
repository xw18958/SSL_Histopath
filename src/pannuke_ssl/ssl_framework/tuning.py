from __future__ import annotations
import math
from pathlib import Path
from typing import Any
import torch, yaml
from pannuke_ssl.utils import atomic_json_dump, write_csv
from .config import apply_tuned_hyperparameters, load_tuning_spec
from .trainer import train_ssl


def _completed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [
        row
        for row in rows
        if row["status"] == "completed" and row["best_validation_linear_macro_f1"] is not None
    ]
    if not valid:
        raise RuntimeError("No tuning candidate completed safely")
    return valid


def _top_ties(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = _completed(rows)
    score = max(float(row["best_validation_linear_macro_f1"]) for row in valid)
    return [
        row
        for row in valid
        if math.isclose(float(row["best_validation_linear_macro_f1"]), score, abs_tol=1e-12)
    ]


def _run_trial(
    c: dict[str, Any],
    *,
    selected: dict[str, Any],
    out: Path,
    stage: str,
    candidate_parameter: str,
    candidate_value: float | int,
    budget: dict[str, Any],
) -> dict[str, Any]:
    trial_config = apply_tuned_hyperparameters(c, selected)
    try:
        trial = train_ssl(
            trial_config,
            out,
            epochs=int(budget["epochs"]),
            interval=int(budget["validation_interval_epochs"]),
            early_stop=bool(budget.get("plateau_early_stopping", False)),
        )
        status = "completed"
        best_epoch = trial["best_epoch"]
        best_score = trial["best_validation_linear_macro_f1"]
        error = None
    except (FloatingPointError, torch.cuda.OutOfMemoryError) as exc:
        torch.cuda.empty_cache()
        status = "failed_safety_stop"
        best_epoch = None
        best_score = None
        error = f"{type(exc).__name__}: {exc}"
    return {
        "stage": stage,
        "candidate_parameter": candidate_parameter,
        "candidate_value": candidate_value,
        "learning_rate": float(selected["learning_rate"]),
        "simplex_components": selected.get("simplex_components"),
        "simplex_sigma": selected.get("simplex_sigma"),
        "status": status,
        "best_epoch": best_epoch,
        "best_validation_linear_macro_f1": best_score,
        "error": error,
    }


def _choose_learning_rate(rows: list[dict[str, Any]], source: float) -> dict[str, Any]:
    tied = _top_ties(rows)
    source_tie = [row for row in tied if math.isclose(float(row["learning_rate"]), source, abs_tol=1e-15)]
    return source_tie[0] if source_tie else tied[0]


def _standard_lr_tuning(c: dict[str, Any], spec: dict[str, Any], root: Path) -> dict[str, Any]:
    p = spec["parameters"]["learning_rate"]
    source = float(p["source_value"])
    lrs = [float(x) for x in p["candidates"]]
    rows = []
    stage_dir = root / "stage_1_learning_rate"
    for lr in lrs:
        rows.append(
            _run_trial(
                c,
                selected={"learning_rate": lr},
                out=stage_dir / f"lr_{lr:.0e}",
                stage="learning_rate",
                candidate_parameter="learning_rate",
                candidate_value=lr,
                budget=spec["budget"],
            )
        )
    chosen = _choose_learning_rate(rows, source)
    write_csv(rows, stage_dir / "summary.csv")
    return {
        "rows": rows,
        "selected_parameters": {"learning_rate": float(chosen["learning_rate"])},
        "selected_row": chosen,
        "stages": [
            {
                "name": "learning_rate",
                "fixed": {},
                "selected_learning_rate": float(chosen["learning_rate"]),
                "selected_validation_linear_macro_f1": float(chosen["best_validation_linear_macro_f1"]),
            }
        ],
        "search_strategy": "single_parameter",
    }


def _simplex_sequential_tuning(c: dict[str, Any], spec: dict[str, Any], root: Path) -> dict[str, Any]:
    p = spec["parameters"]["learning_rate"]
    source = float(p["source_value"])
    lrs = [float(x) for x in p["candidates"]]
    ks = [int(x) for x in spec["parameters"]["simplex_components"]["candidates"]]
    search = spec["search"]
    sigma = float(search["fixed_simplex_sigma"])
    k_stage_lr = float(search["k_stage_learning_rate"])

    k_rows = []
    k_dir = root / "stage_1_K"
    for k in ks:
        k_rows.append(
            _run_trial(
                c,
                selected={
                    "learning_rate": k_stage_lr,
                    "simplex_components": k,
                    "simplex_sigma": sigma,
                },
                out=k_dir / f"K_{k}",
                stage="simplex_components",
                candidate_parameter="simplex_components",
                candidate_value=k,
                budget=spec["budget"],
            )
        )
    chosen_k_row = min(_top_ties(k_rows), key=lambda row: int(row["simplex_components"]))
    chosen_k = int(chosen_k_row["simplex_components"])
    write_csv(k_rows, k_dir / "summary.csv")

    lr_rows = []
    lr_dir = root / "stage_2_learning_rate"
    for lr in lrs:
        if math.isclose(lr, k_stage_lr, rel_tol=0.0, abs_tol=1e-15):
            # The selected-K trial has already evaluated the source learning
            # rate at the selected K. Reuse its validation result so the
            # sequential search executes eight, rather than nine, trials.
            lr_rows.append(
                {
                    **chosen_k_row,
                    "stage": "learning_rate",
                    "candidate_parameter": "learning_rate",
                    "candidate_value": lr,
                    "learning_rate": lr,
                    "simplex_components": chosen_k,
                    "simplex_sigma": sigma,
                    "reused_from_stage": "simplex_components",
                }
            )
            continue
        lr_rows.append(
            _run_trial(
                c,
                selected={
                    "learning_rate": lr,
                    "simplex_components": chosen_k,
                    "simplex_sigma": sigma,
                },
                out=lr_dir / f"lr_{lr:.0e}",
                stage="learning_rate",
                candidate_parameter="learning_rate",
                candidate_value=lr,
                budget=spec["budget"],
            )
        )
    chosen_lr_row = _choose_learning_rate(lr_rows, source)
    chosen_lr = float(chosen_lr_row["learning_rate"])
    write_csv(lr_rows, lr_dir / "summary.csv")

    return {
        "rows": k_rows + lr_rows,
        "selected_parameters": {
            "learning_rate": chosen_lr,
            "simplex_components": chosen_k,
            "simplex_sigma": sigma,
        },
        "selected_row": chosen_lr_row,
        "stages": [
            {
                "name": "simplex_components",
                "fixed_learning_rate": k_stage_lr,
                "fixed_simplex_sigma": sigma,
                "selected_simplex_components": chosen_k,
                "selected_validation_linear_macro_f1": float(chosen_k_row["best_validation_linear_macro_f1"]),
            },
            {
                "name": "learning_rate",
                "fixed_simplex_components": chosen_k,
                "fixed_simplex_sigma": sigma,
                "selected_learning_rate": chosen_lr,
                "selected_validation_linear_macro_f1": float(chosen_lr_row["best_validation_linear_macro_f1"]),
            },
        ],
        "search_strategy": "sequential_greedy",
        "executed_trial_count": len(k_rows) + len(lrs) - 1,
    }


def run_tuning(c: dict[str, Any]):
    spec = load_tuning_spec(c["method"]["name"])
    root = Path(c["output"]["root"]) / c["method"]["name"] / "tuning"
    root.mkdir(parents=True, exist_ok=True)
    if (root / "tuning_summary.json").exists():
        raise FileExistsError("Tuning already exists")

    p = spec["parameters"]["learning_rate"]
    source = float(p["source_value"])
    if c["method"]["name"] == "simplex_sigreg_lejepa":
        search = _simplex_sequential_tuning(c, spec, root)
    else:
        search = _standard_lr_tuning(c, spec, root)

    chosen = search["selected_row"]
    selected_parameters = search["selected_parameters"]
    result = {
        "method": c["method"]["name"],
        "search_strategy": search["search_strategy"],
        "selected_value": float(selected_parameters["learning_rate"]),
        "selected_validation_linear_macro_f1": float(chosen["best_validation_linear_macro_f1"]),
        "source_value": source,
        "source_value_included": True,
        "source_reference": p["source_reference"],
        "selected_parameters": selected_parameters,
        "budget": spec["budget"],
        "stages": search["stages"],
        "rows": search["rows"],
        "executed_trial_count": search.get("executed_trial_count", len(search["rows"])),
        "test_used": False,
    }
    if "simplex_components" in selected_parameters:
        result["selected_simplex_components"] = int(selected_parameters["simplex_components"])
        result["selected_simplex_sigma"] = float(selected_parameters["simplex_sigma"])

    write_csv(search["rows"], root / "tuning_summary.csv")
    atomic_json_dump(result, root / "tuning_summary.json")
    (root / "best_hyperparameters.yaml").write_text(
        yaml.safe_dump(
            {
                "method": c["method"]["name"],
                "selected": selected_parameters,
                "selection": {
                    "metric": "linear_val_macro_f1",
                    "value": result["selected_validation_linear_macro_f1"],
                    "test_used": False,
                    "search_strategy": result["search_strategy"],
                },
                "source_reference": {
                    "learning_rate": source,
                    "included_in_candidates": True,
                    "citation": p["source_reference"],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return result
