from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Any, Callable

from .config import load_yaml, set_dotted
from .probe import run_linear_probe
from .training import run_pretraining
from .utils import atomic_json_dump, write_csv


METHODS: dict[str, Callable[[dict], dict[str, Any]]] = {"b0": run_pretraining}


def _trials(parameters: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(parameters)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(parameters[key] for key in keys))]


def run_tuning(config: dict) -> dict[str, Any]:
    method_name = str(config["method"])
    if method_name not in METHODS:
        raise ValueError(f"Unknown method {method_name!r}; registered methods: {sorted(METHODS)}")
    base = load_yaml(config["base_config"])
    probe_base = load_yaml(config["probe_config"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, parameters in enumerate(_trials(config["parameters"]), start=1):
        trial_dir = output_dir / f"trial_{index:02d}"
        trial_config = load_yaml(config["base_config"])
        for key, value in parameters.items():
            set_dotted(trial_config, key, value)
        trial_config["seed"] = int(base["seed"]) + index
        trial_config["train"]["epochs"] = int(config["pretrain_epochs"])
        trial_config["output_dir"] = str(trial_dir / "pretrain")
        row: dict[str, Any] = {"trial": index, **parameters, "status": "running"}
        trial_started = time.perf_counter()
        try:
            pretrain_result = METHODS[method_name](trial_config)
            probe_config = dict(probe_base)
            probe_config["seed"] = trial_config["seed"]
            probe_config["encoder_checkpoint"] = pretrain_result["checkpoint"]
            probe_config["output_dir"] = str(trial_dir / "probe")
            probe_result = run_linear_probe(
                probe_config, maximum_epochs_override=int(config["probe_epochs"])
            )
            row.update(
                {
                    "status": "complete",
                    "val_macro_f1": probe_result["selection"]["validation_macro_f1"],
                    "pretrain_checkpoint": pretrain_result["checkpoint"],
                }
            )
        except (FloatingPointError, RuntimeError) as error:
            row.update({"status": "failed", "error": str(error), "val_macro_f1": -1.0})
        row["seconds"] = time.perf_counter() - trial_started
        rows.append(row)
        write_csv(rows, output_dir / "leaderboard.csv")
        print(json.dumps(row, sort_keys=True), flush=True)
    completed = [row for row in rows if row["status"] == "complete"]
    if not completed:
        raise RuntimeError("Every tuning trial failed")
    best = max(completed, key=lambda row: float(row[config["selection_metric"]]))
    best_checkpoint = Path(str(best["pretrain_checkpoint"]))
    for row in completed:
        checkpoint = Path(str(row["pretrain_checkpoint"]))
        if checkpoint != best_checkpoint and checkpoint.exists():
            checkpoint.unlink()
        if row is not best:
            probe_checkpoint = output_dir / f"trial_{int(row['trial']):02d}" / "probe" / "best_linear_probe.pt"
            if probe_checkpoint.exists():
                probe_checkpoint.unlink()
    final_config = load_yaml(config["base_config"])
    for key in config["parameters"]:
        set_dotted(final_config, key, best[key])
    summary = {
        "method": method_name,
        "selection_metric": config["selection_metric"],
        "best_trial": best,
        "final_config": final_config,
        "seconds": time.perf_counter() - started,
    }
    atomic_json_dump(summary, output_dir / "tuning_summary.json")
    atomic_json_dump(final_config, output_dir / "selected_b0_config.json")
    return summary
