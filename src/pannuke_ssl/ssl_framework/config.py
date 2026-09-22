from __future__ import annotations
import copy
from pathlib import Path
from typing import Any
from pannuke_ssl.config import deep_update, load_yaml

ROOT = Path(__file__).resolve().parents[3]


def method_config_path(method: str) -> Path:
    return ROOT / f"configs/ssl_standard/methods/{method}.yaml"


def tuning_config_path(method: str) -> Path:
    return ROOT / f"configs/ssl_standard/tuning/{method}.yaml"


def load_standard_config(method: str) -> dict[str, Any]:
    c = deep_update(load_yaml(ROOT / "configs/ssl_standard/base.yaml"), load_yaml(method_config_path(method)))
    if c["method"]["name"] != method:
        raise ValueError("Method config name mismatch")
    validate_standard_config(c)
    return c


def load_tuning_spec(method: str) -> dict[str, Any]:
    s = load_yaml(tuning_config_path(method))
    if (
        s["method"] != method
        or int(s["budget"]["epochs"]) != 20
        or int(s["budget"]["validation_interval_epochs"]) != 5
    ):
        raise ValueError("Unexpected tuning protocol")

    p = s["parameters"]["learning_rate"]
    candidates = [float(x) for x in p["candidates"]]
    source = float(p["source_value"])
    if source not in candidates:
        raise ValueError(f"Source/recommended LR {source:g} must be included in tuning candidates")

    if "simplex_components" in s["parameters"]:
        if method != "simplex_sigreg_lejepa":
            raise ValueError("simplex_components tuning is reserved for simplex_sigreg_lejepa")
        ks = [int(x) for x in s["parameters"]["simplex_components"]["candidates"]]
        if not ks or any(k < 2 for k in ks) or len(set(ks)) != len(ks):
            raise ValueError("Invalid simplex_components candidates")
        search = s.get("search", {})
        if search.get("strategy") != "sequential_greedy":
            raise ValueError("Simplex tuning must use sequential_greedy search")
        if list(search.get("order", ())) != ["simplex_components", "learning_rate"]:
            raise ValueError("Simplex tuning order must be K first, then learning rate")
        if float(search.get("fixed_simplex_sigma")) != 1.0:
            raise ValueError("Simplex sigma is fixed to 1.0 for this protocol")
        if float(search.get("k_stage_learning_rate")) != source:
            raise ValueError("K-stage learning rate must equal the LeJEPA source learning rate")
    return s


def validate_standard_config(c: dict[str, Any]) -> None:
    data = c["data"]
    if int(c["seed"]) != 20260903:
        raise ValueError("Standard seed changed")
    if int(data.get("source_images", 0)) != 7901:
        raise ValueError("PanNuke source size must remain 7901")
    if int(data["expected_ssl_images"]) != 6305 or data.get("ssl_split") != "train":
        raise ValueError("SSL must use only the 6305-image PanNuke train split")
    if (int(c["training"]["max_epochs"]), int(c["training"]["batch_size"])) != (300, 128):
        raise ValueError("Standard full training must be 300 epochs, batch 128")
    if c["representation"]["pooling"] != "mean_patch_tokens":
        raise ValueError("Standard readout must mean-pool final patch tokens")
    if c["validation"]["selection_metric"] != "linear_val_macro_f1" or int(c["validation"]["interval_epochs"]) != 10:
        raise ValueError("Standard validation protocol changed")
    e = c["early_stopping"]
    if (int(e["min_epochs"]), int(e["patience_monitors"]), float(e["minimum_delta"])) != (60, 5, 0.005):
        raise ValueError("Standard early-stop protocol changed")
    d = c["downstream"]
    if (int(d["train_count"]), int(d["validation_count"]), int(d["test_count"])) != (6305, 798, 798) or not d["test_once"]:
        raise ValueError("Standard downstream protocol changed")
    if c["method"]["name"] == "simplex_sigreg_lejepa" and float(c["method"]["objective"]["simplex_sigma"]) != 1.0:
        raise ValueError("Simplex sigma must remain fixed at 1.0")


def apply_lr(c: dict[str, Any], lr: float) -> dict[str, Any]:
    out = copy.deepcopy(c)
    out["method"]["optimizer"]["peak_lr"] = float(lr)
    return out


def apply_tuned_hyperparameters(c: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    """Apply saved tuning results while enforcing the fixed-sigma Simplex protocol."""
    out = copy.deepcopy(c)
    allowed = {"learning_rate", "simplex_components", "simplex_sigma"}
    unknown = set(selected) - allowed
    if unknown:
        raise ValueError(f"Unknown tuned hyperparameters: {sorted(unknown)}")
    if "learning_rate" in selected:
        out["method"]["optimizer"]["peak_lr"] = float(selected["learning_rate"])
    if "simplex_components" in selected:
        if out["method"]["name"] != "simplex_sigreg_lejepa":
            raise ValueError("simplex_components only applies to simplex_sigreg_lejepa")
        out["method"]["objective"]["simplex_components"] = int(selected["simplex_components"])
    if "simplex_sigma" in selected:
        if out["method"]["name"] != "simplex_sigreg_lejepa":
            raise ValueError("simplex_sigma only applies to simplex_sigreg_lejepa")
        sigma = float(selected["simplex_sigma"])
        if sigma != 1.0:
            raise ValueError("simplex_sigma is fixed to 1.0 and must not be tuned")
        out["method"]["objective"]["simplex_sigma"] = sigma
    return out
