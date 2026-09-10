from __future__ import annotations
import copy
from pathlib import Path
from typing import Any
from pannuke_ssl.config import deep_update, load_yaml

ROOT = Path(__file__).resolve().parents[3]

def method_config_path(method: str) -> Path: return ROOT / f"configs/ssl_standard/methods/{method}.yaml"
def tuning_config_path(method: str) -> Path: return ROOT / f"configs/ssl_standard/tuning/{method}.yaml"

def load_standard_config(method: str) -> dict[str, Any]:
    c=deep_update(load_yaml(ROOT/"configs/ssl_standard/base.yaml"),load_yaml(method_config_path(method)))
    if c["method"]["name"]!=method: raise ValueError("Method config name mismatch")
    validate_standard_config(c); return c

def load_tuning_spec(method: str) -> dict[str, Any]:
    s=load_yaml(tuning_config_path(method))
    if s["method"]!=method or int(s["budget"]["epochs"])!=20 or int(s["budget"]["validation_interval_epochs"])!=5: raise ValueError("Unexpected tuning protocol")
    p=s["parameters"]["learning_rate"]; candidates=[float(x) for x in p["candidates"]]; source=float(p["source_value"])
    if source not in candidates: raise ValueError(f"Source/recommended LR {source:g} must be included in tuning grid")
    return s

def validate_standard_config(c: dict[str, Any]) -> None:
    if int(c["seed"])!=20260903 or int(c["data"]["expected_ssl_images"])!=7901: raise ValueError("Standard seed/data protocol changed")
    if (int(c["training"]["max_epochs"]),int(c["training"]["batch_size"]))!=(300,128): raise ValueError("Standard full training must be 300 epochs, batch 128")
    if c["representation"]["pooling"]!="mean_patch_tokens": raise ValueError("Standard readout must mean-pool final patch tokens")
    if c["validation"]["selection_metric"]!="linear_val_macro_f1" or int(c["validation"]["interval_epochs"])!=10: raise ValueError("Standard validation protocol changed")
    e=c["early_stopping"]
    if (int(e["min_epochs"]),int(e["patience_monitors"]),float(e["minimum_delta"]))!=(60,5,.005): raise ValueError("Standard early-stop protocol changed")
    d=c["downstream"]
    if (int(d["train_count"]),int(d["validation_count"]),int(d["test_count"]))!=(2052,247,247) or not d["test_once"]: raise ValueError("Standard downstream protocol changed")

def apply_lr(c: dict[str, Any], lr: float) -> dict[str, Any]:
    out=copy.deepcopy(c); out["method"]["optimizer"]["peak_lr"]=float(lr); return out
