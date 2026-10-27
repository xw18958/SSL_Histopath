"""Protocol-preserving adapters for MAE: no downstream models are retrained here."""
from __future__ import annotations
import ast
from pathlib import Path
import pytest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from patch_benchmark_mae import mae_checkpoint_path
from pannuke_ssl.ssl_framework.config import load_standard_config
from pannuke_ssl.ssl_methods.registry import build_method

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("filename",[
    "fewshot_linear_gpu_v2.py", "run_pathology_segmentation.py",
    "run_bcss5_official.py", "run_glas_instance.py",
])
def test_mae_is_an_allowed_downstream_method(filename):
    tree = ast.parse((ROOT/"scripts"/filename).read_text())
    assert "mae" in ast.unparse(tree)

def test_mae_checkpoint_path_enforces_approved_epoch_and_root(monkeypatch,tmp_path):
    monkeypatch.delenv("SSL_MAE_CKPT_ROOT",raising=False)
    with pytest.raises(OSError, match="SSL_MAE_CKPT_ROOT"):
        mae_checkpoint_path("mae",100)
    monkeypatch.setenv("SSL_MAE_CKPT_ROOT",str(tmp_path))
    with pytest.raises(FileNotFoundError):
        mae_checkpoint_path("mae",100)
    with pytest.raises(ValueError):
        mae_checkpoint_path("mae",250)
    checkpoint=tmp_path/"epoch_200.pt"
    checkpoint.write_bytes(b"test-file-presence-only")
    assert mae_checkpoint_path("mae",200)==checkpoint

def test_baseline_frozen_settings_identical_for_mae():
    x=load_standard_config("mae")
    y=load_standard_config("lejepa")
    for key in ("seed","data","backbone","training","representation","downstream"):
        assert x[key] == y[key]
    assert x["method"]["objective"]["mask_ratio"]==.75
