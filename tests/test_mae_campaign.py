"""Approved MAE campaign checks; no model execution."""
import sys
from pathlib import Path
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from run_mae_selected_downstreams import DATASETS,EPOCHS,checked_checkpoint,existing_done

def test_requested_downstream_data_exactly_11_selected_datasets():
    assert len(DATASETS)==len(set(DATASETS))==11
    assert EPOCHS==(100,200)
    for name in ("pcam_binary","ebhi_seg_6class","osteosarcoma_3class","wsss4luad_3class"):
        assert name not in DATASETS

def test_checkpoint_method_epoch_framework_guard(tmp_path):
    f=tmp_path/"epoch_100.pt"
    torch.save({"framework":"ssl_standard_v1","method_name":"lejepa",
                "epoch":100,"config":{"method":{"name":"lejepa"}}},f)
    with pytest.raises(ValueError,match="mismatch"):
        checked_checkpoint(tmp_path,100)
    with pytest.raises(FileNotFoundError):
        checked_checkpoint(tmp_path,200)

def test_test_once_marker_prevents_overwrite(tmp_path):
    p=tmp_path/"task"
    assert not existing_done(p,"result.json")
    p.mkdir()
    (p/"test_started.json").write_text("{}")
    with pytest.raises(RuntimeError,match="partially executed"):
        existing_done(p,"result.json")
