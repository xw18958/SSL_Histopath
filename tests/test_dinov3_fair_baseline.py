"""DINOv3 fairness and optimization checks (no training jobs started)."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from pannuke_ssl.ssl_framework import load_standard_config
from pannuke_ssl.ssl_framework.trainer import _set_schedule
from pannuke_ssl.ssl_methods.dinov3_standard import StandardDINOv3

ROOT = Path(__file__).resolve().parents[1]

def test_common_setting_identity():
    dino=load_standard_config("dinov3")
    lejepa=load_standard_config("lejepa")
    mae=load_standard_config("mae")
    for shared in ("seed","data","backbone","training","representation","downstream"):
        assert dino[shared]==lejepa[shared]==mae[shared],shared
    assert dino["training"]["max_epochs"]==250
    assert dino["training"]["checkpoint_epochs"]==[100,150,200,250]
    assert dino["data"]["final_ssl_images"]==7901
    assert dino["method"]["views"]["global_count"]==2
    assert dino["method"]["views"]["local_count"]==8
    assert dino["method"]["optimizer"]["layerwise_decay"]==0.9
    assert dino["method"]["optimizer"]["patch_embed_lr_multiplier"]==0.2

def test_tuning_source_batch_scaling():
    from pannuke_ssl.ssl_framework.config import load_tuning_spec
    x=load_tuning_spec("dinov3")
    lr=x["parameters"]["learning_rate"]
    assert lr["source_value"]==pytest.approx(0.001)
    assert any(v==pytest.approx(0.001*math.sqrt(128/1024)) for v in lr["candidates"])
    assert len(lr["candidates"])==3
    assert lr["source_value"] in lr["candidates"]
    assert x["budget"]["epochs"]==20

def test_schedule_keeps_legacy_methods_unchanged():
    class Method:
        def schedule(self,step,total):
            return {"lr":0.0002,"weight_decay":0.3}
    a=torch.nn.Parameter(torch.ones(()))
    b=torch.nn.Parameter(torch.ones(()))
    optimizer=torch.optim.AdamW(
       [{"params":[a]}, {"params":[b],"lr_multiplier":0.5,"wd_multiplier":0.25}],
       lr=1e-4,weight_decay=0.01)
    _set_schedule(Method(),optimizer,0,100)
    assert optimizer.param_groups[0]["lr"]==pytest.approx(0.0002)
    assert optimizer.param_groups[0]["weight_decay"]==pytest.approx(0.3)
    assert optimizer.param_groups[1]["lr"]==pytest.approx(0.0001)
    assert optimizer.param_groups[1]["weight_decay"]==pytest.approx(0.075)

def test_dino_prototypes_and_optimizer_groups():
    c=load_standard_config("dinov3")
    model=StandardDINOv3(c,torch.device("cpu"))
    optimizer=model.build_optimizer()
    student={id(p):name for name,p in model.student.named_parameters()}
    expected_ids={id(p) for p in model.optimizer_parameters()}
    groups=[p for g in optimizer.param_groups for p in g["params"]]
    assert len(groups)==len(expected_ids)==len({id(p) for p in groups})
    assert {id(p) for p in groups}==expected_ids
    def params_matching(text):
        return [p for g in optimizer.param_groups for p in g["params"]
                if id(p) in student and text in student[id(p)]]
    def group_for(param):
        return next(g for g in optimizer.param_groups
                    if any(p is param for p in g["params"]))
    patch=params_matching("patch_embedding.weight")
    assert len(patch)==1, list(student.values())[:10]
    g=group_for(patch[0])
    assert g["lr_multiplier"]==pytest.approx((0.9**13)*0.2)
    assert g["wd_multiplier"]==1.0
    first=params_matching("encoder.layers.0.self_attn.q_proj.weight")
    assert len(first)==1
    assert group_for(first[0])["lr_multiplier"]==pytest.approx(0.9**12)
    last=params_matching("encoder.layers.11.self_attn.q_proj.weight")
    assert len(last)==1
    assert group_for(last[0])["lr_multiplier"]==pytest.approx(0.9)
    norm=params_matching("post_layernorm.weight")
    assert len(norm)==1
    assert group_for(norm[0])["wd_multiplier"]==0.
    # A fixed source DINOv3 setting, not 3 different experiments.
    assert model.student_dino_head.last_layer.out_features==65536
    assert model.student_ibot_head.last_layer.out_features==65536

def test_dinov3_adapters_and_existing_mae_unchanged():
    src=(ROOT/"scripts/run_dinov3_selected_downstreams.py").read_text()
    assert "dinov3" in src and "mae" not in src
    assert "SSL_DINOV3_CKPT_ROOT" in src
    assert "DATASETS" in src and "EPOCHS = (100, 200)" in src
    patch=(ROOT/"scripts/patch_benchmark_dinov3.py").read_text()
    assert 'benchmark.extract_one(args.out,"dinov3"' in patch
    assert "SSL_DINOV3_CKPT_ROOT" in patch
    for name in ("run_mae_selected_downstreams.py","patch_benchmark_mae.py"):
        assert (ROOT/"scripts"/name).is_file()


def test_selected_dataset_protocol_matches_mae():
    import sys
    scripts=str(ROOT/"scripts")
    if scripts not in sys.path:
        sys.path.insert(0,scripts)
    from run_mae_selected_downstreams import DATASETS as mae_datasets, EPOCHS as mae_epochs
    from run_dinov3_selected_downstreams import DATASETS as dino_datasets, EPOCHS as dino_epochs
    assert dino_datasets==mae_datasets
    assert dino_epochs==mae_epochs==(100,200)

def test_prepared_launch_gates_are_disabled_until_explicit_execution():
    import subprocess
    for name in ("approved_dinov3_pretrain_after_tuning.sh",
                 "approved_dinov3_downstreams_after_pretrain.sh"):
        path=ROOT/"scripts"/name
        assert path.is_file()
        subprocess.run(["bash","-n",str(path)],check=True)
        script=path.read_text()
        assert "DINOV3_CODE_ROOT" in script and "DINOV3_RUN_ROOT" in script
        assert "mae_cvpr" not in script
        assert "dinov3_cvpr_20261028" in script
    train=(ROOT/"scripts/approved_dinov3_pretrain_after_tuning.sh").read_text()
    assert "0.0003535533905932738" in train
    assert "0.0001" in train
    assert "0.001" in train
