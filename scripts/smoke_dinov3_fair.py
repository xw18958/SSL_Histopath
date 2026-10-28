#!/usr/bin/env python3
"""One synthetic GPU optimizer step; never reads PanNuke labels or test images."""
from __future__ import annotations
import json
import torch
from pannuke_ssl.ssl_framework import load_standard_config
from pannuke_ssl.ssl_framework.trainer import _set_schedule
from pannuke_ssl.ssl_methods.dinov3_standard import StandardDINOv3

def main():
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    torch.manual_seed(20260903)
    config=load_standard_config("dinov3")
    assert config["training"]["batch_size"]==128
    device=torch.device("cuda")
    model=StandardDINOv3(config,device)
    optimizer=model.build_optimizer()
    model.train_mode()
    batch={
        "global_views":torch.rand(2,2,3,256,256),
        "local_views":torch.rand(2,8,3,96,96),
    }
    stats=_set_schedule(model,optimizer,0,250*62)
    assert len(optimizer.param_groups)>2
    assert len(set(round(g["lr_multiplier"],10) for g in optimizer.param_groups))>2
    assert model.student_dino_head.last_layer.weight.requires_grad is False
    assert model.student_ibot_head.last_layer.weight.requires_grad is False
    with torch.autocast("cuda",dtype=torch.bfloat16):
        step=model.training_step(batch,bf16=True)
    assert torch.isfinite(step.loss),step.metrics
    assert step.loss.requires_grad
    step.loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.optimizer_parameters(),3.0)
    assert torch.isfinite(norm),norm
    optimizer.step()
    m=model.after_optimizer_step(0,250*62)
    assert 0.992<=m["ema_momentum"]<=1.0
    assert all(p.grad is None for p in model.teacher.parameters())
    for step_idx in (62,62*40,62*250-1):
        scheduled=_set_schedule(model,optimizer,step_idx,250*62)
        assert all(abs(g["lr"]-scheduled["lr"]*g["lr_multiplier"])<1e-10 for g in optimizer.param_groups)
        assert all(g["lr"]>=0 for g in optimizer.param_groups)
    _set_schedule(model,optimizer,62*2,250*62)
    assert model.student_dino_head.last_layer.weight.requires_grad is True
    assert model.student_ibot_head.last_layer.weight.requires_grad is True
    print(json.dumps({
        "smoke":"PASS","synthetic_image_batch":2,
        "training_epochs_started":0,"total_epochs_configured":250,
        "loss":round(float(step.loss.detach()),6),
        "gradient_norm":round(float(norm),6),
        "optimizer_param_groups":len(optimizer.param_groups),
        "views_per_source":10,
        "dino_prototypes":model.student_dino_head.last_layer.out_features,
        "ibot_prototypes":model.student_ibot_head.last_layer.out_features,
        "peak_cuda_allocated_gib":round(torch.cuda.max_memory_allocated()/2**30,3),
    },sort_keys=True),flush=True)

if __name__=="__main__":main()
