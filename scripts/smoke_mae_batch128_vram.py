"""ONE synthetic batch-size-128 forward/backward/optimizer step on GPU.

Only a VRAM/schedule readiness smoke. No images, no epochs, no checkpoints,
no downstream evaluations, and no optimizer iterations beyond one.
"""
from __future__ import annotations
import json
import torch
from smoke_mae_standard import smoke_config
from pannuke_ssl.ssl_methods.registry import build_method
from pannuke_ssl.utils import seed_everything


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    config = smoke_config(False)
    assert int(config["training"]["batch_size"]) == 128
    batch_size = 128
    seed_everything(int(config["seed"]))
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()
    method = build_method(config, torch.device("cuda"))
    optimizer = method.build_optimizer()
    images = torch.rand(batch_size, 3, 256, 256, device="cuda")
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, predictions, mask = method(images)
    assert torch.isfinite(loss)
    assert predictions.shape == (128, 64, 3072)
    assert mask.sum().item() == 128 * 48
    loss.backward()
    grads = [p.grad for p in method.encoder.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all().item() for g in grads)
    optimizer.step()
    torch.cuda.synchronize()
    print(json.dumps({
        "status": "PASS",
        "one_optimizer_step_only": True,
        "experiment_started": False,
        "batch_size": batch_size,
        "loss": float(loss.detach()),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated()/2**30, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved()/2**30, 3),
        "gpu": torch.cuda.get_device_name(),
    }, indent=2))


if __name__ == "__main__":
    main()
