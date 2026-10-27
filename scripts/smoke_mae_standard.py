"""ONE GPU optimizer step only. Synthetic by default; optional unlabeled PanNuke train images.

No experiment runs, test-set inspection, stored checkpoints, or downstream evaluation.
"""
from __future__ import annotations
import argparse, copy, json
import torch
from pannuke_ssl.ssl_methods.registry import build_method
from pannuke_ssl.ssl_framework.config import load_standard_config
from pannuke_ssl.ssl_framework.data_validation import build_ssl_loader, module_sha
from pannuke_ssl.utils import seed_everything

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--with-dataset", action="store_true")
    args = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    if not 1 <= args.batch_size <= 4: raise ValueError("Only batch sizes 1..4 allowed")
    c = copy.deepcopy(load_standard_config("mae"))
    c["training"]["batch_size"] = args.batch_size
    c["training"]["num_workers"] = 0
    c["data"]["cache_in_ram"] = False
    seed_everything(int(c["seed"]))
    model = build_method(c, torch.device("cuda"))
    initial_hash = module_sha(model.encoder)
    if args.with_dataset:
        loader = build_ssl_loader(c, model, batch_size=args.batch_size, workers=0)
        assert len(loader.dataset) == 6305
        batch = next(iter(loader))
        assert "label" not in batch and "class_id" not in batch
    else:
        batch = {"image": torch.rand(args.batch_size, 3, 256, 256)}
    images = batch["image"].cuda()
    opt = model.build_optimizer()
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, pred, mask = model(images)
    assert pred.shape == (args.batch_size, 64, 3072)
    assert mask.shape == (args.batch_size, 64) and bool((mask.sum(1) == 48).all())
    assert bool(torch.isfinite(loss))
    loss.backward()
    grads = [p.grad for p in model.backbone.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0
    opt.step()
    with torch.inference_mode():
        tokens = model.encoder(images)
    assert tokens.shape == (args.batch_size, 64, 768)
    assert bool(torch.isfinite(tokens).all())
    peak = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    del model, opt
    torch.cuda.empty_cache()
    seed_everything(int(c["seed"]))
    from pannuke_ssl.models import FreshPLIPVisionEncoder
    reference = FreshPLIPVisionEncoder(c["backbone"]["config_dir"], 256)
    assert module_sha(reference) == initial_hash, "Initial MAE encoder must match baseline"
    print(json.dumps({"status":"PASS", "one_optimizer_step_only":True,
        "experiment_started":False,"samples":args.batch_size,
        "real_unlabeled_train_images":args.with_dataset,"masked_patches":48,
        "loss":float(loss.detach().cpu()),"gradients_finite_and_nonzero":True,
        "initial_encoder_matches_shared_backbone":True,"downstream_test_accessed":False,
        "peak_memory_gib":peak,"gpu":torch.cuda.get_device_name()},indent=2))
if __name__ == "__main__": main()
