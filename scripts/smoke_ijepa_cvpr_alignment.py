"""I-JEPA parity and target-leakage smoke for the shared CVPR baseline."""
from __future__ import annotations
import copy
import json
import torch
from pannuke_ssl.ssl_framework import build_ssl_loader, load_standard_config
from pannuke_ssl.ssl_methods.registry import build_method
from pannuke_ssl.ijepa import encode_context, objective
from pannuke_ssl.models import update_ema
from pannuke_ssl.utils import seed_everything


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU smoke requires CUDA")
    c = load_standard_config("ijepa")
    assert c["method"]["predictor"]["depth"] == 4
    assert c["training"]["max_epochs"] == 250
    assert c["training"]["checkpoint_epochs"] == [100, 150, 200, 250]
    seed_everything(int(c["seed"]))
    temp = copy.deepcopy(c)
    temp["data"]["cache_in_ram"] = False
    temp["training"]["num_workers"] = 0
    method = build_method(temp, torch.device("cuda"))
    loader = build_ssl_loader(temp, method, batch_size=2, workers=0)
    batch = next(iter(loader))
    images = batch["image"].cuda().float().div(255.)
    context, targets = method.masker.sample(2, torch.device("cuda"))
    assert not (context[:, None, :, None] == targets[:, :, None, :]).any()
    method.train_mode()
    with torch.no_grad():
        original = encode_context(method.student, images, context)
        altered = images.clone()
        for i in range(2):
            for t in targets[i].flatten().tolist():
                y, x = divmod(t, 8)
                altered[i, :, y*32:(y+1)*32, x*32:(x+1)*32] = 0.123
        changed = encode_context(method.student, altered, context)
        assert torch.equal(original, changed), "Masked targets changed the context representation"
    optimizer = method.build_optimizer()
    optimizer.zero_grad(set_to_none=True)
    result = method.training_step(batch, bf16=True)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert all(p.grad is None for p in method.teacher.parameters())
    assert method.student.vision_backbone.embeddings.patch_embedding.weight.grad is not None
    assert method.predictor.input_projection.weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in method.optimizer_parameters() if p.grad is not None)
    with torch.no_grad():
        first_teacher = next(method.teacher.parameters()).clone()
    optimizer.step()
    update_ema(method.student, method.teacher, 0.996)
    assert torch.allclose(next(method.teacher.parameters()), .996*first_teacher + .004*next(method.student.parameters()), atol=1e-7, rtol=1e-6)
    print(json.dumps({
        "passed": True, "predictor_depth": len(method.predictor.transformer.layers),
        "mask_context_tokens": context.shape[1],
        "mask_target_tokens_per_block": targets.shape[2],
        "context_target_overlap": 0, "masked_pixels_do_not_change_context": True,
        "objective": "smooth_l1", "ema_teacher_frozen": True,
        "gradients_finite": True, "ema_update_verified": True,
        "source_images": len(loader.dataset),
    }, indent=2))


if __name__ == "__main__":
    main()
