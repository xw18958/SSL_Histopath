"""One forward-only audit on synthetic images; no training or evaluation."""
from __future__ import annotations
import json
import torch
from pannuke_ssl.ssl_methods.registry import build_method
from pannuke_ssl.utils import seed_everything
from smoke_mae_standard import smoke_config
from pannuke_ssl.ssl_methods.mae_standard import random_masking


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    c = smoke_config(False)
    seed_everything(int(c["seed"]))
    m = build_method(c, torch.device("cuda")).eval()
    torch.manual_seed(123)
    imgs = torch.rand(2, 3, 256, 256, device="cuda")
    ids_full = torch.arange(64, device="cuda").unsqueeze(0).expand(2, -1)
    # When all patches are visible, masked pretraining path MUST equal
    # the unmasked downstream encoder's final patch representations.
    with torch.inference_mode():
        from_pretrain = m.encode_visible(imgs, ids_full)[:, 1:]
        from_downstream = m.encoder(imgs)
    max_error = (from_pretrain.float() - from_downstream.float()).abs().max().item()
    assert torch.allclose(from_pretrain, from_downstream, rtol=1e-5, atol=1e-5), max_error
    # A masked patch must not influence the latent encoder at all.
    ids_keep, ids_restore, mask = random_masking(2, 64, 0.75, imgs.device)
    pixmask = mask.reshape(2, 8, 8).repeat_interleave(32, 1).repeat_interleave(32, 2).unsqueeze(1).bool()
    masked_changed = torch.where(pixmask, torch.rand_like(imgs), imgs)
    with torch.inference_mode():
        latent_original = m.encode_visible(imgs, ids_keep)
        latent_changed = m.encode_visible(masked_changed, ids_keep)
        decoder_original = m.decode(latent_original, ids_restore)
        decoder_changed = m.decode(latent_changed, ids_restore)
    assert torch.equal(latent_original, latent_changed), "Masked content leaked into encoder"
    assert torch.equal(decoder_original, decoder_changed), "Masked content leaked into decoder"
    print(json.dumps({"status":"PASS","experiment_started":False,
        "masked_pixel_changes_do_not_reach_encoder": True,
        "masked_pixel_changes_do_not_change_decoder_output": True,
        "full_visible_training_path_matches_downstream":True,
        "max_absolute_error":max_error,
        "backbone":"FreshPLIPVisionEncoder",
        "shapes":list(from_pretrain.shape)},indent=2))


if __name__ == "__main__":
    main()
