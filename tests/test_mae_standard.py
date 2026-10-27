"""CPU unit tests for MAE masking, patchification, and objective."""
from types import SimpleNamespace
import pytest
import torch
from pannuke_ssl.ssl_methods.mae_standard import StandardMAE, fixed_2d_sincos, patchify, random_masking

def test_random_masks_correctly_invert_shuffling():
    torch.manual_seed(99)
    keep, restore, mask = random_masking(5, 64, .75, torch.device("cpu"))
    assert keep.shape == (5, 16) and restore.shape == mask.shape == (5, 64)
    assert torch.equal(mask.sum(1), torch.full((5,), 48.))
    assert bool((mask.gather(1, keep) == 0).all())
    assert torch.equal(restore.argsort(1)[:, :16], keep)
    assert torch.equal(restore.argsort(1).gather(1, restore), torch.arange(64).expand(5, 64))

@pytest.mark.parametrize("ratio", [0, 1, -0.1, 1.2, .999])
def test_illegal_mask_ratios_rejected(ratio):
    with pytest.raises(ValueError):
        random_masking(2, 64, ratio, torch.device("cpu"))

def test_patchify_layout():
    x = torch.arange(3*256*256).reshape(1, 3, 256, 256).float()
    y = patchify(x, 32)
    assert y.shape == (1, 64, 3072)
    assert torch.equal(y[0, 0, :3], x[0, :, 0, 0])
    assert torch.equal(y[0, 0, 3:6], x[0, :, 0, 1])
    assert torch.equal(y[0, 1, :3], x[0, :, 0, 32])
    assert torch.equal(y[0, 8, :3], x[0, :, 32, 0])

def test_decoder_positions_fixed():
    x = fixed_2d_sincos(8, 512)
    assert x.shape == (1, 65, 512)
    assert bool((x[0, 0] == 0).all()) and not torch.equal(x[0, 1], x[0, 2])

def test_masked_only_mse():
    images = torch.zeros(2, 3, 256, 256)
    predicted = torch.zeros(2, 64, 3072)
    mask = torch.zeros(2, 64)
    mask[:, 20:] = 1
    predicted[:, :20] = 9
    stub = SimpleNamespace(norm_pix_loss=False)
    assert float(StandardMAE.reconstruction_loss(stub, images, predicted, mask)) == 0
    predicted[:, 20:] = 2
    assert float(StandardMAE.reconstruction_loss(stub, images, predicted, mask)) == 4

def test_normalized_target_finite():
    images = torch.rand(2, 3, 256, 256)
    stub = SimpleNamespace(norm_pix_loss=True)
    mse = StandardMAE.reconstruction_loss(stub, images, torch.zeros(2, 64, 3072), torch.ones(2, 64))
    assert torch.isfinite(mse) and .9 < float(mse) < 1.1
