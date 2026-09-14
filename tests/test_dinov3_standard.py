import torch

from pannuke_ssl.ssl_framework import load_standard_config
from pannuke_ssl.ssl_methods.dinov3_standard import (
    DINOHead,
    DINOv3Views,
    MaskingGenerator,
    cross_view_dino_loss,
    sinkhorn_knopp,
)


def test_dinov3_head_and_sinkhorn_are_finite():
    head=DINOHead(16,32,hidden_dim=24,bottleneck_dim=8,nlayers=3)
    logits=head(torch.randn(5,16)); assert logits.shape==(5,32) and torch.isfinite(logits).all()
    probs=sinkhorn_knopp(torch.randn(12,32),.07)
    assert probs.shape==(12,32) and torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(-1),torch.ones(12),atol=1e-4)


def test_cross_view_loss_and_exact_mask_counts():
    teacher=sinkhorn_knopp(torch.randn(8,32),.07).reshape(2,4,32)
    student=torch.randn(8,4,32)
    loss=cross_view_dino_loss(student,teacher,student_temperature=.1,ignore_diagonal=False)
    assert loss.ndim==0 and torch.isfinite(loss)
    masker=MaskingGenerator((8,8))
    for count in (0,4,7,19,32):
        mask=masker(count); assert mask.shape==(64,) and mask.dtype==torch.bool and int(mask.sum())==count


def test_dinov3_view_shapes_match_shared_patch_grid():
    class Dummy(torch.utils.data.Dataset):
        def __len__(self): return 1
        def __getitem__(self,index): return torch.randint(0,256,(3,256,256),dtype=torch.uint8),0,index
    views=DINOv3Views(Dummy(),load_standard_config("dinov3"))[0]
    assert views["global_views"].shape==(2,3,256,256)
    assert views["local_views"].shape==(8,3,96,96)
    assert views["global_views"].dtype==torch.float32 and views["local_views"].dtype==torch.float32
    assert 0.0<=float(views["global_views"].min())<=float(views["global_views"].max())<=1.0
