import torch
from pannuke_ssl.ijepa_lejepa_fairness_source_fixed import IJEPAOfficialMaskSampler, IJEPAFairPredictor


def _config():
    return {"ijepa": {
        "context_scale": [0.85, 1.0], "target_scale": [0.15, 0.20],
        "target_aspect": [0.75, 1.5], "target_count": 4,
        "minimum_context_tokens": 10, "minimum_block_tokens": 4,
    }}


def test_masking_has_no_context_target_overlap():
    sampler = IJEPAOfficialMaskSampler(_config(), 20260903)
    # Multiple batches exercise mask relaxation/rejection and batch truncation.
    for _ in range(8):
        context, targets = sampler.sample(16, torch.device("cpu"))
        assert context.ndim == 2 and context.shape[0] == 16
        assert targets.shape[:2] == (16, 4)
        assert context.shape[1] > 10 and targets.shape[2] > 4
        assert context.min() >= 0 and context.max() < 64
        assert targets.min() >= 0 and targets.max() < 64
        assert not (context[:, None, :, None] == targets[:, :, None, :]).any()


def test_masking_seed_is_reproducible():
    a = IJEPAOfficialMaskSampler(_config(), 7).sample(4)
    b = IJEPAOfficialMaskSampler(_config(), 7).sample(4)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_predictor_keeps_four_layers_and_valid_output():
    m = IJEPAFairPredictor(num_heads=12, predictor_dim=384, predictor_depth=4, mlp_ratio=4)
    assert len(m.transformer.layers) == 4
    assert m.input_projection.out_features == 384
    x = torch.randn(2, 12, 768)
    context = torch.arange(12).expand(2, -1)
    targets = torch.arange(12, 24).reshape(1, 4, 3).expand(2, -1, -1)
    y = m(x, context, targets)
    assert y.shape == (2, 4, 3, 768)
    assert torch.isfinite(y).all()
