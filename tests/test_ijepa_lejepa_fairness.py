import torch

from pannuke_ssl.ijepa_lejepa_fairness_source import (
    IJEPAFairPredictor,
    SlicedEppsPulley,
)


def test_fair_ijepa_predictor_depth_heads_and_shape():
    predictor = IJEPAFairPredictor(num_heads=12).eval()
    assert len(predictor.transformer.layers) == 4
    assert predictor.transformer.layers[0].self_attn.num_heads == 12
    tokens = torch.randn(2, 16, 768)
    context = torch.arange(16).repeat(2, 1)
    targets = torch.tensor(
        [
            [[16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 26, 27]],
            [[28, 29, 30], [31, 32, 33], [34, 35, 36], [37, 38, 39]],
        ],
        dtype=torch.long,
    )
    output = predictor(tokens, context, targets)
    assert output.shape == (2, 4, 3, 768)
    assert torch.isfinite(output).all()


def test_released_sigreg_is_finite_and_has_gradients():
    sigreg = SlicedEppsPulley(num_slices=32, t_max=3.0, n_points=17)
    x = torch.randn(48, 32, requires_grad=True)
    loss = sigreg(x)
    assert torch.isfinite(loss)
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
