import torch
import pytest

from pannuke_ssl.models import FrozenCONCHVisionEncoder
from pannuke_ssl.probe import _fit_probe


def test_conch_baseline_contract_is_fixed_to_official_linear_probe_readout():
    assert FrozenCONCHVisionEncoder.MODEL_CONFIG == "conch_ViT-B-16"
    assert FrozenCONCHVisionEncoder.IMAGE_SIZE == 256
    assert FrozenCONCHVisionEncoder.FEATURE_DIM == 512


@pytest.mark.skipif(not torch.cuda.is_available(), reason="shared probe fitting requires CUDA")
def test_shared_probe_infers_512_dimensional_feature_width():
    labels = torch.arange(19, dtype=torch.long)
    result = _fit_probe(
        {"train": (torch.randn(19, 512), labels), "val": (torch.randn(19, 512), labels)},
        learning_rate=1e-3,
        weight_decay=0.0,
        maximum_epochs=1,
        patience=1,
        seed=20260903,
    )
    assert result["state"]["weight"].shape == (19, 512)
