import torch
import pytest

from pannuke_ssl.models import PretrainedPLIPVisionEncoder
from pannuke_ssl.ssl_framework.downstream import run_frozen_downstream


def test_pretrained_plip_readouts_are_explicit_and_fixed():
    assert PretrainedPLIPVisionEncoder.READOUTS == frozenset(("patch_mean", "cls"))


def test_frozen_downstream_rejects_trainable_encoders(tmp_path):
    encoder = torch.nn.Linear(768, 768).eval()
    with pytest.raises(AssertionError, match="trainable parameters"):
        run_frozen_downstream({}, encoder, tmp_path, encoder_metadata={})
