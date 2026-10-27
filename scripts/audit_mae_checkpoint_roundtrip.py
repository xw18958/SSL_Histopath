"""CPU-only integration check for the existing SSL checkpoint serialization.

No training, no evaluation, no permanent checkpoints. Roundtrip into a fresh
MAE model using the exact standard framework checkpoint saver/loader.
"""
from __future__ import annotations
import tempfile
import json
from pathlib import Path
import torch
from pannuke_ssl.ssl_framework.config import load_standard_config
from pannuke_ssl.ssl_framework.trainer import save_checkpoint, load_checkpoint
from pannuke_ssl.ssl_framework.data_validation import module_sha
from pannuke_ssl.ssl_methods.registry import build_method
from pannuke_ssl.utils import seed_everything


def main():
    c = load_standard_config("mae")
    seed_everything(int(c["seed"]))
    m = build_method(c, torch.device("cpu"))
    initial = module_sha(m.encoder)
    with tempfile.TemporaryDirectory(prefix="mae_review_roundtrip_") as tmp:
        checkpoint = Path(tmp) / "checkpoint.pt"
        save_checkpoint(checkpoint, m, c, 0, {"metric": "smoke_only", "test_used": False})
        del m
        seed_everything(int(c["seed"]))
        restored = build_method(c, torch.device("cpu"))
        payload = load_checkpoint(restored, checkpoint, torch.device("cpu"))
        assert payload["method_name"] == "mae" and payload["epoch"] == 0
        assert payload["encoder_sha256"] == initial
        assert module_sha(restored.encoder) == initial
        assert "decoder_pos_embed" in payload["method_state"]
        assert len(restored.decoder_blocks) == 8
    print(json.dumps({"status":"PASS","no_training":True,
                      "checkpoint_strict_load":True,
                      "restored_encoder_sha256":initial,
                      "temporary_checkpoint_cleaned":True},indent=2))


if __name__ == "__main__":
    main()
