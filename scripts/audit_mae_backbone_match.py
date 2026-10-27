"""CPU audit: same initial encoder backbone for LeJEPA, Simplex and MAE.

No image decoding, training, checkpoint writes, or downstream evaluation.
"""
from __future__ import annotations
import json
import torch
from pannuke_ssl.utils import seed_everything
from pannuke_ssl.ssl_framework.config import load_standard_config
from pannuke_ssl.ssl_framework.data_validation import module_sha
from pannuke_ssl.ssl_methods.registry import build_method

def main():
    records = {}
    for name in ("lejepa", "simplex_sigreg_lejepa", "mae"):
        c = load_standard_config(name)
        seed_everything(int(c["seed"]))
        method = build_method(c, torch.device("cpu"))
        e = method.encoder
        records[name] = {
            "encoder_type": type(e).__name__,
            "encoder_sha256": module_sha(e),
            "patch_size": e.patch_size,
            "hidden_size": e.hidden_size,
            "num_patches": e.num_patches,
            "source_images": c["data"]["final_ssl_images"],
            "epochs": c["training"]["max_epochs"],
            "checkpoints": c["training"]["checkpoint_epochs"],
            "seed": c["seed"],
            "representation": c["representation"],
        }
        del method, e
    for field in records["mae"]:
        assert len({json.dumps(record[field], sort_keys=True) for record in records.values()}) == 1, field
    assert records["mae"]["encoder_type"] == "FreshPLIPVisionEncoder"
    assert records["mae"]["patch_size"] == 32
    assert records["mae"]["num_patches"] == 64
    assert records["mae"]["hidden_size"] == 768
    print(json.dumps({"status": "PASS", "all_fields_match": True, "methods": records}, indent=2))

if __name__ == "__main__":
    main()
