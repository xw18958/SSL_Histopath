from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from pannuke_ssl.models import FrozenCONCHVisionEncoder
from pannuke_ssl.ssl_framework import load_standard_config
from pannuke_ssl.ssl_framework.downstream import run_frozen_downstream
from pannuke_ssl.ssl_framework.external_datasets import EXTERNAL_DATASETS, PANNUKE_DATASET
from pannuke_ssl.utils import atomic_json_dump


PROJECT_ROOT = Path("/raid1/xwan0900/SSL_proj")
MODEL_ROOT = Path("/raid1/xwan0900/models/conch_v1")
CHECKPOINT = MODEL_ROOT / "pytorch_model.bin"
META = MODEL_ROOT / "meta.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/ssl_standard"
OUTPUT_NAME = "conch_v1_pretrained_attn_pool"
HF_REPOSITORY = "MahmoodLab/CONCH"
HF_REVISION = "f9ca9f877171a28ade80228fb195ac5d79003357"
CHECKPOINT_SHA256 = "40a9644b9ba0e83a74576e0a5e5f7313599fa9c9cdaf3c20f8a3e271b0e9ae7c"
META_SHA256 = "152edc9b784bf2eeef01c7c2991904c2d67b20c4d734e628c54206fbd37bc32f"


def _validate_dataset_mode(dataset: str, *, preflight_only: bool) -> None:
    if preflight_only and dataset != PANNUKE_DATASET:
        raise ValueError("--preflight-only is fixed to the default PanNuke gate; optional datasets are downstream-only")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_artifact_metadata() -> dict[str, Any]:
    for path, expected in ((CHECKPOINT, CHECKPOINT_SHA256), (META, META_SHA256)):
        if not path.is_file():
            raise FileNotFoundError(f"Required pinned CONCH artifact is absent: {path}")
        observed = _sha256(path)
        if observed != expected:
            raise RuntimeError(f"CONCH artifact checksum mismatch for {path}: {observed} != {expected}")
    return {
        "hf_repository": HF_REPOSITORY,
        "hf_revision": HF_REVISION,
        "checkpoint_file": str(CHECKPOINT),
        "checkpoint_bytes": CHECKPOINT.stat().st_size,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "metadata_file": str(META),
        "metadata_sha256": META_SHA256,
    }


def _baseline_config(dataset: str) -> dict[str, Any]:
    config = copy.deepcopy(load_standard_config("simplex_sigreg_lejepa"))
    config["method"]["name"] = OUTPUT_NAME
    config["method"]["source_metadata"] = {
        "implementation": "official frozen CONCH v1 image encoder",
        "model_config": FrozenCONCHVisionEncoder.MODEL_CONFIG,
        "hf_repository": HF_REPOSITORY,
        "hf_revision": HF_REVISION,
        "native_image_size": FrozenCONCHVisionEncoder.NATIVE_IMAGE_SIZE,
        "evaluation_image_size": FrozenCONCHVisionEncoder.IMAGE_SIZE,
        "position_interpolation": "official CONCH create_model_from_pretrained(force_image_size=256) and load_checkpoint resize_pos_embed",
        "readout": "official_attention_pool_before_contrast_projection_and_l2_normalization",
        "official_linear_probe_call": "model.encode_image(images, proj_contrast=False, normalize=False)",
        "encoder_frozen": True,
        "evaluation_dataset": dataset,
        "fairness_protocol": "shared frozen-encoder linear probe: raw 256px images, train-only feature normalization, validation-only selection, and one test decode",
    }
    config["representation"] = {
        "source": "official_conch_v1_attention_pool",
        "pooling": "official_attention_pool_pre_projection",
    }
    config["downstream"]["feature_dim"] = FrozenCONCHVisionEncoder.FEATURE_DIM
    return config


def _preflight(root: Path) -> tuple[FrozenCONCHVisionEncoder, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CONCH preflight")
    artifacts = _local_artifact_metadata()
    encoder = FrozenCONCHVisionEncoder(CHECKPOINT).cuda().eval()
    images = torch.zeros((1, 3, 256, 256), dtype=torch.float32, device="cuda")
    features = encoder(images)
    trainable = [name for name, parameter in encoder.named_parameters() if parameter.requires_grad]
    if features.shape != (1, FrozenCONCHVisionEncoder.FEATURE_DIM):
        raise AssertionError(f"Unexpected CONCH smoke shape: {tuple(features.shape)}")
    if trainable or encoder.training:
        raise AssertionError("CONCH encoder must be frozen and in evaluation mode")
    provenance = {
        **artifacts,
        "model_class": "conch.open_clip_custom.CoCa",
        "official_loader": "conch.open_clip_custom.create_model_from_pretrained",
        "model_config": FrozenCONCHVisionEncoder.MODEL_CONFIG,
        "native_image_size": FrozenCONCHVisionEncoder.NATIVE_IMAGE_SIZE,
        "evaluation_image_size": FrozenCONCHVisionEncoder.IMAGE_SIZE,
        "feature_dim": FrozenCONCHVisionEncoder.FEATURE_DIM,
        "readout": "attention_pooled_pre_projection_pre_normalization",
        "official_linear_probe_call": "model.encode_image(images, proj_contrast=False, normalize=False)",
        "position_interpolation": "official resize_pos_embed while loading the 448px checkpoint into force_image_size=256",
        "normalization": {
            "mean": [float(value) for value in encoder.image_mean.flatten().cpu()],
            "std": [float(value) for value in encoder.image_std.flatten().cpu()],
        },
        "encoder_eval_mode": not encoder.training,
        "encoder_frozen": True,
        "parameters_total": sum(parameter.numel() for parameter in encoder.parameters()),
        "parameters_requires_grad": len(trainable),
    }
    preflight = {
        "input_shape": list(images.shape),
        "feature_shape": list(features.shape),
        "feature_dim": FrozenCONCHVisionEncoder.FEATURE_DIM,
        "attention_pooled": bool(encoder.model.visual.use_attentional_pool_contrast),
        "encoder_eval_mode": True,
        "parameters_requires_grad": 0,
        "checkpoint_verified": True,
        "passed": True,
    }
    root.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(provenance, root / "encoder_provenance.json")
    atomic_json_dump(preflight, root / "preflight.json")
    atomic_json_dump(preflight, root / "smoke_test.json")
    del images, features
    torch.cuda.empty_cache()
    return encoder, provenance


def _run(config: dict[str, Any], encoder: FrozenCONCHVisionEncoder, provenance: dict[str, Any], root: Path, dataset: str) -> dict[str, Any]:
    downstream = root / "downstream" if dataset == PANNUKE_DATASET else root / "downstream_datasets" / dataset
    run_root = root if dataset == PANNUKE_DATASET else downstream.parent
    marker = downstream / "test_started.json"
    metrics = downstream / "test_metrics.json"
    if marker.exists():
        if not metrics.exists():
            raise RuntimeError(f"{marker} exists without completed metrics; refusing to decode the test split again")
        return json.loads(metrics.read_text())
    run_root.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(config, run_root / "resolved_config.json")
    atomic_json_dump(provenance, run_root / "encoder_provenance.json")
    return run_frozen_downstream(config, encoder, downstream, encoder_metadata=provenance, dataset=dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed frozen CONCH v1 linear-probe baseline")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dataset", choices=(PANNUKE_DATASET, *EXTERNAL_DATASETS), default=PANNUKE_DATASET)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = args.output_root / OUTPUT_NAME
    try: _validate_dataset_mode(args.dataset,preflight_only=args.preflight_only)
    except ValueError as error: parser.error(str(error))
    artifact_root = root if args.dataset == PANNUKE_DATASET else root / "downstream_datasets" / args.dataset
    encoder, provenance = _preflight(artifact_root)
    if args.preflight_only:
        print(json.dumps({"preflight": json.loads((root / "preflight.json").read_text())}, indent=2), flush=True)
        return
    result = _run(_baseline_config(args.dataset), encoder, provenance, root, args.dataset)
    print(json.dumps({"result": result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
