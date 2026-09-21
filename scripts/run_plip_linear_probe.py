from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from pannuke_ssl.models import PretrainedPLIPVisionEncoder
from pannuke_ssl.ssl_framework import load_standard_config
from pannuke_ssl.ssl_framework.downstream import run_frozen_downstream
from pannuke_ssl.ssl_framework.external_datasets import EXTERNAL_DATASETS, PANNUKE_DATASET
from pannuke_ssl.utils import atomic_json_dump


PROJECT_ROOT = Path("/raid1/xwan0900/SSL_proj")
PLIP_MODEL_DIR = Path("/raid1/xwan0900/models/plip_model")
OUTPUT_ROOT = PROJECT_ROOT / "outputs/ssl_standard"
SIMPLEX_ROOT = OUTPUT_ROOT / "simplex_sigreg_lejepa"
READOUTS = {
    "patch_mean": "plip_pretrained_patch_mean",
    "cls": "plip_pretrained_cls",
}


def _validate_dataset_mode(dataset: str, *, smoke_only: bool) -> None:
    if smoke_only and dataset != PANNUKE_DATASET:
        raise ValueError("--smoke-only is fixed to the default PanNuke gate; optional datasets are downstream-only")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _comparison_metadata() -> dict[str, Any]:
    pretrain = _read_json(SIMPLEX_ROOT / "pretrain/run_summary.json")
    test = _read_json(SIMPLEX_ROOT / "downstream/test_metrics.json")
    if int(pretrain["best_epoch"]) != 260 or int(test["encoder_epoch"]) != 260:
        raise RuntimeError("The fixed Simplex comparator must be its completed epoch-260 checkpoint")
    macro_f1 = float(test["test"]["macro_f1"])
    if abs(macro_f1 - 0.9232786951425002) > 1e-12:
        raise RuntimeError(f"Unexpected fixed Simplex test macro-F1: {macro_f1}")
    return {
        "method": "simplex_sigreg_lejepa",
        "checkpoint": str(SIMPLEX_ROOT / "pretrain/checkpoints/best.pt"),
        "checkpoint_epoch": 260,
        "readout": "mean_final_patch_tokens",
        "test_macro_f1": macro_f1,
        "test_metrics_path": str(SIMPLEX_ROOT / "downstream/test_metrics.json"),
    }


def _baseline_config(method_name: str, readout: str, dataset: str) -> dict[str, Any]:
    config = copy.deepcopy(load_standard_config("simplex_sigreg_lejepa"))
    config["method"]["name"] = method_name
    source_metadata = {
        "implementation": "frozen pretrained PLIP CLIPVisionModel baseline",
        "weights": str(PLIP_MODEL_DIR),
        "source_image_size": 224,
        "evaluation_image_size": 256,
        "position_interpolation": "CLIPVisionModel.interpolate_pos_encoding=True (7x7 to 8x8 patch grid)",
        "readout": "mean_final_patch_tokens" if readout == "patch_mean" else "native_post_layernorm_cls",
        "encoder_frozen": True,
        "evaluation_dataset": dataset,
        "fairness_protocol": "shared frozen-encoder linear probe: train-only normalization, validation-only selection, and one test decode",
    }
    if dataset == PANNUKE_DATASET:
        source_metadata["fairness_protocol"] = "same PanNuke metadata split, train-only normalization, linear 768-to-19 probe grid, validation-only selection, and single test decode as simplex_sigreg_lejepa"
        source_metadata["simplex_comparator"] = _comparison_metadata()
    config["method"]["source_metadata"] = source_metadata
    config["representation"] = {
        "source": "pretrained_plip_final_vision_representation",
        "pooling": "mean_final_patch_tokens" if readout == "patch_mean" else "native_post_layernorm_cls",
    }
    return config


def _encoder_metadata(encoder: PretrainedPLIPVisionEncoder, readout: str) -> dict[str, Any]:
    weights = PLIP_MODEL_DIR / "model.safetensors"
    config = PLIP_MODEL_DIR / "config.json"
    trainable = [name for name, parameter in encoder.named_parameters() if parameter.requires_grad]
    if trainable:
        raise AssertionError(f"PLIP encoder is not fully frozen: {trainable[:5]}")
    return {
        "model_class": "transformers.CLIPVisionModel",
        "weights_directory": str(PLIP_MODEL_DIR),
        "weights_file": str(weights),
        "weights_bytes": weights.stat().st_size,
        "weights_sha256": _sha256(weights),
        "config_file": str(config),
        "config_sha256": _sha256(config),
        "source_image_size": encoder.source_image_size,
        "evaluation_image_size": encoder.image_size,
        "patch_size": encoder.patch_size,
        "hidden_size": encoder.hidden_size,
        "patch_grid": [encoder.image_size // encoder.patch_size, encoder.image_size // encoder.patch_size],
        "position_interpolation": "CLIPVisionModel.interpolate_pos_encoding=True from 224px (7x7) to 256px (8x8)",
        "normalization": {
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
        },
        "readout": "mean_final_patch_tokens" if readout == "patch_mean" else "native_post_layernorm_cls",
        "encoder_eval_mode": not encoder.training,
        "encoder_frozen": True,
        "parameters_total": sum(parameter.numel() for parameter in encoder.parameters()),
        "parameters_requires_grad": len(trainable),
    }


def _run_smoke(readout: str, output_root: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the PLIP smoke test")
    encoder = PretrainedPLIPVisionEncoder(PLIP_MODEL_DIR, image_size=256, readout=readout).cuda().eval()
    images = torch.zeros((1, 3, 256, 256), dtype=torch.float32, device="cuda")
    patches = encoder.patch_tokens(images)
    cls = encoder.cls_features(images)
    features = encoder(images)
    if patches.shape != (1, 64, 768):
        raise AssertionError(f"Expected [1,64,768] patch representation, got {tuple(patches.shape)}")
    if cls.shape != (1, 768) or features.shape != (1, 768):
        raise AssertionError(f"Expected [1,768] CLS/readout representations, got {tuple(cls.shape)}, {tuple(features.shape)}")
    if any(parameter.requires_grad for parameter in encoder.parameters()) or encoder.training:
        raise AssertionError("PLIP smoke test found a trainable or training-mode encoder")
    result = {
        "readout": readout,
        "input_shape": list(images.shape),
        "patch_tokens_shape": list(patches.shape),
        "cls_shape": list(cls.shape),
        "selected_features_shape": list(features.shape),
        "encoder_eval_mode": True,
        "parameters_requires_grad": 0,
        "passed": True,
    }
    root = output_root / READOUTS[readout]
    root.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(result, root / "smoke_test.json")
    del encoder, images, patches, cls, features
    torch.cuda.empty_cache()
    return result


def _run_baseline(readout: str, output_root: Path, dataset: str) -> dict[str, Any]:
    method_name = READOUTS[readout]
    root = output_root / method_name
    downstream = root / "downstream" if dataset == PANNUKE_DATASET else root / "downstream_datasets" / dataset
    run_root = root if dataset == PANNUKE_DATASET else downstream.parent
    existing_metrics = downstream / "test_metrics.json"
    marker = downstream / "test_started.json"
    if marker.exists():
        if not existing_metrics.exists():
            raise RuntimeError(f"{marker} exists but no completed metrics exist; refusing a second test decode")
        return _read_json(existing_metrics)
    run_root.mkdir(parents=True, exist_ok=True)
    config = _baseline_config(method_name, readout, dataset)
    atomic_json_dump(config, run_root / "resolved_config.json")
    encoder = PretrainedPLIPVisionEncoder(PLIP_MODEL_DIR, image_size=256, readout=readout).eval()
    provenance = _encoder_metadata(encoder, readout)
    if dataset == PANNUKE_DATASET:
        provenance["simplex_comparator"] = _comparison_metadata()
    atomic_json_dump(provenance, run_root / "encoder_provenance.json")
    return run_frozen_downstream(config, encoder, downstream, encoder_metadata=provenance, dataset=dataset)


def _write_comparison(output_root: Path) -> dict[str, Any]:
    simplex = _comparison_metadata()
    patch = _read_json(output_root / READOUTS["patch_mean"] / "downstream/test_metrics.json")
    cls = _read_json(output_root / READOUTS["cls"] / "downstream/test_metrics.json")
    simplex_f1 = float(simplex["test_macro_f1"])
    report = {
        "primary_comparison": {
            "protocol": "256px final-patch-token mean pooling with the same fixed downstream protocol",
            "simplex_sigreg_lejepa_epoch_260_macro_f1": simplex_f1,
            "plip_pretrained_patch_mean_macro_f1": float(patch["test"]["macro_f1"]),
            "plip_minus_simplex_macro_f1": float(patch["test"]["macro_f1"]) - simplex_f1,
        },
        "supplementary_plip_native_cls": {
            "protocol": "same 256px downstream protocol; PLIP native post-layernorm CLS readout",
            "plip_pretrained_cls_macro_f1": float(cls["test"]["macro_f1"]),
            "plip_cls_minus_simplex_macro_f1": float(cls["test"]["macro_f1"]) - simplex_f1,
        },
        "simplex_comparator": simplex,
        "outputs": {
            "patch_mean": str(output_root / READOUTS["patch_mean"] / "downstream"),
            "cls": str(output_root / READOUTS["cls"] / "downstream"),
        },
    }
    atomic_json_dump(report, output_root / "plip_pretrained_comparison.json")
    lines = [
        "# Frozen pretrained PLIP linear-probe comparison",
        "",
        "| Result | Test macro-F1 | Delta vs Simplex epoch-260 |",
        "| --- | ---: | ---: |",
        f"| Simplex-SIGReg-LeJEPA (matched patch mean) | {simplex_f1:.5f} | 0.00000 |",
        f"| PLIP pretrained (matched patch mean, primary) | {float(patch['test']['macro_f1']):.5f} | {report['primary_comparison']['plip_minus_simplex_macro_f1']:+.5f} |",
        f"| PLIP pretrained (native CLS, supplementary) | {float(cls['test']['macro_f1']):.5f} | {report['supplementary_plip_native_cls']['plip_cls_minus_simplex_macro_f1']:+.5f} |",
        "",
        "Both PLIP rows use fixed 2,052/247/247 PanNuke splits, train-only feature normalization, the shared 768-to-19 linear-probe grid, validation-only selection, and one test decode.",
    ]
    (output_root / "plip_pretrained_comparison.md").write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen-pretrained PLIP linear probes")
    parser.add_argument("--readout", choices=("all", *READOUTS), default="all")
    parser.add_argument("--dataset", choices=(PANNUKE_DATASET, *EXTERNAL_DATASETS), default=PANNUKE_DATASET)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    try: _validate_dataset_mode(args.dataset,smoke_only=args.smoke_only)
    except ValueError as error: parser.error(str(error))
    selected = tuple(READOUTS) if args.readout == "all" else (args.readout,)
    smokes = {readout: _run_smoke(readout, args.output_root) for readout in selected}
    if args.smoke_only:
        print(json.dumps({"smoke": smokes}, indent=2), flush=True)
        return
    results = {readout: _run_baseline(readout, args.output_root, args.dataset) for readout in selected}
    payload: dict[str, Any] = {"smoke": smokes, "baselines": results}
    if args.readout == "all" and args.dataset == PANNUKE_DATASET:
        payload["comparison"] = _write_comparison(args.output_root)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
