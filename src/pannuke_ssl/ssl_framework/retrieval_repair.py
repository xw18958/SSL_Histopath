"""Repair retrieval without retraining SSL or touching classification outputs.

TRAIN/VAL selection and TEST evaluation are separate actions. Dataset manifests
remain immutable; the effective evaluation protocol is versioned separately.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import CLIPModel

from pannuke_ssl.utils import atomic_json_dump, seed_everything, write_csv
from .data_validation import module_sha
from .image_text_datasets import load_image_text_manifest
from .image_text_retrieval import _freeze, _ssl_features, _text_features, _train_trial, _tune
from .retrieval_protocol import (
    RETRIEVAL_VERSION, REFERENCE_CAPTIONS, caption_groups, unique_caption_indices,
    validate_plip_assets, file_sha256, caption_aware_metrics, caption_aware_chance,
)
from .trainer import load_checkpoint

_DATA_CONTENT_CACHE = {}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def effective_protocol(base):
    protocol = dict(base)
    protocol.update({
        "version": RETRIEVAL_VERSION,
        "loss": "symmetric_unique_caption_multi_positive_cross_entropy",
        "caption_identity": "exact_stored_caption",
        "text_candidates": "unique_captions",
        "i2t_query_weighting": "uniform_images",
        "t2i_query_weighting": "uniform_unique_captions",
        "relevance": "any_image_with_exact_matching_caption",
        "ties": "expected_hit_under_uniform_exact_score_tie_order",
        "selection_metric": "val_overall_mean_recall",
        "checkpoint_policy": "report_all_requested_checkpoints_no_primary_designation",
    })
    return protocol


def source_identity():
    root = Path(__file__).resolve().parents[1]
    names = ["ssl_framework/retrieval_protocol.py", "ssl_framework/retrieval_repair.py",
             "ssl_framework/image_text_retrieval.py", "ssl_framework/trainer.py", "models.py"]
    return {name: file_sha256(root / name) for name in names}


def rows_identity(rows):
    return digest([{k: row.get(k) for k in ("relative_path", "text", "group_id", "split")} for row in rows])


def data_content_identity(manifest):
    rows = [r for split in ("train", "val", "test") for r in manifest["split_rows"][split]]
    root = manifest["root"]
    state = [(r["relative_path"], (root / r["relative_path"]).stat().st_size,
              (root / r["relative_path"]).stat().st_mtime_ns) for r in rows]
    key = (str(root), manifest["manifest_sha256"], digest(state))
    if key not in _DATA_CONTENT_CACHE:
        content = [(r["relative_path"], file_sha256(root / r["relative_path"])) for r in rows]
        _DATA_CONTENT_CACHE[key] = digest(content)
    return _DATA_CONTENT_CACHE[key]


def context_identity(c, checkpoint, manifest, assets):
    protocol = effective_protocol(manifest["alignment_protocol"])
    portable = dict(protocol)
    portable["plip_model_dir"] = "validated_plip_asset_bundle"
    identity = {
        "evaluation_protocol_version": RETRIEVAL_VERSION,
        "protocol_sha256": digest(portable), "protocol": portable,
        "dataset": manifest["dataset"], "manifest_sha256": manifest["manifest_sha256"],
        "split_record_sha256": {s: rows_identity(rows) for s, rows in manifest["split_rows"].items()},
        "input_image_files_sha256": data_content_identity(manifest),
        "checkpoint_sha256": file_sha256(Path(checkpoint)),
        "method": c["method"]["name"], "seed": int(c["seed"]),
        "assets": assets, "source": source_identity(),
        "image_features": "256px_bicubic_center_crop_bf16_post_ln_patch_mean",
        "tf32": False,
    }
    identity["input_signature"] = digest(identity)
    return identity, protocol


def _context(c, checkpoint, dataset):
    from pannuke_ssl.ssl_methods.registry import build_method
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen-feature extraction")
    torch.set_num_threads(4)
    seed_everything(int(c["seed"]))
    # Preserve the original standalone downstream FP32-matmul default;
    # the image encoder continues to use BF16 autocast as before.
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    manifest = load_image_text_manifest(dataset, Path(c["manifests"]["root"]))
    model_dir = Path(manifest["alignment_protocol"]["plip_model_dir"])
    tokenizer, assets = validate_plip_assets(model_dir)
    identity, protocol = context_identity(c, checkpoint, manifest, assets)
    method = build_method(c, device)
    ck = load_checkpoint(method, Path(checkpoint), device)
    encoder = _freeze(method.encoder.to(device))
    plip = CLIPModel.from_pretrained(model_dir, local_files_only=True).to(device)
    _freeze(plip)
    text_branch, text_projection = plip.text_model, plip.text_projection
    scale = float(plip.logit_scale.detach().exp().cpu())
    del method, plip, ck
    torch.cuda.empty_cache()
    refs = _text_features(text_branch, text_projection, tokenizer, REFERENCE_CAPTIONS, device, 2)
    if not torch.isfinite(refs).all() or torch.allclose(refs[0], refs[1], atol=1e-6, rtol=1e-6):
        raise RuntimeError("PLIP reference captions have collapsed text features")
    return dict(c=c, manifest=manifest, identity=identity, protocol=protocol, device=device,
                encoder=encoder, text_branch=text_branch, text_projection=text_projection,
                tokenizer=tokenizer, logit_scale=scale)


def _cached(path, signature, compute):
    if path.is_file():
        cached = torch.load(path, map_location="cpu", weights_only=True)
        if cached["signature"] != signature:
            raise RuntimeError(f"Feature-cache signature mismatch: {path}")
        return cached["features"]
    features = compute().detach().float().cpu()
    if not torch.isfinite(features).all():
        raise RuntimeError("Nonfinite frozen features")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"signature": signature, "features": features}, temporary)
    temporary.replace(path)
    return features


def _features(ctx, split, cache_root):
    rows = ctx["manifest"]["split_rows"][split]
    groups = caption_groups([r["text"] for r in rows])
    identity = ctx["identity"]
    image_key = digest({"checkpoint": identity["checkpoint_sha256"],
                        "records": rows_identity(rows), "readout": identity["image_features"],
                        "source": identity["source"], "tf32": identity["tf32"],
                        "image_files": identity["input_image_files_sha256"]})
    text_key = digest({"assets": identity["assets"], "records": rows_identity(rows),
                       "tokenization": "padding_true_truncation_77_unique_captions", "source": identity["source"]})
    cache_root = Path(cache_root)
    image = _cached(cache_root / "images" / f"{image_key}.pt", image_key,
                    lambda: _ssl_features(ctx["encoder"], rows, ctx["manifest"]["root"], ctx["c"], ctx["device"]))
    first, inverse = unique_caption_indices(groups)
    texts = [rows[i]["text"] for i in first.tolist()]
    text_unique = _cached(cache_root / "text" / f"{text_key}.pt", text_key,
                         lambda: _text_features(ctx["text_branch"], ctx["text_projection"], ctx["tokenizer"],
                                                texts, ctx["device"], int(ctx["protocol"]["text_batch_size"])))
    text = text_unique[inverse]
    return image, text, groups


def _assert_identity(selection, identity):
    if selection.get("evaluation_protocol_version") != RETRIEVAL_VERSION:
        raise RuntimeError("Results belong to a superseded retrieval protocol")
    if selection.get("input_signature") != identity["input_signature"]:
        raise RuntimeError("Retrieval input/source/asset signature changed; use a fresh output directory")


def run_train_val(c, checkpoint, out, *, dataset, cache_root):
    out = Path(out)
    if (out / "test_started.json").exists():
        raise RuntimeError("TEST already started; TRAIN/VAL cannot be rerun in this output directory")
    ctx = _context(c, checkpoint, dataset)
    if (out / "projector_selection.json").is_file():
        selected = json.loads((out / "projector_selection.json").read_text())
        _assert_identity(selected, ctx["identity"])
        if (out / "best_image_projector.pt").is_file():
            return selected
    out.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(ctx["identity"], out / "evaluation_identity.json")
    before = {name: module_sha(ctx[name]) for name in ("encoder", "text_branch", "text_projection")}
    train_i, train_t, train_groups = _features(ctx, "train", cache_root)
    val_i, val_t, val_groups = _features(ctx, "val", cache_root)
    protocol = ctx["protocol"]
    kwargs = dict(seed=int(c["seed"]), device=ctx["device"], logit_scale=ctx["logit_scale"],
                  train_caption_ids=train_groups, val_caption_ids=val_groups)
    lr, wd, board = _tune(train_i, train_t, val_i, val_t, protocol, **kwargs)
    write_csv(board, out / "projector_tuning.csv")
    final = _train_trial(train_i, train_t, val_i, val_t, lr=lr, wd=wd,
                         epochs=int(protocol["final_maximum_epochs"]), batch_size=int(protocol["batch_size"]),
                         patience=int(protocol["final_early_stopping_patience"]), **kwargs)
    write_csv(final["history"], out / "projector_training.csv")
    after = {name: module_sha(ctx[name]) for name in before}
    if before != after:
        raise AssertionError("A frozen image/text component changed during head training")
    identity = ctx["identity"]
    epoch = int(torch.load(checkpoint, map_location="cpu", weights_only=False)["epoch"])
    selection = {
        **identity, "dataset": dataset, "encoder_epoch": epoch,
        "learning_rate": lr, "weight_decay": wd, "best_epoch": final["best_epoch"],
        "epochs_run": len(final["history"]), "validation": final["val_metrics"],
        "selection_metric": "val_overall_mean_recall", "test_used": False,
        "ssl_encoder_frozen": True, "plip_text_branch_frozen": True,
        "trainable_component": "image_projection_head_only", "projector": "linear_bias_false",
        "frozen_weight_hashes_before": before, "frozen_weight_hashes_after": after,
        "fixed_plip_logit_scale": ctx["logit_scale"],
        "split_counts": {"train_images": len(train_i), "train_unique_captions": len(torch.unique(train_groups)),
                         "val_images": len(val_i), "val_unique_captions": len(torch.unique(val_groups))},
        "prior_protocol_results_superseded": True,
    }
    temporary = out / "best_image_projector.tmp"
    torch.save({"projector": final["state"], "selection": selection,
                "input_dim": train_i.shape[1], "output_dim": train_t.shape[1]}, temporary)
    temporary.replace(out / "best_image_projector.pt")
    selection["head_sha256"] = file_sha256(out / "best_image_projector.pt")
    atomic_json_dump(selection, out / "projector_selection.json")
    return selection


def run_test(c, checkpoint, out, *, dataset, cache_root):
    out = Path(out)
    selection = json.loads((out / "projector_selection.json").read_text())
    ctx = _context(c, checkpoint, dataset)
    _assert_identity(selection, ctx["identity"])
    head_path = out / "best_image_projector.pt"
    if file_sha256(head_path) != selection["head_sha256"]:
        raise RuntimeError("Selected head checksum changed")
    completed = out / "test_retrieval_metrics.json"
    if completed.is_file():
        result = json.loads(completed.read_text())
        _assert_identity(result, ctx["identity"])
        return result
    marker = out / "test_started.json"
    if marker.exists():
        raise RuntimeError("Incomplete previous TEST attempt; preserve artifacts and diagnose before resuming")
    with marker.open("x") as stream:
        json.dump({"evaluation_protocol_version": RETRIEVAL_VERSION, "input_signature": selection["input_signature"],
                   "projector_selected": True, "test_used_for_selection": False, "started_unix": time.time()}, stream)
    test_i, test_t, groups = _features(ctx, "test", cache_root)
    saved = torch.load(head_path, map_location="cpu", weights_only=False)
    head = nn.Linear(saved["input_dim"], saved["output_dim"], bias=False).to(ctx["device"])
    head.load_state_dict(saved["projector"])
    head.eval()
    with torch.inference_mode():
        image_emb = head(test_i.to(ctx["device"])).cpu()
    metrics = caption_aware_metrics(image_emb, test_t, groups)
    first, _ = unique_caption_indices(groups)
    captions = [ctx["manifest"]["split_rows"]["test"][i]["text"] for i in first.tolist()]
    token_rows = ctx["tokenizer"](captions, padding=False, truncation=True, max_length=77)["input_ids"]
    token_unique = len({tuple(x) for x in token_rows})
    result = {
        **ctx["identity"], "dataset": dataset, "encoder_epoch": selection["encoder_epoch"],
        "pairs": len(test_i), "image_candidates": len(test_i), "unique_caption_candidates": len(first),
        "distinct_token_sequences": token_unique, "truncation_or_tokenization_collisions": len(first) - token_unique,
        "test": metrics, "chance": caption_aware_chance(groups), "selection": selection,
        "test_evaluated_once_within_protocol": True, "prior_protocol_results_superseded": True,
    }
    torch.save({"image_embeddings": image_emb, "text_embeddings": test_t,
                "caption_ids": groups, "input_signature": selection["input_signature"]}, out / "test_embeddings.pt")
    atomic_json_dump(result, completed)
    return result


def run_smoke(c, checkpoint, out, *, dataset):
    ctx = _context(c, checkpoint, dataset)
    # Deliberately use only TRAIN/VAL; no TEST feature extraction or metrics.
    tensors = {}
    before = {name: module_sha(ctx[name]) for name in ("encoder", "text_branch", "text_projection")}
    for split in ("train", "val"):
        rows = ctx["manifest"]["split_rows"][split][:32]
        image = _ssl_features(ctx["encoder"], rows, ctx["manifest"]["root"], c, ctx["device"])
        texts = [r["text"] for r in rows]
        groups = caption_groups(texts)
        first, inverse = unique_caption_indices(groups)
        text = _text_features(ctx["text_branch"], ctx["text_projection"], ctx["tokenizer"],
                              [texts[i] for i in first.tolist()], ctx["device"], 32)[inverse]
        tensors[split] = (image, text, groups)
    ti, tt, tg = tensors["train"]
    vi, vt, vg = tensors["val"]
    trial = _train_trial(ti, tt, vi, vt, lr=.001, wd=0., epochs=2, batch_size=16,
                          seed=int(c["seed"]), device=ctx["device"], logit_scale=ctx["logit_scale"],
                          train_caption_ids=tg, val_caption_ids=vg)
    after = {name: module_sha(ctx[name]) for name in before}
    if before != after or list(trial["state"]) != ["weight"]:
        raise AssertionError("Smoke test violated frozen-component contract")
    tok = ctx["tokenizer"](REFERENCE_CAPTIONS, padding=True, return_tensors="pt")
    report = {"status": "PASS", "dataset": dataset, "test_touched": False,
              "identity": ctx["identity"], "frozen_before": before, "frozen_after": after,
              "reference_tokens": tok["input_ids"].tolist(), "train_loss": [r["train_loss"] for r in trial["history"]],
              "train_feature_digest": digest(ti.tolist()), "val_feature_digest": digest(vi.tolist()),
              "train_shape": list(ti.shape), "val_shape": list(vi.shape)}
    atomic_json_dump(report, Path(out))
    return report
