# DINOv3 — matched histopathology SSL baseline

The DINOv3 experiment follows the **same approved MAE and Simplex-LeJEPA K=8
comparison pipeline**, rather than the historical `pipeline` action with
validation-based early stopping. See
[`docs/DINOV3_FAIR_BASELINE_AUDIT_20261028.md`](docs/DINOV3_FAIR_BASELINE_AUDIT_20261028.md)
and [the live campaign record](docs/DINOV3_CAMPAIGN_20261028.md).

## Fixed experimental conditions

| Setting | DINOv3 and established baselines |
|---|---|
| Backbone | Fresh shared CLIP/PLIP ViT-B/32, randomly initialized |
| Source images | 7,901 unlabeled PanNuke images; same source keys |
| Seed | 20260903 |
| Effective source-image batch size | 128 |
| Precision | BF16; TF32 enabled |
| Development tuning | Three **sequential** 20-epoch LR trials |
| Tuning objective | PanNuke development validation linear Macro-F1; no external test selection |
| Final pretraining | 250 epochs on all 7,901 images; **no early stopping** |
| Saved epochs | 100, 150, 200, 250 |
| Primary downstream checkpoints | 100 and 200 |
| Primary feature | Mean of final patch tokens (768-D); DINOv3 uses EMA teacher |

All external downstream protocols, splits, probe settings, metrics, fixed
few-shot support subsets and segmentation heads are inherited from the
**unchanged MAE scripts**. Previously excluded classification datasets remain
excluded. This is the DINOv3 **base-stage objective**, not a reproduction of
the separate later Gram-anchoring stage.

## DINOv3 method-specific settings

- DINO class-token self-distillation, iBOT masked patch distillation, KoLeo.
- Two 256px global crops and eight 96px local crops.
- Separate 65,536-prototype DINO and iBOT heads.
- AdamW, layerwise LR decay 0.9, patch embedding LR multiplier 0.2.
- EMA teacher, teacher temperature schedule, official loss and masking rules.
- Original released DINOv3 augmentation **as implemented in Meta's source**:
  9px Gaussian blur kernel and `RandomApply(p=1-nominal_p)`. This yields
  effective blur rates 0%, 90% and 50% for global1/global2/local crops.
  The unusual inversion is explicit and regression-tested, not silently
  replaced with the DINOv2 augmentation recipe.
- Projection-head last-layer freeze uses **the real loader length**:
  50 steps/epoch during development, 62 steps/epoch during full pretraining.

Method-specific augmentations, objectives and optimizers are kept intact.
Matching epochs/data/batch does not imply equal FLOPs; report computation
separately. The shared ViT-B/32 patch-32 backbone and the omission of Gram
anchoring must be disclosed in paper tables.

## Approved execution flow

```bash
# Set SSL_PROJECT_ROOT, SSL_DATA_ROOT, SSL_MODEL_ROOT, SSL_RUN_ROOT,
# PYTHONPATH, and CUDA_VISIBLE_DEVICES before running.
python scripts/run_ssl_standard.py tune --method dinov3 \
  --run-id ssl-dinov3-cvpr20261028
# Full-data pretraining is launched by the gated server script,
# only after verifying the selected learning rate:
bash scripts/approved_dinov3_pretrain_after_tuning.sh
# Downstream roles start only when the trained fixed-epoch
# checkpoint files and source SHA-256 checks have been verified:
bash scripts/approved_dinov3_downstreams_after_pretrain.sh
```

These `approved_*` launch scripts require the `DINOV3_*` environment variables
described in the source code and campaign record. Do not run the generic
`pipeline` action for this CVPR experiment: it has different early-stopping
behavior.

## Status (28 October 2026, Sydney)

Corrections merged as [PR #10](https://github.com/xw18958/SSL_Histopath/pull/10).
The three-trial tuning campaign has started on `gpu1-jinman-2`. Pretraining
and matching downstream launchers are installed and **waiting at dependency
gates** on `gpu1-358-0` and `gpu2-358-0`; do not interpret a queued job as a
completed experiment or claim DINOv3 benchmark scores before results exist.
