> **Update (2026-10-20):** GlaS gland-instance evaluation with full native-resolution objects is now implemented in [GLAS_INSTANCE_FROZEN.md](GLAS_INSTANCE_FROZEN.md), with results in `benchmark_reports/glas_instance_v1`. The binary GlaS probe in this document is supplementary and not an official instance-segmentation challenge result.

# Frozen pathology semantic segmentation (October 2026)

This is a **new, independent downstream**. It does not modify existing SSL training, classification, few-shot, retrieval, or checkpoint files.

## Models / evaluation

- Original **LeJEPA** and **Regular Simplex LeJEPA K=8**, saved epochs **100** and **200**.
- Freeze the **complete pre-projector encoder**; extract the actual last-layer [64,768] spatial patch tokens (8×8 ViT-B/32).
- Deterministic RGB bicubic short-side resize 256, center crop 256; masks same spatial geometry with **nearest-neighbor** resampling. Inputs RGB/255 use the existing encoder's internal CLIP normalization.
- Lightweight downstream: independent *trainable* single 1×1 convolution (768→2) on the token map, followed by bilinear logits upsampling to 256×256, cross-entropy loss (ignore label 255). **No fine-tuning** of the encoder or projector.
- Sequential greedy head HP tuning on validation **only**: learning rates 0.001/0.01 with weight decay 0, then weight decay 0/0.0001 at selected LR, each for exactly 5 epochs. A fresh final head uses 25-epoch maximum and patience 5 on validation mIoU. No HP selection on test.
- Single fixed seed 20260903 for training the downstream head. This is not independent SSL pretraining repetition.
- **Primary:** pixel-level (micro-aggregated) mean IoU across the two classes; **secondary:** foreground Dice and foreground IoU. Aggregation is by pooled pixel confusion matrix, not per-image averaged Dice.
- Every job records the SHA256 of its immutable full SSL checkpoint and the manifest. A pre-test exclusive marker prevents repeating a held-out test run silently. Smoke outputs are kept separate from full results.
- Full segmentation evaluates the **same samples/splits** across methods and epoch checkpoints. Every dataset is a *semantic binary segmentation task*, not a GlaS instance segmentation / object F1 comparison.

## Histopathology datasets only

All original sources under SSL_DATA_ROOT remain untouched.

1. **GlaS** — 165 colorectal H&E gland-segmentation pairs. Source: GlaS MICCAI'15 challenge / public Kaggle mirror; raw directory `$SSL_DATA_ROOT/GlaS`. Preserve official **85 train / 60 Test A / 20 Test B**; reserve 15 of original training images for validation (70 train). The source does not provide full patient IDs for all crops; patient independence of the internal 70/15 selection cannot be asserted. Our metric is **binary gland semantic mIoU**, NOT the official gland instance segmentation metric. Report Test A and Test B separately as supplements.
2. **EBHI-SEG** — 2,228 original images, but 2 Low-grade IN images are missing masks locally. Exactly **2,226 verifiable pairs**; all six image categories pooled in one two-class tissue-region foreground/background segmentation. Masks are grayscale antialiased (0..255), threshold **>=128**. Split by a filename-derived specimen proxy, *not* an assumed patient identifier. No unpaired image is silently treated as background. This is a novel evaluation split, **not an official segmentation leaderboard split**.
3. **BCSS** — 151 TCGA breast cancer ROIs from [Beijuka/BCSS](https://huggingface.co/datasets/Beijuka/BCSS), original files 4.91GB. Extract exactly six grid-positioned 256×256 patches per ROI before filtering largely unannotated tiles. Preserve patient-ID grouping, 70% train / 15% validation / 15% test **by patient**. Original label 1 (tumour) is positive; valid 2..21 excluding 7/15 is negative; 0/7/15 and unknown/ambiguous are ignored as label 255. This is **binary tumour semantic segmentation**, not the full 22-class BCSS task. We disclose the grid tile sampling and patient split, and do not claim state-of-the-art 22-class benchmarks.

The BCSS dataset's original 151 ROIs do not have an official train/validation/test split. Generated patch files live only in `$SSL_DATA_ROOT/segmentation_benchmark_v1/bcss_binary`; deterministic manifest JSONs are stored with experiment outputs and contain explicit source identities and patient or proxy groups. Generated data should **not be committed to Git**.

## Dataset preparation (run once on dataset master)

```bash
python scripts/prepare_pathology_segmentation.py --data-root "$SSL_DATA_ROOT" --out-dir "$SSL_RUN_ROOT/segmentation_20261020/manifests" --dataset glas
python scripts/prepare_pathology_segmentation.py --data-root "$SSL_DATA_ROOT" --out-dir "$SSL_RUN_ROOT/segmentation_20261020/manifests" --dataset ebhi
python scripts/prepare_pathology_segmentation.py --data-root "$SSL_DATA_ROOT" --out-dir "$SSL_RUN_ROOT/segmentation_20261020/manifests" --dataset bcss --tiles-per-roi 6
```

The preparation script **refuses to overwrite** an existing manifest, and validates unique sample names, split coverage, source files and source-group disjointness.

## Run a single checkpoint

```bash
python scripts/run_pathology_segmentation.py \
  --data-root "$SSL_DATA_ROOT" \
  --manifest "$SSL_RUN_ROOT/segmentation_20261020/manifests/glas.json" \
  --checkpoint "/path/to/epoch_100.pt" \
  --model-root "$SSL_MODEL_ROOT" \
  --out-dir "$SSL_RUN_ROOT/segmentation_20261020/results/glas/lejepa/epoch_100" \
  --method lejepa --epoch 100 --gpu 0
```

Pass `--smoke` with an entirely separate `--out-dir` for 8-sample per-split smoke runs. Results are in `selection.json`, `linear_head.pt`, `test_started.json`, and `result.json`. Results do **not** overwrite any earlier benchmark results. Run checkpoint choices must be resolved to the existing baseline and K8 saved model directories and verified before launching.

## Scientific caveats

- ViT-B/32 supplies only an 8×8 map per 256×256 image. This probes **coarse spatial semantics**, not fine nuclei boundaries.
- Downsampling and 8×8 patch resolution limit fine-boundary accuracy. No architectural interpolation or mask preprocessing change may depend on test scores.
- Different binary targets across datasets have different levels of intrinsic difficulty. Report per-dataset results; an unweighted aggregate across datasets should be explicitly qualified.
- GlaS and EBHI internal train/val split grouping cannot be certified patient-disjoint. BCSS uses source patient IDs.
- No model-comparison claim of statistical significance from one downstream head fit per setting. The saved SSL checkpoints are single-training-run comparisons.
