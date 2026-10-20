# Corrected BCSS five-class frozen semantic segmentation (CVPR evaluation protocol)

This is a **new and separate** task, not an overwrite of the earlier custom
binary BCSS, GlaS semantic foreground, or EBHI-SEG binary tasks. Former
binary BCSS scores **must not** appear as the principal segmentation result.

## Authoritative protocol sources

- BCSS authors' segmentation baseline:
  https://bcsegmentation.grand-challenge.org/Baseline/
- Official BCSS class codes and zero=outside-ROI:
  https://github.com/PathologyDataScience/BCSS/blob/master/meta/gtruth_codes.tsv
- MedARC eva-probe (0.5µm/px and 76/30/45 institute splits):
  https://github.com/MedARC-AI/eva-probe/blob/main/docs/datasets/bcss.md

## Native labels and published five-region grouping

Original BCSS class codes have 22 values 0..21. It is **incorrect** to
treat 0 as tissue background. Code 0 is unlabeled outside ROI, 7 is
exclude and 15 undetermined; all are ignored in CE and mIoU.

Map original codes to exactly the authors' five region categories:

| Model class | Group | Original GT codes |
|---:|---|---|
| 0 | tumor | 1, 19 (angioinvasion), 20 (DCIS) |
| 1 | stroma | 2 |
| 2 | inflammatory infiltration | 3 (lymphocytic), 10 (plasma), 11 (other immune) |
| 3 | necrosis | 4 |
| 4 | other | 5, 6, 8, 9, 12, 13, 14, 16, 17, 18, 21 |
| ignore=255 | outside/excluded/uncertain | 0, 7, 15 |

This preserves the authors' five-class **multiclass** benchmark rather than
inventing a binary target. The original 22 codes remain in the source data.

## Splits, sampling and labels

The institution-based split matches the BCSS authors' held-out test:
- TEST: OL, LL, E2, EW, GM, S3 (45 ROIs)
- VALIDATION: BH, C8, A8, A1, E9 (30 ROIs)
- TRAIN: the other institutions (76 ROIs).

There is no patient or institutional overlap between splits. The 151
source color-normalized 0.25µm/px ROI images/masks are retained intact
under `datasets/BCSS`.

Follow eva-probe's 0.5µm/px resizing, then extract an **explicit fixed
4x4 spatial grid per ROI**, 16 RGB/mask tiles of 256x256. Grid selection
is **mask-independent** (no selection by class frequency/held-out
annotation content). The 2,416 generated paired tiles are stored under
`datasets/segmentation_benchmark_v2/bcss5`; this is a **custom
predefined tile sampling** and should not be called the published patch
extraction protocol of the original BCSS paper or eva-probe.

The resulting 1,216 TRAIN / 480 VAL / 720 TEST tiles are all used.
No source RGB or mask is ever overwritten. The preparation script checks
the 5-class coverage in each split and refuses a missing class.

## Frozen downstream

- Models: original LeJEPA and regular-simplex LeJEPA K=8, epochs 100/200.
- Representation: frozen ViT-B/32 last-layer patch tokens, [B,64,768],
  before SSL projector, exactly the original supervised downstream
  image normalisation and deterministic preprocessing.
- Spatial linear head: 1x1 conv (768→5) at 8x8 token resolution;
  bilinear upsample logits to 256x256 for pixel CE and evaluation.
  No encoder updates.
- Validation-only **sequential greedy** LR candidates 1e-3,1e-2 (5
  epochs), then WD 0,1e-4 (5 epochs). Final head maximum 25 epochs,
  val mIoU early stopping patience 5. Cross-entropy uses inverse-square-root-frequency class weights computed **only from TRAIN mask pixels**, normalized to mean 1. Fixed seed 20260903.
- Primary metric: macro mean IoU of five published tissue classes
  from globally pooled held-out test confusion counts; secondary:
  per-class IoUs and macro Dice. Ignored pixels excluded.
- Verify class pixel counts, SHA256 identities (10 source shards,
  manifest and SSL checkpoint), identical test records & group
  split across all models, no repeated held-out selection.
- No independent pretraining runs, no claim of statistical significance.

## Execution

```bash
python scripts/prepare_bcss5_official.py \
 --data-root "$SSL_DATA_ROOT" \
 --out-dir "$SSL_DATA_ROOT/segmentation_benchmark_v2/bcss5" \
 --manifest "$SSL_RUN_ROOT/segmentation_bcss5_v1/bcss5.json"

python scripts/run_bcss5_official.py \
 --data-root "$SSL_DATA_ROOT" \
 --manifest "$SSL_RUN_ROOT/segmentation_bcss5_v1/bcss5.json" \
 --checkpoint /absolute/path/epoch_100.pt \
 --model-root "$SSL_MODEL_ROOT" \
 --out-dir "$SSL_RUN_ROOT/segmentation_bcss5_v1/results/lejepa/epoch_100" \
 --method lejepa --epoch 100 --gpu 0
```

Pass `--smoke` with a distinct output path for isolated 8-image smoke runs.

## Publication caveats

- BCSS scores are **author-ontology-equivalent five-class segmentation**,
  with the authors' institute-held-out split but our explicit 16-tile grid.
  They are *not directly comparable* to a result from a different patch
  sampling, resolution or segmentation head.
- The frozen ViT-B/32 output has only an 8x8 spatial grid, which limits
  boundary precision.
- GlaS is an **instance** segmentation challenge. Previous GlaS
  foreground mIoU is only a binary semantic proxy; do not report it as
  canonical instance-segmentation leaderboard performance.
- EBHI-SEG is genuinely a binary mask task. Its six image diagnostic
  categories do not constitute pixel segmentation classes.
