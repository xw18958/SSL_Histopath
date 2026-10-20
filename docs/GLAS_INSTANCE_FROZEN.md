# GlaS *instance* segmentation: frozen spatial representation probe (2026-10-20)

## Why a new experiment was necessary

The previously published `pathology_segmentation_v1` GlaS table is **binary
semantic gland segmentation**, not instance segmentation. That result is
retained for historical context, but **must not be shown as an official
GlaS gland-instance challenge benchmark**.

This experiment preserves individual ground-truth gland IDs and makes
**instance-labelled predictions** at the **original native image resolution**.

## Dataset and split

Source: MICCAI 2015 Warwick-QU GlaS dataset (GlaS challenge; Sirinukunwattana
et al., *Medical Image Analysis*, 2017). The dataset already exists under
`$SSL_DATA_ROOT/GlaS/Warwick_QU_Dataset`.

- Original training set 85 images. Fixed train/validation partition of 70/15
  chosen only from these 85 images using existing immutable GlaS manifest.
  The partition is image-disjoint but **patient-level disjointness is not
  proven by the available metadata**.
- **Official Test A 60 images** and **official Test B 20 images** are both
  retained and reported **separately**. The source image & annotation
  have their original width and height (frequently 775×522).
- Masks are **integer gland-instance ID maps**. Label 0 denotes background
  and each distinct positive integer is a different gland. At no point are
  the integer GT instance IDs collapsed to the binary labels used in
  the older semantic benchmark.
- The actual source manifest `glas.json` is independently checksum-verified,
  as are the SSL checkpoint weights. Raw images/masks remain unchanged.
  Test image files are not read during training or hyperparameter selection.

## Model and output

- Four frozen pre-projector encoders: LeJEPA @ 100 and 200 epochs,
  Simplex LeJEPA K=8 @ 100 and 200 epochs.
- ViT-B/32: frozen last-layer spatial tokens shaped 8×8×768 from 256×256 RGB
  crops, no predictor/projector or backbone finetuning.
- Lightweight trainable 1×1 spatial convolution head with 2 output channels:
  gland foreground logits, and normalized **per-instance interior distance
  transform**. GT distance maps are computed per original gland ID.
- Deterministic **six mask-independent native 256×256 training crops per
  train image** (420 train crops). The complete 70 original training images
  are represented. Downstream training optimises BCE foreground plus
  0.5× SmoothL1 distance loss on foreground pixels.
- For validation and testing, the head applies overlapping native-size
  tiles (256×256, stride 192, final tile anchored to image border),
  averages predicted maps in overlaps and reconstructs **native-resolution**
  maps; no squashing original images into square.
- Predicted gland interior scores generate local-maximum marker seeds;
  watershed on learned distance predictions separates individual glands.
  Min area and seed thresholds are selected on **validation only**.
- Sequential greedy validation tuning: learning rates [0.001, 0.01],
  weight decay [0, 0.0001], 3-epoch quick fits per candidate,
  18-epoch maximum final fit, validation patience 4; fixed training seed.
  Single SSL checkpoint per method/epoch; no repeated pretraining.
- Postprocessing fixed *before test*: sequential val object-Dice optimization
  on seed thresholds 0.25 / 0.35 / 0.50 and min area 100 / 200 / 500.
  Those are **not exhaustive grid search**.

## Metrics and limitations

Report **three challenge-family metrics, separately for Test A and Test B**:

1. **Object-detection F1** (higher better): ground-truth gland detection
   based on >50% of one GT object being covered by its dominant intersecting
   predicted gland. Distinct GT objects are matched **one-to-one** so
   duplicate detections never inflate F1. Per-image F1 averaged over split.
2. **Symmetric area-weighted object Dice** (higher better): each predicted
   gland and GT gland is matched to the object with maximum nonbackground
   overlap, and the two area-weighted Dice sums are averaged. Per-image
   object Dice averaged over split.
3. **Symmetric area-weighted object Hausdorff distance** (lower better):
   symmetric pixel-set Hausdorff distances between corresponding objects,
   with non-overlap matched to nearest object by Hausdorff distance.
   Empty prediction receives image diagonal penalty. Per-image averages.

Mathematical metric family follows the GlaS challenge and public metric
translation (e.g. `twpkevin06222/Gland-Segmentation/metric.py`), but this
implementation makes explicit one-to-one detection matching and empty-mask
penalty conventions. **Do not claim bit-for-bit equivalence to the
unavailable original Matlab evaluator** or direct leaderboard comparability
unless the Matlab evaluator is subsequently run on exported instance maps.
The 8×8 token grid cannot precisely represent thin boundaries; qualitative
failure analysis and validation-selected watershed behavior are important.
Many specialized full-resolution instance segmentors will outperform
this lightweight probe, which specifically tests frozen SSL spatial features.

## Run/checkpoint result storage

```
SSL_runs/glas_instance_20261020/
  smoke/...                          # tiny smoke runs, never used for paper
  results/lejepa/epoch_100/
  results/lejepa/epoch_200/
  results/simplex_k8/epoch_100/
  results/simplex_k8/epoch_200/
```

Each results directory contains `instance_head.pt`, `selection.json`,
`test_started.json`, `result.json`. The test file includes complete
per-image detection F1, object Dice, object Hausdorff and numbers of glands,
with separate Test A/B means. Do not mix the legacy semantic GlaS numbers
with these instance metrics.

Code: `scripts/run_glas_instance.py`.
