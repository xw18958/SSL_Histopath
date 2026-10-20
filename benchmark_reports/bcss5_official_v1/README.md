# Corrected BCSS FIVE-CLASS frozen segmentation (source-author grouping)

**Status:** Four full GPU jobs completed, independently verified. This is the
valid BCSS multiclass experiment that supersedes (but never overwrites) the
older BCSS binary tumour-versus-other probe in `pathology_segmentation_v1`.

## Five-class test results, % (higher is better)

| Epoch | LeJEPA mIoU | Simplex K8 mIoU | Simplex Δ (percentage points) | LeJEPA mDice | Simplex K8 mDice |
|---|---:|---:|---:|---:|---:|
| 100 | 42.82 | **43.20** | +0.38 | 57.58 | **58.07** |
| 200 | 45.11 | **46.84** | +1.73 | 60.36 | **62.04** |

mIoU averages **five fixed BCSS tissue groups**, never 2 classes. The
confusion matrix is pooled over every valid pixel in the 720 held-out
tiles; zero (outside ROI), exclude and undetermined are ignored.

### Per-class IoU (%) and paired changes

| Class | LeJEPA 100 | Simplex 100 | LeJEPA 200 | Simplex 200 |
|---|---:|---:|---:|---:|
| Tumor | 61.10 | **61.88** | 61.12 | **64.41** |
| Stroma | **56.43** | 55.94 | 58.40 | **59.93** |
| Inflammatory infiltrate | 52.80 | **54.65** | 54.26 | **55.19** |
| Necrosis | 14.97 | **20.47** | 21.51 | **29.15** |
| Other | **28.83** | 23.08 | **30.25** | 25.52 |

All per-class figures in `per_class.csv` are independently recomputed
from the saved test confusion matrices. The global improvement does not
imply every tissue group improves: the heterogeneous OTHER category is
worse for Simplex at both epochs.

## Scientific protocol

- Original source: BCSS 151 ROIs. The original paper trains five tissue
  groups (tumor, stroma, immune infiltration, necrosis, other) using
  precise class code grouping; see `docs/BCSS5_OFFICIAL_SEGMENTATION.md`.
- Hospital/institute-disjoint split: **76 train / 30 val / 45 test** ROIs.
  Official testing institutes: OL, LL, E2, EW, GM, S3.
- Evaluation image level: **0.5µm/px**, fixed mask-independent
  **4×4 sample grid** of 256×256 tiles per ROI. This produces
  1,216 train / 480 val / 720 test patches.
- Frozen pre-projector ViT-B/32 spatial last patch token grid (8×8×768).
  Only a 768→5 1×1 linear convolution probe is trained, then bilinear
  upsampling. Training uses 5-class inverse-sqrt-frequency weights
  calculated from TRAIN class pixels.
- Sequential greedy five-epoch validation-only learning-rate and
  weight-decay selection; final head up to 25 epochs, validation patience 5.
  **Test only after head and epoch selection**.
- LeJEPA and Simplex K8 at their existing 100-epoch and 200-epoch
  checkpoints. No SSL model retraining, tuning or changes.
- Runtime: all 3 previously authorised servers (4 NVIDIA GPUs).
  All four full jobs and smoke tests completed.
- Test predictions are **not** claimed as official BCSS leaderboard
  scores: tile sampling, feature architecture and probing head differ
  from the original paper. No 22-way fine-grained scores are claimed.
- Earlier GlaS pixel mIoU is a binary semantic supplementary measurement,
  **not** the official gland-instance challenge score. EBHI-SEG truly has
  binary pixel-level masks, and need not be converted to image diagnosis
  categories.

## Reproducibility and source of truth

- `scripts/prepare_bcss5_official.py`: immutable, checksum-audited
  source-to-paired-tile conversion; patient/institute leakage checks.
- `scripts/run_bcss5_official.py`: frozen feature extraction and
  validation-tuned five-class linear segmentation head.
- `scripts/aggregate_bcss5_official.py`: independently recomputes
  every pixel confusion matrix metric, verifies all 4 checkpoints,
  sample identities/counts, class weights and identical label support.
- `summary.csv`, `per_class.csv`, `verification.json`:
  public reproducible summary and exact SHA-256 identities.
- Full `result.json`, selection traces, `linear_head.pt`, manifests,
  logs and generated patches remain on the authorised servers under
  `SSL_runs/segmentation_bcss5_v1` and
  `datasets/segmentation_benchmark_v2/bcss5`.
- The original `SSL_Histopath` training and previous benchmark
  pipelines are unchanged.
