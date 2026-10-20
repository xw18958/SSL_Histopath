# Frozen linear pathology semantic segmentation — verified results

Full 12-job evaluation: **completed and independently verified**. Public results are compact aggregate summaries only; experimental heads and per-run test artifacts remain in the three authorized server run directories.

- Source code: `scripts/prepare_pathology_segmentation.py`, `scripts/run_pathology_segmentation.py`, `scripts/aggregate_pathology_segmentation.py`.
- Implementation/protocol: `docs/PATHOLOGY_SEGMENTATION.md`.
- Full matched dataset/checkpoint results: `per_dataset.csv`.
- Paired Simplex-minus-LeJEPA results: `paired_comparisons.csv`.
- SHA-256 provenance and independently checked confusion metrics: `verification.json`.

## Primary metric: semantic binary mIoU (%)

| Dataset | LeJEPA 100 | Simplex K8 100 | Delta (pp) | LeJEPA 200 | Simplex K8 200 | Delta (pp) |
|---|---:|---:|---:|---:|---:|---:|
| GlaS | 54.47 | 54.89 | +0.42 | 55.01 | 56.14 | +1.13 |
| EBHI-SEG | 59.41 | 59.84 | +0.43 | 63.90 | 64.33 | +0.43 |
| BCSS (sampled tumor vs non-tumor) | 67.33 | 67.51 | +0.17 | 67.50 | 69.66 | +2.16 |
| Mean, equal dataset weight | 60.40 | 60.75 | +0.34 | 62.14 | 63.38 | +1.24 |

The mean is an **equal-dataset-weight descriptive mean**, not a common segmentation population risk estimate.

### Secondary metric: foreground Dice (%)

| Dataset | LeJEPA 100 | Simplex 100 | LeJEPA 200 | Simplex 200 |
|---|---:|---:|---:|---:|
| GlaS | 70.64 | 72.98 | 74.26 | 72.14 |
| EBHI-SEG | 82.56 | 81.68 | 84.17 | 84.40 |
| BCSS | 74.32 | 74.38 | 73.25 | 77.68 |

**Important:** Higher mIoU does not mean every secondary metric improves. These comparisons should not be described as uniformly superior across all segmentation metrics.

## Protocol coverage

| Dataset | Source/pair count | Train | Validation | Test | Target |
|---|---:|---:|---:|---:|---|
| GlaS | 165 | 70 | 15 | 80 (official Test A=60, Test B=20) | Binary colorectal gland mask |
| EBHI-SEG | 2,226 matched pairs | 1,461 | 365 | 400 | Binary annotated-region mask |
| BCSS | 151 original large ROIs; **825 selected patches** | 580 | 127 | 118 | Binary tumor-versus-valid-nontumor tissue |

Frozen pre-projector LeJEPA and Simplex K=8 encoders at 100/200 epochs; 8×8 spatial features with 768 channels; one trained 1×1 linear segmentation probe with bilinear upsampling; sequential validation hyperparameter selection; held-out TEST evaluated only after final head selection. All 12 tests are complete; only the 2,226 matched EBHI-SEG image/mask pairs are used.

## Limitations for publication

- This is **semantic binary segmentation**, and **not** official GlaS gland-instance segmentation with object-level F1 or Hausdorff metrics. Do not compare these numbers against the original instance segmentation challenge leaderboard.
- EBHI-SEG source filenames only support a specimen-like grouping proxy; genuine patient-level disjointness cannot be guaranteed. The source has two unmatched images excluded.
- GlaS official test images are preserved, but the 70/15 train/validation split from its official training images is image-level because source patient IDs are not available.
- BCSS uses a **custom but fixed patient-grouped split** and **fixed sparse grid sampling** of six 256×256 crops per large ROI, with predominantly unlabeled tiles removed. Its binary scores cannot be compared to reported 22-class BCSS benchmarks. The 151 ROIs have no official split.
- All 4 pretraining checkpoints originate from the user's existing experiments; no SSL retraining or encoder tuning occurred. The head-only linear probes are task-trained.
- The main backbone is ViT-B/32, supplying a very coarse 8×8 grid. Thin structure and boundary delineation are inherently resolution-limited.
- The results are descriptive matched-checkpoint comparisons, not an estimate of general population or clinical performance.

## Server provenance

- `gpu1-jinman-2`: LeJEPA epoch 100, all three datasets.
- `gpu1-358-0`: Simplex K8 epoch 100, all three datasets.
- `gpu2-358-0` GPU0: LeJEPA epoch 200, all three datasets; GPU1: Simplex K8 epoch 200, all three datasets.

Per-run `linear_head.pt`, `selection.json`, `test_started.json`, `result.json`, worker logs and fixed manifests remain in each server's `SSL_runs/segmentation_20261020` directory. The dataset preparation folders remain under the established `datasets` root. This addition does not edit the original SSL training or previous evaluation pipeline.
