# I-JEPA 100/200-epoch benchmark results

Matched comparison: LeJEPA, Simplex-LeJEPA K=8, MAE and I-JEPA.
All scores are percentages. Classification uses test accuracy; image-text retrieval symmetric mean recall; patch retrieval mAP; trainable 1/5/10-shot accuracy (100 fixed support sets); semantic segmentation mIoU (GlaS TestA/TestB averaged equally); GlaS instance segmentation object Dice and F1 (TestA/TestB averaged equally).

Matched: fresh ViT-B/32 encoder (256px), PanNuke, batch 128, 100/200 checkpoint epochs, frozen downstream encoders and evaluation protocols. I-JEPA predictor depth remains 4. Different native SSL objectives and teacher/masking schemes mean compute/FLOPs are not matched.

## Unweighted means across datasets

| Task | Epoch | N | LeJEPA | Simplex K8 | MAE | I-JEPA | I-JEPA wins vs MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classification (accuracy) | 100 | 11 | 75.86 | 76.40 | 69.59 | 69.94 | 4/11 |
| Classification (accuracy) | 200 | 11 | 77.82 | 78.89 | 71.94 | 72.21 | 3/11 |
| Trainable few-shot (1 shot) | 100 | 11 | 48.10 | 47.99 | 41.89 | 43.79 | 8/11 |
| Trainable few-shot (1 shot) | 200 | 11 | 48.55 | 48.83 | 43.04 | 44.41 | 8/11 |
| Trainable few-shot (10 shot) | 100 | 11 | 61.59 | 62.02 | 48.88 | 55.40 | 8/11 |
| Trainable few-shot (10 shot) | 200 | 11 | 63.06 | 64.15 | 51.29 | 56.21 | 9/11 |
| Trainable few-shot (5 shot) | 100 | 11 | 57.58 | 57.75 | 46.86 | 52.43 | 8/11 |
| Trainable few-shot (5 shot) | 200 | 11 | 58.70 | 59.58 | 48.76 | 52.79 | 9/11 |
| GlaS instance (object Dice) | 100 | 1 | 44.53 | 42.20 | 38.98 | 37.29 | 0/1 |
| GlaS instance (object Dice) | 200 | 1 | 44.47 | 44.81 | 42.30 | 25.01 | 0/1 |
| GlaS instance (object F1) | 100 | 1 | 22.84 | 17.32 | 24.61 | 15.51 | 0/1 |
| GlaS instance (object F1) | 200 | 1 | 23.03 | 23.88 | 22.05 | 10.40 | 0/1 |
| Patch retrieval (mAP) | 100 | 11 | 49.92 | 50.09 | 45.80 | 47.14 | 7/11 |
| Patch retrieval (mAP) | 200 | 11 | 50.26 | 50.69 | 46.52 | 47.33 | 7/11 |
| Image-text retrieval (mean recall) | 100 | 2 | 6.87 | 7.50 | 5.93 | 5.32 | 0/2 |
| Image-text retrieval (mean recall) | 200 | 2 | 7.70 | 8.99 | 7.13 | 5.67 | 0/2 |
| Semantic segmentation (mIoU) | 100 | 3 | 52.30 | 52.40 | 42.42 | 40.24 | 1/3 |
| Semantic segmentation (mIoU) | 200 | 3 | 54.23 | 55.24 | 49.36 | 44.57 | 0/3 |

## Classification (accuracy)

| Dataset | LeJEPA 100 | Simplex K8 100 | MAE 100 | I-JEPA 100 | LeJEPA 200 | Simplex K8 200 | MAE 200 | I-JEPA 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BACH | 72.50 | 70.00 | 55.00 | 55.00 | 67.50 | 77.50 | 60.00 | 55.00 |
| BreaKHis | 61.93 | 66.76 | 50.85 | 57.10 | 74.43 | 73.01 | 59.94 | 55.97 |
| CRC-HE-7K | 96.08 | 97.39 | 95.10 | 92.16 | 98.37 | 98.04 | 96.08 | 96.08 |
| Endometrial | 53.77 | 51.89 | 44.34 | 50.94 | 53.77 | 54.72 | 45.28 | 50.00 |
| GaHISSDB | 87.00 | 88.50 | 84.00 | 84.50 | 91.50 | 92.00 | 85.00 | 85.00 |
| Kather | 91.53 | 91.13 | 90.73 | 87.50 | 93.95 | 94.56 | 90.93 | 90.73 |
| LC25000 | 96.00 | 96.40 | 93.20 | 92.80 | 97.60 | 97.20 | 93.40 | 91.80 |
| MHIST | 75.76 | 76.77 | 65.66 | 59.60 | 78.79 | 79.80 | 70.20 | 69.19 |
| Oral OSCC | 77.87 | 81.15 | 82.79 | 81.97 | 78.69 | 80.33 | 76.23 | 81.97 |
| RenalCell | 62.04 | 63.58 | 60.19 | 54.32 | 65.12 | 63.58 | 64.51 | 56.48 |
| SICAPv2 | 60.00 | 56.84 | 43.68 | 53.42 | 56.32 | 57.11 | 49.74 | 62.11 |

## Image-text retrieval (mean recall)

| Dataset | LeJEPA 100 | Simplex K8 100 | MAE 100 | I-JEPA 100 | LeJEPA 200 | Simplex K8 200 | MAE 200 | I-JEPA 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ARCH | 10.08 | 11.69 | 9.35 | 8.19 | 11.86 | 13.84 | 11.20 | 8.61 |
| PathCap | 3.65 | 3.31 | 2.52 | 2.44 | 3.53 | 4.13 | 3.07 | 2.74 |

## Patch retrieval (mAP)

| Dataset | LeJEPA 100 | Simplex K8 100 | MAE 100 | I-JEPA 100 | LeJEPA 200 | Simplex K8 200 | MAE 200 | I-JEPA 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BACH | 37.57 | 37.08 | 35.69 | 35.18 | 39.39 | 39.71 | 36.22 | 34.03 |
| BreaKHis | 21.74 | 22.04 | 20.66 | 23.79 | 23.16 | 23.33 | 21.00 | 24.74 |
| CRC-HE-7K | 68.48 | 68.07 | 56.46 | 56.53 | 66.81 | 68.67 | 58.89 | 60.35 |
| Endometrial | 27.82 | 27.82 | 26.96 | 26.68 | 28.17 | 28.57 | 27.02 | 26.69 |
| GaHISSDB | 65.18 | 64.90 | 57.10 | 61.08 | 63.40 | 63.41 | 57.22 | 60.74 |
| Kather | 58.97 | 59.24 | 54.83 | 55.07 | 58.87 | 59.63 | 56.48 | 55.34 |
| LC25000 | 68.58 | 69.95 | 63.95 | 69.50 | 70.99 | 72.12 | 67.24 | 67.86 |
| MHIST | 54.20 | 54.13 | 52.72 | 52.04 | 53.99 | 54.11 | 52.53 | 52.63 |
| Oral OSCC | 68.08 | 68.16 | 69.01 | 68.18 | 69.03 | 69.03 | 68.78 | 68.47 |
| RenalCell | 45.79 | 46.46 | 39.41 | 40.05 | 46.88 | 47.07 | 38.88 | 40.11 |
| SICAPv2 | 32.76 | 33.11 | 26.97 | 30.41 | 32.19 | 31.92 | 27.45 | 29.68 |

## Trainable few-shot

| Dataset | LeJEPA 100 | Simplex K8 100 | MAE 100 | I-JEPA 100 | LeJEPA 200 | Simplex K8 200 | MAE 200 | I-JEPA 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BACH (1-shot accuracy) | 36.60 | 35.60 | 34.12 | 34.48 | 39.55 | 39.48 | 34.95 | 34.25 |
| BACH (10-shot accuracy) | 53.45 | 53.92 | 44.77 | 42.52 | 59.42 | 61.10 | 47.65 | 43.12 |
| BACH (5-shot accuracy) | 46.75 | 46.27 | 40.27 | 39.55 | 51.85 | 52.50 | 41.90 | 39.38 |
| BreaKHis (1-shot accuracy) | 22.92 | 23.22 | 21.24 | 24.04 | 24.39 | 24.53 | 21.73 | 24.93 |
| BreaKHis (10-shot accuracy) | 36.42 | 37.36 | 27.19 | 36.12 | 38.80 | 39.35 | 28.05 | 36.75 |
| BreaKHis (5-shot accuracy) | 32.17 | 32.62 | 25.79 | 32.38 | 34.21 | 34.39 | 26.53 | 33.26 |
| CRC-HE-7K (1-shot accuracy) | 67.66 | 66.96 | 49.35 | 49.99 | 66.62 | 68.62 | 53.53 | 54.34 |
| CRC-HE-7K (10-shot accuracy) | 84.50 | 84.18 | 58.14 | 70.03 | 86.07 | 87.11 | 64.74 | 71.49 |
| CRC-HE-7K (5-shot accuracy) | 80.61 | 80.38 | 55.74 | 63.55 | 81.86 | 83.21 | 61.63 | 65.95 |
| Endometrial (1-shot accuracy) | 27.71 | 27.14 | 26.33 | 25.82 | 27.82 | 28.35 | 26.49 | 25.59 |
| Endometrial (10-shot accuracy) | 33.12 | 33.39 | 28.10 | 27.09 | 32.99 | 35.26 | 28.90 | 27.11 |
| Endometrial (5-shot accuracy) | 29.70 | 29.97 | 26.90 | 26.55 | 30.67 | 31.52 | 27.15 | 26.07 |
| GaHISSDB (1-shot accuracy) | 65.33 | 64.92 | 56.91 | 61.43 | 63.51 | 63.47 | 57.17 | 61.64 |
| GaHISSDB (10-shot accuracy) | 77.03 | 77.45 | 66.62 | 73.86 | 76.72 | 77.38 | 67.85 | 73.55 |
| GaHISSDB (5-shot accuracy) | 74.80 | 74.96 | 64.28 | 71.03 | 74.06 | 74.86 | 65.25 | 71.19 |
| Kather (1-shot accuracy) | 56.19 | 56.30 | 40.95 | 45.25 | 56.10 | 56.86 | 44.85 | 46.62 |
| Kather (10-shot accuracy) | 79.28 | 80.05 | 54.75 | 67.91 | 80.09 | 81.38 | 61.38 | 69.61 |
| Kather (5-shot accuracy) | 74.41 | 75.07 | 50.41 | 63.31 | 74.88 | 76.37 | 56.51 | 64.39 |
| LC25000 (1-shot accuracy) | 67.08 | 67.89 | 55.86 | 62.31 | 69.97 | 71.04 | 58.95 | 60.98 |
| LC25000 (10-shot accuracy) | 83.11 | 83.90 | 61.69 | 81.51 | 87.67 | 88.38 | 67.74 | 81.24 |
| LC25000 (5-shot accuracy) | 79.29 | 79.99 | 61.06 | 77.89 | 83.94 | 84.67 | 66.01 | 76.08 |
| MHIST (1-shot accuracy) | 52.96 | 52.75 | 51.16 | 51.53 | 52.67 | 52.40 | 51.27 | 51.59 |
| MHIST (10-shot accuracy) | 60.90 | 61.05 | 52.62 | 52.31 | 59.54 | 60.40 | 52.49 | 54.14 |
| MHIST (5-shot accuracy) | 57.72 | 57.75 | 51.92 | 51.86 | 56.81 | 57.38 | 51.61 | 53.15 |
| Oral OSCC (1-shot accuracy) | 54.20 | 54.14 | 56.33 | 55.03 | 54.33 | 53.14 | 55.72 | 56.18 |
| Oral OSCC (10-shot accuracy) | 61.86 | 61.50 | 58.97 | 60.73 | 64.08 | 65.54 | 59.27 | 62.14 |
| Oral OSCC (5-shot accuracy) | 57.99 | 57.31 | 56.26 | 57.87 | 57.62 | 59.82 | 56.05 | 57.75 |
| RenalCell (1-shot accuracy) | 45.43 | 45.73 | 41.92 | 40.71 | 46.14 | 46.74 | 42.00 | 41.82 |
| RenalCell (10-shot accuracy) | 61.48 | 61.60 | 56.41 | 57.05 | 64.09 | 64.47 | 57.10 | 58.46 |
| RenalCell (5-shot accuracy) | 57.00 | 57.31 | 53.30 | 53.97 | 58.26 | 59.14 | 53.45 | 55.14 |
| SICAPv2 (1-shot accuracy) | 33.02 | 33.22 | 26.64 | 31.12 | 32.96 | 32.46 | 26.73 | 30.58 |
| SICAPv2 (10-shot accuracy) | 46.36 | 47.86 | 28.46 | 40.24 | 44.21 | 45.33 | 28.99 | 40.73 |
| SICAPv2 (5-shot accuracy) | 42.90 | 43.57 | 29.53 | 38.79 | 41.52 | 41.58 | 30.28 | 38.38 |

## Semantic segmentation (mIoU)

| Dataset | LeJEPA 100 | Simplex K8 100 | MAE 100 | I-JEPA 100 | LeJEPA 200 | Simplex K8 200 | MAE 200 | I-JEPA 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BCSS (5 classes) | 42.82 | 43.20 | 36.33 | 33.33 | 45.11 | 46.84 | 38.25 | 36.61 |
| EBHI-SEG | 59.41 | 59.84 | 57.63 | 46.15 | 63.90 | 64.33 | 59.33 | 50.53 |
| GlaS (mean Test A/B) | 54.65 | 54.17 | 33.31 | 41.23 | 53.68 | 54.56 | 50.51 | 46.57 |

## GlaS instance (object Dice)

| Dataset | LeJEPA 100 | Simplex K8 100 | MAE 100 | I-JEPA 100 | LeJEPA 200 | Simplex K8 200 | MAE 200 | I-JEPA 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GlaS instance (mean A/B) | 44.53 | 42.20 | 38.98 | 37.29 | 44.47 | 44.81 | 42.30 | 25.01 |

## GlaS instance (object F1)

| Dataset | LeJEPA 100 | Simplex K8 100 | MAE 100 | I-JEPA 100 | LeJEPA 200 | Simplex K8 200 | MAE 200 | I-JEPA 200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GlaS instance (mean A/B) | 22.84 | 17.32 | 24.61 | 15.51 | 23.03 | 23.88 | 22.05 | 10.40 |

## Data and audit trail

- comparison_all_metrics.csv: 124 individual metric records, plus I-JEPA percentage-point differences.
- comparison_summary.csv: task- and epoch-level dataset means.
- comparison_audit.json: checksums and exact cross-method split, support selection and protocol checks.
- source/ijepa/: all 78 primary I-JEPA JSON results, selection details and dataset provenance.
- experiment_metadata/: tuning, 250-epoch final pretraining, configuration and checkpoint hashes; no weight files.
- Reference results: benchmark_reports/mae_cvpr_20261027/source/.

This is a single-seed experiment; do not claim multiple-seed evidence.
