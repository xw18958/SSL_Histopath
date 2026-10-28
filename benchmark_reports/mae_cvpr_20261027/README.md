# MAE vs LeJEPA vs Simplex-LeJEPA (K=8)

Comparison across matched 100- and 200-epoch checkpoints. All scores are percentages (%), higher is better.
All three methods use the same fresh ViT-B/32 encoder and PanNuke pretraining images.
Exact downstream split, preprocessing, protocol and support subset identities were checked before aggregation.

## Summary — unweighted mean across datasets

| Task | Epoch | Datasets | LeJEPA | Simplex K=8 | MAE | K8 > MAE | K8 > LeJEPA |
|---|---:|---:|---:|---:|---:|---:|---:|
| classification | 100 | 11 | 75.86 | 76.40 | 69.59 | 10/11 | 7/11 |
| classification | 200 | 11 | 77.82 | 78.89 | 71.94 | 10/11 | 7/11 |
| retrieval | 100 | 2 | 6.87 | 7.50 | 5.93 | 2/2 | 1/2 |
| retrieval | 200 | 2 | 7.70 | 8.99 | 7.13 | 2/2 | 2/2 |
| patch | 100 | 11 | 49.92 | 50.09 | 45.80 | 10/11 | 7/11 |
| patch | 200 | 11 | 50.26 | 50.69 | 46.52 | 11/11 | 9/11 |
| fewshot 1-shot accuracy | 100 | 11 | 48.10 | 47.99 | 41.89 | 10/11 | 5/11 |
| fewshot 1-shot accuracy | 200 | 11 | 48.55 | 48.83 | 43.04 | 10/11 | 6/11 |
| fewshot 5-shot accuracy | 100 | 11 | 57.58 | 57.75 | 46.86 | 11/11 | 8/11 |
| fewshot 5-shot accuracy | 200 | 11 | 58.70 | 59.58 | 48.76 | 11/11 | 11/11 |
| fewshot 10-shot accuracy | 100 | 11 | 61.59 | 62.02 | 48.88 | 11/11 | 9/11 |
| fewshot 10-shot accuracy | 200 | 11 | 63.06 | 64.15 | 51.29 | 11/11 | 11/11 |
| semantic | 100 | 3 | 52.30 | 52.40 | 42.42 | 3/3 | 2/3 |
| semantic | 200 | 3 | 54.23 | 55.24 | 49.36 | 3/3 | 3/3 |
| instance | 100 | 1 | 44.53 | 42.20 | 38.98 | 1/1 | 0/1 |
| instance | 200 | 1 | 44.47 | 44.81 | 42.30 | 1/1 | 1/1 |

## Classification

Frozen linear-probe test accuracy.

| Dataset | LeJEPA 100 | Simplex K=8 100 | MAE 100 | LeJEPA 200 | Simplex K=8 200 | MAE 200 |
|---|---:|---:|---:|---:|---:|---:|
| MHIST | 75.76 | 76.77 | 65.66 | 78.79 | 79.80 | 70.20 |
| CRC-HE-7K | 96.08 | 97.39 | 95.10 | 98.37 | 98.04 | 96.08 |
| BreaKHis | 61.93 | 66.76 | 50.85 | 74.43 | 73.01 | 59.94 |
| Kather | 91.53 | 91.13 | 90.73 | 93.95 | 94.56 | 90.93 |
| BACH | 72.50 | 70.00 | 55.00 | 67.50 | 77.50 | 60.00 |
| SICAPv2 | 60.00 | 56.84 | 43.68 | 56.32 | 57.11 | 49.74 |
| Oral OSCC | 77.87 | 81.15 | 82.79 | 78.69 | 80.33 | 76.23 |
| Endometrial | 53.77 | 51.89 | 44.34 | 53.77 | 54.72 | 45.28 |
| GaHISSDB | 87.00 | 88.50 | 84.00 | 91.50 | 92.00 | 85.00 |
| RenalCell | 62.04 | 63.58 | 60.19 | 65.12 | 63.58 | 64.51 |
| LC25000 | 96.00 | 96.40 | 93.20 | 97.60 | 97.20 | 93.40 |

## Image-text retrieval

Symmetric mean recall of image-text and text-image retrieval.

| Dataset | LeJEPA 100 | Simplex K=8 100 | MAE 100 | LeJEPA 200 | Simplex K=8 200 | MAE 200 |
|---|---:|---:|---:|---:|---:|---:|
| ARCH | 10.08 | 11.69 | 9.35 | 11.86 | 13.84 | 11.20 |
| PathCap | 3.65 | 3.31 | 2.52 | 3.53 | 4.13 | 3.07 |

## Patch image-to-image retrieval

Train gallery, test query, mAP on frozen embeddings.

| Dataset | LeJEPA 100 | Simplex K=8 100 | MAE 100 | LeJEPA 200 | Simplex K=8 200 | MAE 200 |
|---|---:|---:|---:|---:|---:|---:|
| MHIST | 54.20 | 54.13 | 52.72 | 53.99 | 54.11 | 52.53 |
| CRC-HE-7K | 68.48 | 68.07 | 56.46 | 66.81 | 68.67 | 58.89 |
| BreaKHis | 21.74 | 22.04 | 20.66 | 23.16 | 23.33 | 21.00 |
| Kather | 58.97 | 59.24 | 54.83 | 58.87 | 59.63 | 56.48 |
| BACH | 37.57 | 37.08 | 35.69 | 39.39 | 39.71 | 36.22 |
| SICAPv2 | 32.76 | 33.11 | 26.97 | 32.19 | 31.92 | 27.45 |
| Oral OSCC | 68.08 | 68.16 | 69.01 | 69.03 | 69.03 | 68.78 |
| Endometrial | 27.82 | 27.82 | 26.96 | 28.17 | 28.57 | 27.02 |
| GaHISSDB | 65.18 | 64.90 | 57.10 | 63.40 | 63.41 | 57.22 |
| RenalCell | 45.79 | 46.46 | 39.41 | 46.88 | 47.07 | 38.88 |
| LC25000 | 68.58 | 69.95 | 63.95 | 70.99 | 72.12 | 67.24 |

## Trainable few-shot, 1 shot

100 matched support sets, logistic regression, accuracy.

| Dataset | LeJEPA 100 | Simplex K=8 100 | MAE 100 | LeJEPA 200 | Simplex K=8 200 | MAE 200 |
|---|---:|---:|---:|---:|---:|---:|
| MHIST | 52.96 | 52.75 | 51.16 | 52.67 | 52.40 | 51.27 |
| CRC-HE-7K | 67.66 | 66.96 | 49.35 | 66.62 | 68.62 | 53.53 |
| BreaKHis | 22.92 | 23.22 | 21.24 | 24.39 | 24.53 | 21.73 |
| Kather | 56.19 | 56.30 | 40.95 | 56.10 | 56.86 | 44.85 |
| BACH | 36.60 | 35.60 | 34.12 | 39.55 | 39.48 | 34.95 |
| SICAPv2 | 33.02 | 33.22 | 26.64 | 32.96 | 32.46 | 26.73 |
| Oral OSCC | 54.20 | 54.14 | 56.33 | 54.33 | 53.14 | 55.72 |
| Endometrial | 27.71 | 27.14 | 26.33 | 27.82 | 28.35 | 26.49 |
| GaHISSDB | 65.33 | 64.92 | 56.91 | 63.51 | 63.47 | 57.17 |
| RenalCell | 45.43 | 45.73 | 41.92 | 46.14 | 46.74 | 42.00 |
| LC25000 | 67.08 | 67.89 | 55.86 | 69.97 | 71.04 | 58.95 |

## Trainable few-shot, 5 shots

100 matched support sets, logistic regression, accuracy.

| Dataset | LeJEPA 100 | Simplex K=8 100 | MAE 100 | LeJEPA 200 | Simplex K=8 200 | MAE 200 |
|---|---:|---:|---:|---:|---:|---:|
| MHIST | 57.72 | 57.75 | 51.92 | 56.81 | 57.38 | 51.61 |
| CRC-HE-7K | 80.61 | 80.38 | 55.74 | 81.86 | 83.21 | 61.63 |
| BreaKHis | 32.17 | 32.62 | 25.79 | 34.21 | 34.39 | 26.53 |
| Kather | 74.41 | 75.07 | 50.41 | 74.88 | 76.37 | 56.51 |
| BACH | 46.75 | 46.27 | 40.27 | 51.85 | 52.50 | 41.90 |
| SICAPv2 | 42.90 | 43.57 | 29.53 | 41.52 | 41.58 | 30.28 |
| Oral OSCC | 57.99 | 57.31 | 56.26 | 57.62 | 59.82 | 56.05 |
| Endometrial | 29.70 | 29.97 | 26.90 | 30.67 | 31.52 | 27.15 |
| GaHISSDB | 74.80 | 74.96 | 64.28 | 74.06 | 74.86 | 65.25 |
| RenalCell | 57.00 | 57.31 | 53.30 | 58.26 | 59.14 | 53.45 |
| LC25000 | 79.29 | 79.99 | 61.06 | 83.94 | 84.67 | 66.01 |

## Trainable few-shot, 10 shots

100 matched support sets, logistic regression, accuracy.

| Dataset | LeJEPA 100 | Simplex K=8 100 | MAE 100 | LeJEPA 200 | Simplex K=8 200 | MAE 200 |
|---|---:|---:|---:|---:|---:|---:|
| MHIST | 60.90 | 61.05 | 52.62 | 59.54 | 60.40 | 52.49 |
| CRC-HE-7K | 84.50 | 84.18 | 58.14 | 86.07 | 87.11 | 64.74 |
| BreaKHis | 36.42 | 37.36 | 27.19 | 38.80 | 39.35 | 28.05 |
| Kather | 79.28 | 80.05 | 54.75 | 80.09 | 81.38 | 61.38 |
| BACH | 53.45 | 53.92 | 44.77 | 59.42 | 61.10 | 47.65 |
| SICAPv2 | 46.36 | 47.86 | 28.46 | 44.21 | 45.33 | 28.99 |
| Oral OSCC | 61.86 | 61.50 | 58.97 | 64.08 | 65.54 | 59.27 |
| Endometrial | 33.12 | 33.39 | 28.10 | 32.99 | 35.26 | 28.90 |
| GaHISSDB | 77.03 | 77.45 | 66.62 | 76.72 | 77.38 | 67.85 |
| RenalCell | 61.48 | 61.60 | 56.41 | 64.09 | 64.47 | 57.10 |
| LC25000 | 83.11 | 83.90 | 61.69 | 87.67 | 88.38 | 67.74 |

## Semantic segmentation

GlaS reports equal mean of official Test A/B. BCSS uses five official classes.

| Dataset | LeJEPA 100 | Simplex K=8 100 | MAE 100 | LeJEPA 200 | Simplex K=8 200 | MAE 200 |
|---|---:|---:|---:|---:|---:|---:|
| GlaS (mean Test A/B) | 54.65 | 54.17 | 33.31 | 53.68 | 54.56 | 50.51 |
| EBHI-SEG | 59.41 | 59.84 | 57.63 | 63.90 | 64.33 | 59.33 |
| BCSS (5 classes) | 42.82 | 43.20 | 36.33 | 45.11 | 46.84 | 38.25 |

## GlaS gland instance segmentation

Object Dice is the arithmetic mean of official Test A and B results.

| Dataset | LeJEPA 100 | Simplex K=8 100 | MAE 100 | LeJEPA 200 | Simplex K=8 200 | MAE 200 |
|---|---:|---:|---:|---:|---:|---:|
| GlaS instance (mean A/B) | 44.53 | 42.20 | 38.98 | 44.47 | 44.81 | 42.30 |

## Data and protocol integrity

- Exactly three models: LeJEPA, Simplex-LeJEPA K=8, MAE. No K=64 or other K enters these tables.
- Classification: matched source manifest SHA, sample counts, preprocessing and classes.
- Retrieval: matched source records, caption IDs, protocol SHA and candidate sets.
- Patch retrieval: matched gallery/query IDs, row order and support subsets.
- Few-shot: same 100 fixed support selections per dataset, shot size and solver protocol.
- Segmentation: same source-manifest hashes and head protocols; five-class BCSS and separate gland-instance evaluation.
- Checkpoints at epochs 100 and 200 were selected before seeing test results. No cross-dataset winner-based filtering.
- All task means weight datasets equally, not individual samples.
- MAE uses the shared ViT-B/32 backbone; this is not a native ViT-B/16 MAE replication.
- Equal epochs and dataset exposure are not equal FLOPs: MAE uses masked one-view reconstruction, JEPA uses eight views.
- Selected MAE learning rate was 0.00015 from three sequential 20-epoch development trials.

## Reproduce

Run python3 scripts/aggregate_mae_cvpr_comparison.py from the Git repository root.
comparison_all_metrics.csv: all 124 metric rows, including auxiliary gland object F1.
comparison_summary.csv: all task means and dataset win counts.
comparison_audit.json: source JSON paths and SHA-256 hashes for each metric.
source/: exact small original experiment result JSONs, with no images or weight files.
experiment_metadata/: LR tuning outcome, resolved MAE configuration, 250-epoch training summary, and SHA-256 checkpoint manifest.
MAE checkpoints remain on gpu1-358-0; each is approximately 438 MB, so the four checkpoint weights are not stored in this Git repository.
