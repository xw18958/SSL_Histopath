# DINOv3 vs LeJEPA vs Simplex-LeJEPA (K=8) vs MAE

Final verified comparison, 28 October 2026. All values are percentages; higher is better. Shared ViT-B/32 architecture and 7,901 PanNuke pretraining images, fixed epochs 100/200 and identical approved downstream protocols.

## Unweighted dataset means

| Task | Epoch | Datasets | LeJEPA | Simplex K=8 | MAE | DINOv3 |
|---|---:|---:|---:|---:|---:|---:|
| Classification (accuracy) | 100 | 11 | 75.86 | 76.40 | 69.59 | 69.07 |
| Classification (accuracy) | 200 | 11 | 77.82 | 78.89 | 71.94 | 70.43 |
| Image-text retrieval (mean recall) | 100 | 2 | 6.87 | 7.50 | 5.93 | 5.15 |
| Image-text retrieval (mean recall) | 200 | 2 | 7.70 | 8.99 | 7.13 | 5.21 |
| Patch i2i retrieval (mAP) | 100 | 11 | 49.92 | 50.09 | 45.80 | 45.18 |
| Patch i2i retrieval (mAP) | 200 | 11 | 50.26 | 50.69 | 46.52 | 43.63 |
| Few-shot average (all 1/5/10 shots) | 100 | 33 | 55.76 | 55.92 | 45.88 | 49.93 |
| Few-shot average (all 1/5/10 shots) | 200 | 33 | 56.77 | 57.52 | 47.69 | 48.31 |
| Semantic segmentation (mIoU) | 100 | 3 | 52.30 | 52.40 | 42.42 | 45.75 |
| Semantic segmentation (mIoU) | 200 | 3 | 54.23 | 55.24 | 49.36 | 45.59 |
| GlaS instance (object Dice) | 100 | 1 | 44.53 | 42.20 | 38.98 | 37.95 |
| GlaS instance (object Dice) | 200 | 1 | 44.47 | 44.81 | 42.30 | 39.74 |

## Trainable few-shot means by shot count

| Shot | Epoch | Datasets | LeJEPA | Simplex K=8 | MAE | DINOv3 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 11 | 48.10 | 47.99 | 41.89 | 43.49 |
| 1 | 200 | 11 | 48.55 | 48.83 | 43.04 | 41.55 |
| 5 | 100 | 11 | 57.58 | 57.75 | 46.86 | 51.65 |
| 5 | 200 | 11 | 58.70 | 59.58 | 48.76 | 49.80 |
| 10 | 100 | 11 | 61.59 | 62.02 | 48.88 | 54.65 |
| 10 | 200 | 11 | 63.06 | 64.15 | 51.29 | 53.59 |

## Classification - Accuracy

| Dataset | Epoch | LeJEPA | Simplex K=8 | MAE | DINOv3 |
|---|---:|---:|---:|---:|---:|
| BACH | 100 | **72.50** | 70.00 | 55.00 | 60.00 |
| BreaKHis | 100 | 61.93 | **66.76** | 50.85 | 49.43 |
| CRC-HE-7K | 100 | 96.08 | **97.39** | 95.10 | 92.48 |
| Endometrial | 100 | **53.77** | 51.89 | 44.34 | 41.98 |
| GaHISSDB | 100 | 87.00 | **88.50** | 84.00 | 85.50 |
| Kather | 100 | **91.53** | 91.13 | 90.73 | 87.50 |
| LC25000 | 100 | 96.00 | **96.40** | 93.20 | 92.80 |
| MHIST | 100 | 75.76 | **76.77** | 65.66 | 68.18 |
| Oral OSCC | 100 | 77.87 | 81.15 | **82.79** | 76.23 |
| RenalCell | 100 | 62.04 | **63.58** | 60.19 | 58.02 |
| SICAPv2 | 100 | **60.00** | 56.84 | 43.68 | 47.63 |
| BACH | 200 | 67.50 | **77.50** | 60.00 | 65.00 |
| BreaKHis | 200 | **74.43** | 73.01 | 59.94 | 54.55 |
| CRC-HE-7K | 200 | **98.37** | 98.04 | 96.08 | 92.16 |
| Endometrial | 200 | 53.77 | **54.72** | 45.28 | 46.23 |
| GaHISSDB | 200 | 91.50 | **92.00** | 85.00 | 85.50 |
| Kather | 200 | 93.95 | **94.56** | 90.93 | 90.12 |
| LC25000 | 200 | **97.60** | 97.20 | 93.40 | 92.40 |
| MHIST | 200 | 78.79 | **79.80** | 70.20 | 70.71 |
| Oral OSCC | 200 | 78.69 | **80.33** | 76.23 | 73.77 |
| RenalCell | 200 | **65.12** | 63.58 | 64.51 | 59.26 |
| SICAPv2 | 200 | 56.32 | **57.11** | 49.74 | 45.00 |


## Image-text retrieval - symmetric Mean Recall

| Dataset | Epoch | LeJEPA | Simplex K=8 | MAE | DINOv3 |
|---|---:|---:|---:|---:|---:|
| ARCH | 100 | 10.08 | **11.69** | 9.35 | 7.38 |
| PathCap | 100 | **3.65** | 3.31 | 2.52 | 2.93 |
| ARCH | 200 | 11.86 | **13.84** | 11.20 | 8.30 |
| PathCap | 200 | 3.53 | **4.13** | 3.07 | 2.12 |


## Patch image-to-image retrieval - mAP

| Dataset | Epoch | LeJEPA | Simplex K=8 | MAE | DINOv3 |
|---|---:|---:|---:|---:|---:|
| BACH | 100 | **37.57** | 37.08 | 35.69 | 35.46 |
| BreaKHis | 100 | 21.74 | **22.04** | 20.66 | 18.57 |
| CRC-HE-7K | 100 | **68.48** | 68.07 | 56.46 | 54.23 |
| Endometrial | 100 | 27.82 | **27.82** | 26.96 | 27.13 |
| GaHISSDB | 100 | **65.18** | 64.90 | 57.10 | 61.83 |
| Kather | 100 | 58.97 | **59.24** | 54.83 | 52.67 |
| LC25000 | 100 | 68.58 | **69.95** | 63.95 | 63.14 |
| MHIST | 100 | **54.20** | 54.13 | 52.72 | 53.07 |
| Oral OSCC | 100 | 68.08 | 68.16 | **69.01** | 66.77 |
| RenalCell | 100 | 45.79 | **46.46** | 39.41 | 36.24 |
| SICAPv2 | 100 | 32.76 | **33.11** | 26.97 | 27.83 |
| BACH | 200 | 39.39 | **39.71** | 36.22 | 34.25 |
| BreaKHis | 200 | 23.16 | **23.33** | 21.00 | 17.87 |
| CRC-HE-7K | 200 | 66.81 | **68.67** | 58.89 | 49.53 |
| Endometrial | 200 | 28.17 | **28.57** | 27.02 | 27.03 |
| GaHISSDB | 200 | 63.40 | **63.41** | 57.22 | 58.69 |
| Kather | 200 | 58.87 | **59.63** | 56.48 | 50.92 |
| LC25000 | 200 | 70.99 | **72.12** | 67.24 | 61.20 |
| MHIST | 200 | 53.99 | **54.11** | 52.53 | 51.58 |
| Oral OSCC | 200 | **69.03** | 69.03 | 68.78 | 66.65 |
| RenalCell | 200 | 46.88 | **47.07** | 38.88 | 34.61 |
| SICAPv2 | 200 | **32.19** | 31.92 | 27.45 | 27.60 |


## Semantic segmentation - mIoU

| Dataset | Epoch | LeJEPA | Simplex K=8 | MAE | DINOv3 |
|---|---:|---:|---:|---:|---:|
| BCSS (5 classes) | 100 | 42.82 | **43.20** | 36.33 | 38.18 |
| EBHI-SEG | 100 | 59.41 | **59.84** | 57.63 | 55.68 |
| GlaS (mean Test A/B) | 100 | **54.65** | 54.17 | 33.31 | 43.39 |
| BCSS (5 classes) | 200 | 45.11 | **46.84** | 38.25 | 39.79 |
| EBHI-SEG | 200 | 63.90 | **64.33** | 59.33 | 58.73 |
| GlaS (mean Test A/B) | 200 | 53.68 | **54.56** | 50.51 | 38.26 |


## GlaS instance segmentation - Object Dice

| Dataset | Epoch | LeJEPA | Simplex K=8 | MAE | DINOv3 |
|---|---:|---:|---:|---:|---:|
| GlaS instance (mean A/B) | 100 | **44.53** | 42.20 | 38.98 | 37.95 |
| GlaS instance (mean A/B) | 200 | 44.47 | **44.81** | 42.30 | 39.74 |


## Few-shot dataset detail (trainable logistic regression, 100 fixed support subsets)

### 1-shot accuracy

| Dataset | Epoch | LeJEPA | Simplex K=8 | MAE | DINOv3 |
|---|---:|---:|---:|---:|---:|
| BACH | 100 | **36.60** | 35.60 | 34.12 | 32.83 |
| BreaKHis | 100 | 22.92 | **23.22** | 21.24 | 19.56 |
| CRC-HE-7K | 100 | **67.66** | 66.96 | 49.35 | 52.86 |
| Endometrial | 100 | **27.71** | 27.14 | 26.33 | 26.13 |
| GaHISSDB | 100 | **65.33** | 64.92 | 56.91 | 62.08 |
| Kather | 100 | 56.19 | **56.30** | 40.95 | 52.05 |
| LC25000 | 100 | 67.08 | **67.89** | 55.86 | 63.82 |
| MHIST | 100 | 52.96 | 52.75 | 51.16 | **53.04** |
| Oral OSCC | 100 | 54.20 | 54.14 | **56.33** | 53.24 |
| RenalCell | 100 | 45.43 | **45.73** | 41.92 | 34.82 |
| SICAPv2 | 100 | 33.02 | **33.22** | 26.64 | 27.97 |
| BACH | 200 | **39.55** | 39.48 | 34.95 | 33.10 |
| BreaKHis | 200 | 24.39 | **24.53** | 21.73 | 19.26 |
| CRC-HE-7K | 200 | 66.62 | **68.62** | 53.53 | 48.70 |
| Endometrial | 200 | 27.82 | **28.35** | 26.49 | 26.32 |
| GaHISSDB | 200 | **63.51** | 63.47 | 57.17 | 57.45 |
| Kather | 200 | 56.10 | **56.86** | 44.85 | 50.29 |
| LC25000 | 200 | 69.97 | **71.04** | 58.95 | 59.64 |
| MHIST | 200 | **52.67** | 52.40 | 51.27 | 51.08 |
| Oral OSCC | 200 | 54.33 | 53.14 | **55.72** | 51.04 |
| RenalCell | 200 | 46.14 | **46.74** | 42.00 | 33.36 |
| SICAPv2 | 200 | **32.96** | 32.46 | 26.73 | 26.79 |


### 5-shot accuracy

| Dataset | Epoch | LeJEPA | Simplex K=8 | MAE | DINOv3 |
|---|---:|---:|---:|---:|---:|
| BACH | 100 | **46.75** | 46.27 | 40.27 | 42.65 |
| BreaKHis | 100 | 32.17 | **32.62** | 25.79 | 27.43 |
| CRC-HE-7K | 100 | **80.61** | 80.38 | 55.74 | 67.26 |
| Endometrial | 100 | 29.70 | **29.97** | 26.90 | 27.88 |
| GaHISSDB | 100 | 74.80 | **74.96** | 64.28 | 69.31 |
| Kather | 100 | 74.41 | **75.07** | 50.41 | 67.44 |
| LC25000 | 100 | 79.29 | **79.99** | 61.06 | 78.78 |
| MHIST | 100 | 57.72 | **57.75** | 51.92 | 55.92 |
| Oral OSCC | 100 | **57.99** | 57.31 | 56.26 | 55.22 |
| RenalCell | 100 | 57.00 | **57.31** | 53.30 | 45.50 |
| SICAPv2 | 100 | 42.90 | **43.57** | 29.53 | 30.75 |
| BACH | 200 | 51.85 | **52.50** | 41.90 | 41.73 |
| BreaKHis | 200 | 34.21 | **34.39** | 26.53 | 26.25 |
| CRC-HE-7K | 200 | 81.86 | **83.21** | 61.63 | 62.27 |
| Endometrial | 200 | 30.67 | **31.52** | 27.15 | 28.71 |
| GaHISSDB | 200 | 74.06 | **74.86** | 65.25 | 65.31 |
| Kather | 200 | 74.88 | **76.37** | 56.51 | 65.65 |
| LC25000 | 200 | 83.94 | **84.67** | 66.01 | 75.12 |
| MHIST | 200 | 56.81 | **57.38** | 51.61 | 51.98 |
| Oral OSCC | 200 | 57.62 | **59.82** | 56.05 | 54.07 |
| RenalCell | 200 | 58.26 | **59.14** | 53.45 | 44.60 |
| SICAPv2 | 200 | 41.52 | **41.58** | 30.28 | 32.12 |


### 10-shot accuracy

| Dataset | Epoch | LeJEPA | Simplex K=8 | MAE | DINOv3 |
|---|---:|---:|---:|---:|---:|
| BACH | 100 | 53.45 | **53.92** | 44.77 | 46.35 |
| BreaKHis | 100 | 36.42 | **37.36** | 27.19 | 31.25 |
| CRC-HE-7K | 100 | **84.50** | 84.18 | 58.14 | 72.02 |
| Endometrial | 100 | 33.12 | **33.39** | 28.10 | 29.49 |
| GaHISSDB | 100 | 77.03 | **77.45** | 66.62 | 72.50 |
| Kather | 100 | 79.28 | **80.05** | 54.75 | 72.05 |
| LC25000 | 100 | 83.11 | **83.90** | 61.69 | 81.30 |
| MHIST | 100 | 60.90 | **61.05** | 52.62 | 57.80 |
| Oral OSCC | 100 | **61.86** | 61.50 | 58.97 | 56.85 |
| RenalCell | 100 | 61.48 | **61.60** | 56.41 | 48.84 |
| SICAPv2 | 100 | 46.36 | **47.86** | 28.46 | 32.66 |
| BACH | 200 | 59.42 | **61.10** | 47.65 | 48.60 |
| BreaKHis | 200 | 38.80 | **39.35** | 28.05 | 30.96 |
| CRC-HE-7K | 200 | 86.07 | **87.11** | 64.74 | 68.50 |
| Endometrial | 200 | 32.99 | **35.26** | 28.90 | 30.75 |
| GaHISSDB | 200 | 76.72 | **77.38** | 67.85 | 68.47 |
| Kather | 200 | 80.09 | **81.38** | 61.38 | 70.07 |
| LC25000 | 200 | 87.67 | **88.38** | 67.74 | 77.42 |
| MHIST | 200 | 59.54 | **60.40** | 52.49 | 53.86 |
| Oral OSCC | 200 | 64.08 | **65.54** | 59.27 | 56.50 |
| RenalCell | 200 | 64.09 | **64.47** | 57.10 | 50.45 |
| SICAPv2 | 200 | 44.21 | **45.33** | 28.99 | 33.89 |


## Audit and reproduction

- comparison_all_metrics.csv contains 124 rows (including GlaS instance object F1 as an auxiliary metric). comparison_summary.csv contains unweighted task means and per-dataset win counts.
- comparison_audit.json records the SHA-256 of each original DINOv3 result file and corresponding immutable MAE source, along with manifest/protocol identity assertions. All audit checks passed.
- Downstream dataset source manifests and splits, retrieval relevance/candidate protocols, patch gallery/query row order, few-shot support subset hashes and 100 repeats, and segmentation manifests were checked against the MAE campaign.
- GlaS Test A/B: use the arithmetic mean for semantic mIoU and instance Object Dice, not the pooled or area-weighted combined metric.
- DINOv3 is the base-stage DINO+iBOT+KoLeo objective adapted to shared patch-32 CLIP/PLIP ViT-B/32, without separate Gram anchoring. This does not reproduce the entire DINOv3 paper pipeline.
- Equal epochs, data and batch size are not equal FLOPs. Preserve each method-specific objective, optimizer and augmentations.
- No checkpoint selection by downstream test, no re-tuning completed probes and no overwrite of completed tests.
- DINOv3 epoch-200 GaHISSDB was recovered using the preselected frozen probe after file-descriptor exhaustion. The original selection and exclusive test marker remain. Other interrupted jobs resumed after raising resource limits, disabling pin memory and restoring the original patch manifest.
- Raw DINOv3 result JSONs are retained on gpu1-358-0 and gpu2-358-0 in /home/xwan0900/dinov3_cvpr_20261028/downstream_results/. The four checkpoints remain on gpu1-358-0 under ssl_standard/runs/ssl-dinov3-cvpr20261028/pretrain_full/checkpoints/. Checkpoints are not committed to Git.
- Reproduce via: python scripts/aggregate_dinov3_cvpr_comparison.py --gpu1 PATH_TO_GPU1_DOWNSTREAMS --gpu2 PATH_TO_GPU2_DOWNSTREAMS --out OUTPUT_DIR --repo REPO_DIR.
