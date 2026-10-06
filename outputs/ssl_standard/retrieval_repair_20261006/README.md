# Corrected retrieval results — 6 October 2026

The tokenizer failure and repeated-caption handling have been repaired. Both servers use matching code, PLIP assets, canonical manifests, and retrieval image bytes. Twenty retrieval heads were retrained and evaluated. The existing SSL checkpoints and all 150 classification evaluations were retained.

All saved checkpoints are reported; none is designated primary. Classification metrics are frozen linear-probe TEST results. Full encoder fine-tuning and zero-shot results are not available in this experiment bundle.

This is a versioned correction after the original TEST results were inspected. Replacement hyperparameters and heads were selected using TRAIN/VAL, with all selections sealed before replacement TEST evaluation.

## Retrieval

Scores below are Mean Recall (%), averaging image-to-text and text-to-image R@1/5/10. Evaluation uses unique caption candidates and accepts every image associated with the exact caption. Query weighting is uniform over images for image-to-text and unique captions for text-to-image. Exact-score ties are averaged over uniform tie order. ARCH has 700 images and 582 caption queries; IPATH has 700 images and 426 caption queries. No distinct full-caption tokenization collisions were observed on either TEST set.

| SSL epoch | ARCH LeJEPA | ARCH Simplex | IPATH LeJEPA | IPATH Simplex |
|---:|---:|---:|---:|---:|
| 100 | 10.08 | 10.57 | 8.46 | 8.23 |
| 150 | 10.88 | 10.95 | 9.39 | 7.02 |
| 200 | 11.86 | 12.44 | 8.80 | 7.88 |
| 250 | 12.70 | 11.94 | 10.02 | 8.60 |
| 300 | 12.93 | 13.12 | 9.86 | 8.36 |

The previous large retrieval advantage is not supported by the corrected experiment. ARCH performance is close, while LeJEPA has higher IPATH Mean Recall at all five saved checkpoints. The corrected relevance definition differs from the retired pair-ID metric, so old and new percentages should not be compared as identical benchmarks.

## Retained classification

Means below give each of the 15 datasets equal weight. These values were not retrained or reevaluated.

| SSL epoch | LeJEPA Accuracy | Simplex Accuracy | LeJEPA macro-F1 | Simplex macro-F1 |
|---:|---:|---:|---:|---:|
| 100 | 74.25 | 75.94 | 71.59 | 74.18 |
| 150 | 76.61 | 75.84 | 74.63 | 73.94 |
| 200 | 76.95 | 77.23 | 74.70 | 75.21 |
| 250 | 77.70 | 77.51 | 75.11 | 75.20 |
| 300 | 77.37 | 76.65 | 75.42 | 74.46 |

Dataset-specific custom splits and their actual independence units are retained in `dataset_provenance.json`. Comparisons with published scores require matching protocols.

## Detailed retrieval

| Dataset | SSL epoch | Method | I→T R@1 | R@5 | R@10 | T→I R@1 | R@5 | R@10 | Mean Recall |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| ARCH | 100 | lejepa | 2.71 | 9.86 | 15.43 | 2.92 | 11.51 | 18.04 | 10.08 |
| ARCH | 100 | simplex_sigreg_lejepa | 3.86 | 10.43 | 18.57 | 2.58 | 11.00 | 17.01 | 10.57 |
| ARCH | 150 | lejepa | 2.86 | 10.14 | 16.71 | 3.26 | 12.54 | 19.76 | 10.88 |
| ARCH | 150 | simplex_sigreg_lejepa | 2.71 | 9.86 | 17.57 | 3.61 | 12.37 | 19.59 | 10.95 |
| ARCH | 200 | lejepa | 3.14 | 11.57 | 19.86 | 3.78 | 12.71 | 20.10 | 11.86 |
| ARCH | 200 | simplex_sigreg_lejepa | 2.71 | 12.86 | 21.43 | 3.09 | 13.23 | 21.31 | 12.44 |
| ARCH | 250 | lejepa | 4.57 | 14.29 | 20.57 | 3.95 | 12.37 | 20.45 | 12.70 |
| ARCH | 250 | simplex_sigreg_lejepa | 3.00 | 12.00 | 18.00 | 4.12 | 12.89 | 21.65 | 11.94 |
| ARCH | 300 | lejepa | 4.57 | 14.14 | 20.71 | 3.95 | 13.06 | 21.13 | 12.93 |
| ARCH | 300 | simplex_sigreg_lejepa | 4.14 | 13.57 | 20.29 | 4.30 | 13.92 | 22.51 | 13.12 |
| IPATH | 100 | lejepa | 2.57 | 8.29 | 13.14 | 3.29 | 9.15 | 14.32 | 8.46 |
| IPATH | 100 | simplex_sigreg_lejepa | 2.29 | 8.57 | 13.86 | 2.58 | 8.45 | 13.62 | 8.23 |
| IPATH | 150 | lejepa | 2.00 | 8.71 | 15.57 | 3.52 | 10.80 | 15.73 | 9.39 |
| IPATH | 150 | simplex_sigreg_lejepa | 1.86 | 7.14 | 12.71 | 1.64 | 6.81 | 11.97 | 7.02 |
| IPATH | 200 | lejepa | 2.57 | 9.14 | 13.14 | 2.58 | 9.62 | 15.73 | 8.80 |
| IPATH | 200 | simplex_sigreg_lejepa | 2.43 | 7.00 | 12.29 | 1.41 | 9.15 | 15.02 | 7.88 |
| IPATH | 250 | lejepa | 3.57 | 10.29 | 14.57 | 3.29 | 11.03 | 17.37 | 10.02 |
| IPATH | 250 | simplex_sigreg_lejepa | 2.29 | 8.00 | 14.57 | 1.64 | 10.09 | 15.02 | 8.60 |
| IPATH | 300 | lejepa | 3.14 | 9.86 | 15.43 | 2.82 | 11.03 | 16.90 | 9.86 |
| IPATH | 300 | simplex_sigreg_lejepa | 2.00 | 8.71 | 14.57 | 2.82 | 8.69 | 13.38 | 8.36 |

## Detailed retained classification

| Dataset | SSL epoch | Method | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---|---:|---:|---:|---:|
| bach | 100 | lejepa | 72.50 | 72.50 | 71.45 | 71.45 |
| bach | 100 | simplex_k64 | 65.00 | 65.00 | 64.82 | 64.82 |
| bach | 150 | lejepa | 65.00 | 65.00 | 61.31 | 61.31 |
| bach | 150 | simplex_k64 | 67.50 | 67.50 | 66.92 | 66.92 |
| bach | 200 | lejepa | 67.50 | 67.50 | 66.12 | 66.12 |
| bach | 200 | simplex_k64 | 77.50 | 77.50 | 75.83 | 75.83 |
| bach | 250 | lejepa | 72.50 | 72.50 | 71.69 | 71.69 |
| bach | 250 | simplex_k64 | 75.00 | 75.00 | 73.98 | 73.98 |
| bach | 300 | lejepa | 70.00 | 70.00 | 69.22 | 69.22 |
| bach | 300 | simplex_k64 | 75.00 | 75.00 | 74.51 | 74.51 |
| breakhis_8subtype | 100 | lejepa | 61.93 | 61.93 | 60.56 | 60.56 |
| breakhis_8subtype | 100 | simplex_k64 | 68.47 | 68.47 | 67.93 | 67.93 |
| breakhis_8subtype | 150 | lejepa | 75.85 | 75.85 | 75.46 | 75.46 |
| breakhis_8subtype | 150 | simplex_k64 | 72.44 | 72.44 | 71.91 | 71.91 |
| breakhis_8subtype | 200 | lejepa | 74.43 | 74.43 | 74.27 | 74.27 |
| breakhis_8subtype | 200 | simplex_k64 | 70.74 | 70.74 | 69.87 | 69.87 |
| breakhis_8subtype | 250 | lejepa | 77.84 | 77.84 | 77.41 | 77.41 |
| breakhis_8subtype | 250 | simplex_k64 | 76.14 | 76.14 | 76.17 | 76.17 |
| breakhis_8subtype | 300 | lejepa | 74.72 | 74.72 | 74.30 | 74.30 |
| breakhis_8subtype | 300 | simplex_k64 | 73.58 | 73.58 | 73.30 | 73.30 |
| crc_val_he_7k | 100 | lejepa | 96.08 | 96.08 | 96.07 | 96.07 |
| crc_val_he_7k | 100 | simplex_k64 | 96.08 | 96.08 | 96.08 | 96.08 |
| crc_val_he_7k | 150 | lejepa | 97.71 | 97.71 | 97.71 | 97.71 |
| crc_val_he_7k | 150 | simplex_k64 | 98.04 | 98.04 | 98.04 | 98.04 |
| crc_val_he_7k | 200 | lejepa | 98.37 | 98.37 | 98.37 | 98.37 |
| crc_val_he_7k | 200 | simplex_k64 | 96.73 | 96.73 | 96.73 | 96.73 |
| crc_val_he_7k | 250 | lejepa | 98.69 | 98.69 | 98.69 | 98.69 |
| crc_val_he_7k | 250 | simplex_k64 | 98.37 | 98.37 | 98.36 | 98.36 |
| crc_val_he_7k | 300 | lejepa | 98.04 | 98.04 | 98.04 | 98.04 |
| crc_val_he_7k | 300 | simplex_k64 | 98.69 | 98.69 | 98.69 | 98.69 |
| ebhi_seg_6class | 100 | lejepa | 73.80 | 59.09 | 56.41 | 71.87 |
| ebhi_seg_6class | 100 | simplex_k64 | 76.47 | 72.57 | 72.16 | 76.40 |
| ebhi_seg_6class | 150 | lejepa | 71.66 | 63.40 | 64.87 | 71.50 |
| ebhi_seg_6class | 150 | simplex_k64 | 72.73 | 63.82 | 65.05 | 71.65 |
| ebhi_seg_6class | 200 | lejepa | 75.40 | 63.05 | 61.24 | 74.28 |
| ebhi_seg_6class | 200 | simplex_k64 | 74.87 | 63.60 | 64.52 | 73.28 |
| ebhi_seg_6class | 250 | lejepa | 76.47 | 64.22 | 59.74 | 75.18 |
| ebhi_seg_6class | 250 | simplex_k64 | 74.87 | 61.00 | 61.12 | 72.97 |
| ebhi_seg_6class | 300 | lejepa | 74.33 | 64.93 | 63.38 | 72.64 |
| ebhi_seg_6class | 300 | simplex_k64 | 75.94 | 67.96 | 66.12 | 74.32 |
| endometrial_4class | 100 | lejepa | 53.77 | 53.77 | 53.90 | 53.90 |
| endometrial_4class | 100 | simplex_k64 | 55.19 | 55.19 | 55.23 | 55.23 |
| endometrial_4class | 150 | lejepa | 55.66 | 55.66 | 55.73 | 55.73 |
| endometrial_4class | 150 | simplex_k64 | 51.89 | 51.89 | 51.89 | 51.89 |
| endometrial_4class | 200 | lejepa | 53.77 | 53.77 | 53.47 | 53.47 |
| endometrial_4class | 200 | simplex_k64 | 55.66 | 55.66 | 55.77 | 55.77 |
| endometrial_4class | 250 | lejepa | 56.13 | 56.13 | 56.23 | 56.23 |
| endometrial_4class | 250 | simplex_k64 | 54.72 | 54.72 | 54.83 | 54.83 |
| endometrial_4class | 300 | lejepa | 54.72 | 54.72 | 54.74 | 54.74 |
| endometrial_4class | 300 | simplex_k64 | 54.72 | 54.72 | 54.83 | 54.83 |
| gashissdb_binary | 100 | lejepa | 87.00 | 87.00 | 86.98 | 86.98 |
| gashissdb_binary | 100 | simplex_k64 | 86.50 | 86.50 | 86.47 | 86.47 |
| gashissdb_binary | 150 | lejepa | 89.50 | 89.50 | 89.50 | 89.50 |
| gashissdb_binary | 150 | simplex_k64 | 88.00 | 88.00 | 87.99 | 87.99 |
| gashissdb_binary | 200 | lejepa | 91.50 | 91.50 | 91.49 | 91.49 |
| gashissdb_binary | 200 | simplex_k64 | 90.50 | 90.50 | 90.49 | 90.49 |
| gashissdb_binary | 250 | lejepa | 87.50 | 87.50 | 87.45 | 87.45 |
| gashissdb_binary | 250 | simplex_k64 | 89.00 | 89.00 | 88.99 | 88.99 |
| gashissdb_binary | 300 | lejepa | 86.50 | 86.50 | 86.46 | 86.46 |
| gashissdb_binary | 300 | simplex_k64 | 85.50 | 85.50 | 85.47 | 85.47 |
| kather_2016 | 100 | lejepa | 91.53 | 91.53 | 91.57 | 91.57 |
| kather_2016 | 100 | simplex_k64 | 92.74 | 92.74 | 92.74 | 92.74 |
| kather_2016 | 150 | lejepa | 91.94 | 91.94 | 91.88 | 91.88 |
| kather_2016 | 150 | simplex_k64 | 93.75 | 93.75 | 93.84 | 93.84 |
| kather_2016 | 200 | lejepa | 93.95 | 93.95 | 93.99 | 93.99 |
| kather_2016 | 200 | simplex_k64 | 92.14 | 92.14 | 92.08 | 92.08 |
| kather_2016 | 250 | lejepa | 95.77 | 95.77 | 95.75 | 95.75 |
| kather_2016 | 250 | simplex_k64 | 94.15 | 94.15 | 94.14 | 94.14 |
| kather_2016 | 300 | lejepa | 93.95 | 93.95 | 93.96 | 93.96 |
| kather_2016 | 300 | simplex_k64 | 95.16 | 95.16 | 95.22 | 95.22 |
| lc25000_5class | 100 | lejepa | 96.00 | 96.00 | 96.00 | 96.00 |
| lc25000_5class | 100 | simplex_k64 | 96.00 | 96.00 | 95.99 | 95.99 |
| lc25000_5class | 150 | lejepa | 97.80 | 97.80 | 97.80 | 97.80 |
| lc25000_5class | 150 | simplex_k64 | 97.20 | 97.20 | 97.19 | 97.19 |
| lc25000_5class | 200 | lejepa | 97.60 | 97.60 | 97.60 | 97.60 |
| lc25000_5class | 200 | simplex_k64 | 97.40 | 97.40 | 97.40 | 97.40 |
| lc25000_5class | 250 | lejepa | 97.80 | 97.80 | 97.80 | 97.80 |
| lc25000_5class | 250 | simplex_k64 | 97.20 | 97.20 | 97.20 | 97.20 |
| lc25000_5class | 300 | lejepa | 97.80 | 97.80 | 97.80 | 97.80 |
| lc25000_5class | 300 | simplex_k64 | 96.80 | 96.80 | 96.80 | 96.80 |
| mhist | 100 | lejepa | 75.76 | 75.76 | 75.72 | 75.72 |
| mhist | 100 | simplex_k64 | 79.80 | 79.80 | 79.78 | 79.78 |
| mhist | 150 | lejepa | 79.29 | 79.29 | 79.23 | 79.23 |
| mhist | 150 | simplex_k64 | 78.79 | 78.79 | 78.79 | 78.79 |
| mhist | 200 | lejepa | 78.79 | 78.79 | 78.77 | 78.77 |
| mhist | 200 | simplex_k64 | 79.29 | 79.29 | 79.29 | 79.29 |
| mhist | 250 | lejepa | 82.83 | 82.83 | 82.78 | 82.78 |
| mhist | 250 | simplex_k64 | 83.33 | 83.33 | 83.33 | 83.33 |
| mhist | 300 | lejepa | 82.83 | 82.83 | 82.81 | 82.81 |
| mhist | 300 | simplex_k64 | 82.83 | 82.83 | 82.82 | 82.82 |
| oral_oscc | 100 | lejepa | 77.87 | 73.62 | 71.69 | 78.63 |
| oral_oscc | 100 | simplex_k64 | 85.25 | 76.08 | 78.00 | 84.63 |
| oral_oscc | 150 | lejepa | 81.97 | 78.68 | 76.70 | 82.51 |
| oral_oscc | 150 | simplex_k64 | 81.97 | 78.68 | 76.70 | 82.51 |
| oral_oscc | 200 | lejepa | 78.69 | 78.90 | 74.39 | 79.89 |
| oral_oscc | 200 | simplex_k64 | 84.43 | 79.11 | 78.76 | 84.52 |
| oral_oscc | 250 | lejepa | 82.79 | 79.22 | 77.53 | 83.23 |
| oral_oscc | 250 | simplex_k64 | 84.43 | 81.48 | 79.67 | 84.83 |
| oral_oscc | 300 | lejepa | 82.79 | 80.40 | 77.98 | 83.38 |
| oral_oscc | 300 | simplex_k64 | 81.97 | 76.31 | 75.69 | 82.17 |
| osteosarcoma_3class | 100 | lejepa | 46.18 | 31.99 | 34.49 | 50.58 |
| osteosarcoma_3class | 100 | simplex_k64 | 46.47 | 33.36 | 35.34 | 50.71 |
| osteosarcoma_3class | 150 | lejepa | 45.88 | 32.63 | 35.98 | 51.66 |
| osteosarcoma_3class | 150 | simplex_k64 | 50.29 | 35.91 | 36.04 | 52.41 |
| osteosarcoma_3class | 200 | lejepa | 52.94 | 54.53 | 41.45 | 58.34 |
| osteosarcoma_3class | 200 | simplex_k64 | 50.88 | 53.14 | 39.96 | 56.12 |
| osteosarcoma_3class | 250 | lejepa | 50.59 | 35.35 | 38.96 | 56.69 |
| osteosarcoma_3class | 250 | simplex_k64 | 52.35 | 54.28 | 40.10 | 56.44 |
| osteosarcoma_3class | 300 | lejepa | 47.94 | 34.75 | 37.68 | 53.57 |
| osteosarcoma_3class | 300 | simplex_k64 | 50.29 | 53.13 | 38.84 | 54.37 |
| pcam_binary | 100 | lejepa | 81.00 | 81.00 | 80.99 | 80.99 |
| pcam_binary | 100 | simplex_k64 | 81.00 | 81.00 | 80.91 | 80.91 |
| pcam_binary | 150 | lejepa | 86.50 | 86.50 | 86.48 | 86.48 |
| pcam_binary | 150 | simplex_k64 | 78.50 | 78.50 | 78.50 | 78.50 |
| pcam_binary | 200 | lejepa | 86.50 | 86.50 | 86.48 | 86.48 |
| pcam_binary | 200 | simplex_k64 | 79.00 | 79.00 | 78.92 | 78.92 |
| pcam_binary | 250 | lejepa | 88.00 | 88.00 | 87.96 | 87.96 |
| pcam_binary | 250 | simplex_k64 | 75.50 | 75.50 | 74.90 | 74.90 |
| pcam_binary | 300 | lejepa | 90.00 | 90.00 | 89.96 | 89.96 |
| pcam_binary | 300 | simplex_k64 | 70.00 | 70.00 | 68.62 | 68.62 |
| renalcell_6class | 100 | lejepa | 62.04 | 62.04 | 60.60 | 60.60 |
| renalcell_6class | 100 | simplex_k64 | 63.89 | 63.89 | 63.36 | 63.36 |
| renalcell_6class | 150 | lejepa | 65.12 | 65.12 | 64.45 | 64.45 |
| renalcell_6class | 150 | simplex_k64 | 65.74 | 65.74 | 64.99 | 64.99 |
| renalcell_6class | 200 | lejepa | 65.12 | 65.12 | 64.44 | 64.44 |
| renalcell_6class | 200 | simplex_k64 | 68.83 | 68.83 | 67.71 | 67.71 |
| renalcell_6class | 250 | lejepa | 64.81 | 64.81 | 63.21 | 63.21 |
| renalcell_6class | 250 | simplex_k64 | 67.28 | 67.28 | 66.27 | 66.27 |
| renalcell_6class | 300 | lejepa | 62.35 | 62.35 | 60.68 | 60.68 |
| renalcell_6class | 300 | simplex_k64 | 67.90 | 67.90 | 66.61 | 66.61 |
| sicapv2_4class | 100 | lejepa | 60.00 | 60.00 | 59.23 | 59.23 |
| sicapv2_4class | 100 | simplex_k64 | 61.32 | 61.32 | 58.88 | 58.88 |
| sicapv2_4class | 150 | lejepa | 60.26 | 60.26 | 57.39 | 57.39 |
| sicapv2_4class | 150 | simplex_k64 | 58.16 | 58.16 | 58.62 | 58.62 |
| sicapv2_4class | 200 | lejepa | 56.32 | 56.32 | 55.47 | 55.47 |
| sicapv2_4class | 200 | simplex_k64 | 53.16 | 53.16 | 53.42 | 53.42 |
| sicapv2_4class | 250 | lejepa | 51.84 | 51.84 | 49.74 | 49.74 |
| sicapv2_4class | 250 | simplex_k64 | 55.26 | 55.26 | 53.99 | 53.99 |
| sicapv2_4class | 300 | lejepa | 59.21 | 59.21 | 58.99 | 58.99 |
| sicapv2_4class | 300 | simplex_k64 | 53.68 | 53.68 | 51.74 | 51.74 |
| wsss4luad_3class | 100 | lejepa | 78.33 | 78.33 | 78.16 | 78.16 |
| wsss4luad_3class | 100 | simplex_k64 | 85.00 | 85.00 | 84.95 | 84.95 |
| wsss4luad_3class | 150 | lejepa | 85.00 | 85.00 | 84.98 | 84.98 |
| wsss4luad_3class | 150 | simplex_k64 | 82.67 | 82.67 | 82.63 | 82.63 |
| wsss4luad_3class | 200 | lejepa | 83.33 | 83.33 | 83.01 | 83.01 |
| wsss4luad_3class | 200 | simplex_k64 | 87.33 | 87.33 | 87.36 | 87.36 |
| wsss4luad_3class | 250 | lejepa | 82.00 | 82.00 | 81.72 | 81.72 |
| wsss4luad_3class | 250 | simplex_k64 | 85.00 | 85.00 | 84.93 | 84.93 |
| wsss4luad_3class | 300 | lejepa | 85.33 | 85.33 | 85.25 | 85.25 |
| wsss4luad_3class | 300 | simplex_k64 | 87.67 | 87.67 | 87.65 | 87.65 |

## Artifact policy

Original LeJEPA retrieval metrics and heads are invalid because of the tokenizer failure. Original Simplex retrieval outputs are superseded by the caption-aware protocol. Original files remain archived with checksums on both servers; neither category enters the corrected tables. Raw images, captions, canonical splits, pretraining artifacts, and classification artifacts were preserved.

Verification: 15 focused tests passed on each server; four TRAIN/VAL-only smoke tests passed; 17 canonical manifests and 9,334 retrieval image files match between servers; all retained artifact checksums match their original preservation ledgers. See `verification_report.json` for the precise checks and signatures.
