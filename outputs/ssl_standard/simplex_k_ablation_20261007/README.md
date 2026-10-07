# Simplex LeJEPA K ablation — completed 7 October 2026

All 216 evaluations completed: 180 classification and 36 retrieval evaluations across K=8/16/32 and epochs 100/150/200/250. The native verifier checked all 12 checkpoint SHA-256 identities against completion records and every downstream result.

Settings: batch 128, seed 20260903, sigma 1, peak LR 0.0005, stop and LR schedule 250. Existing valid completed evaluations were preserved during recovery.

Classification uses frozen encoders with trained heads. Means below give each of the 15 datasets equal weight. Retrieval values are overall mean recall: the mean of R@1, R@5 and R@10 in both directions. All values are percentages. Each retrieval dataset uses the original 4,667-pair sample and native TRAIN/VAL/TEST procedures. No fine-tuning or zero-shot evaluations were run in this campaign.

| K | Epoch | Mean accuracy | Mean macro F1 | ARCH recall | IPATH recall | PathCap recall |
|---|---|---|---|---|---|---|
| 8 | 100 | 74.91 | 72.14 | 11.69 | 7.35 | 3.31 |
| 8 | 150 | 75.51 | 73.29 | 13.81 | 6.80 | 4.40 |
| 8 | 200 | 77.04 | 75.35 | 13.84 | 8.68 | 4.13 |
| 8 | 250 | 76.71 | 74.84 | 14.00 | 7.30 | 4.42 |
| 16 | 100 | 73.37 | 71.04 | 11.64 | 7.49 | 3.45 |
| 16 | 150 | 73.91 | 71.21 | 11.26 | 7.12 | 3.48 |
| 16 | 200 | 77.77 | 75.57 | 12.87 | 8.52 | 3.45 |
| 16 | 250 | 76.57 | 74.35 | 13.06 | 8.33 | 4.06 |
| 32 | 100 | 74.08 | 71.52 | 12.05 | 6.25 | 3.58 |
| 32 | 150 | 75.09 | 72.86 | 12.18 | 8.71 | 3.24 |
| 32 | 200 | 76.81 | 74.75 | 12.54 | 7.98 | 3.36 |
| 32 | 250 | 77.66 | 75.72 | 12.67 | 7.58 | 3.38 |

All four checkpoints are shown; these tables do not select checkpoints by test performance. Full per-dataset classification metrics and directional retrieval R@1/5/10 are in [classification_results.csv](classification_results.csv) and [retrieval_results.csv](retrieval_results.csv). Machine-readable complete metrics are in [all_results.json](all_results.json).

Recovery code is on branch `feat/simplex-k-ablation-ready-20261007`; persistent worker storage, transport retry and idempotent dispatch prevent interrupted transport from restarting valid work.
