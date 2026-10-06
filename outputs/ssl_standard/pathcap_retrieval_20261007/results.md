# PathCap results

All four CUDA-capable NVIDIA GPUs on the three servers were used. gpu1-358-0 also exposes Intel integrated graphics in nvtop.

The fixed sample contains 3,267 TRAIN / 700 VAL / 700 TEST images. The TEST set contains 679 unique captions. All ten heads were selected before TEST began. Only image projection heads were trained.

| SSL epoch | Method | I→T R@1 | R@5 | R@10 | T→I R@1 | R@5 | R@10 | Mean Recall |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 100 | LeJEPA | 0.43 | 3.86 | 6.43 | 0.59 | 4.27 | 6.33 | 3.65 |
| 100 | Simplex K=64 | 0.57 | 3.43 | 5.86 | 0.59 | 3.39 | 6.63 | 3.41 |
| 150 | LeJEPA | 1.00 | 3.14 | 5.71 | 0.59 | 3.39 | 6.04 | 3.31 |
| 150 | Simplex K=64 | 0.86 | 3.00 | 6.00 | 0.59 | 3.68 | 7.22 | 3.56 |
| 200 | LeJEPA | 0.86 | 3.00 | 6.14 | 0.44 | 4.12 | 6.63 | 3.53 |
| 200 | Simplex K=64 | 0.71 | 1.86 | 6.43 | 1.03 | 3.24 | 6.63 | 3.32 |
| 250 | LeJEPA | 0.57 | 3.29 | 6.86 | 0.74 | 4.57 | 8.10 | 4.02 |
| 250 | Simplex K=64 | 0.57 | 3.00 | 4.43 | 0.44 | 1.62 | 4.27 | 2.39 |
| 300 | LeJEPA | 0.57 | 3.71 | 6.43 | 0.74 | 4.27 | 7.51 | 3.87 |
| 300 | Simplex K=64 | 0.43 | 3.71 | 6.86 | 0.74 | 3.68 | 7.22 | 3.77 |

All scores above are percentages. Random Mean Recall under the caption-aware relevance definition is 0.785%.

Verification: 22 focused unit tests and eight GPU smoke tests passed. Checkpoint checksums and frozen weights match; no earlier experiments were rerun.

Full persistent artifacts: /raid1/xwan0900/SSL_runs/pathcap_retrieval_20261007 on Jinman. Smoke tests, temporary runtime, and logs are outside the project.
