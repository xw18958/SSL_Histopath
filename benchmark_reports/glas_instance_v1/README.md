# GlaS gland-instance segmentation — frozen SSL encoder probe

**Verified four full runs, original GlaS Test A (60) and Test B (20).**
This is a NEW gland **instance** segmentation experiment; do not confuse
with the older GlaS **binary semantic** mIoU in
`benchmark_reports/pathology_segmentation_v1`.

### Detection of individual glands: object F1 (%), higher better

| Test | Epochs | LeJEPA | Simplex K=8 | Simplex Δ (pp) |
|---|---:|---:|---:|---:|
| Test A | 100 | **18.92** | 15.68 | -3.24 |
| Test A | 200 | **19.24** | 18.56 | -0.69 |
| Test B | 100 | **26.77** | 18.96 | -7.81 |
| Test B | 200 | 26.82 | **29.21** | +2.39 |

### Symmetric area-weighted object Dice (%), higher better

| Test | Epochs | LeJEPA | Simplex K=8 | Simplex Δ (pp) |
|---|---:|---:|---:|---:|
| Test A | 100 | **35.73** | 35.41 | -0.32 |
| Test A | 200 | **35.10** | 34.81 | -0.29 |
| Test B | 100 | **53.32** | 48.99 | -4.34 |
| Test B | 200 | 53.85 | **54.81** | +0.96 |

### Symmetric area-weighted object Hausdorff (pixels), LOWER better

| Test | Epochs | LeJEPA | Simplex K=8 |
|---|---:|---:|---:|
| Test A | 100 | 458.66 | **436.14** |
| Test A | 200 | **469.50** | 472.29 |
| Test B | 100 | 349.72 | **342.55** |
| Test B | 200 | 350.40 | **344.59** |

**Interpretation:** Simplex K=8 is not uniformly better. At 100 epochs
LeJEPA has stronger instance detection and object Dice in both test sets.
At 200 epochs Simplex slightly exceeds LeJEPA on Test B, but not Test A.
Absolute object-F1 values are poor, showing that an **8×8 frozen ViT-B/32
grid with a lightweight instance-distance head has limited gland-instance
separation capacity.** Do not cherry-pick Test B to claim a general win.

## Method and provenance

Full instructions: `docs/GLAS_INSTANCE_FROZEN.md`.

- Source images: 85 official train and 80 official test, with 70/15
  train/validation subset of the original training set.
- Original full native-size RGB images and *integer instance ID masks*;
  train images sampled into 256×256 crops without masking or instance collapse.
- Frozen ViT-B/32 spatial tokens, head with gland foreground and normalized
  per-gland distance regression. Overlapping native-resolution test inference
  with validation-selected marker watershed.
- Test A **60** and Test B **20** image-wise scores averaged separately;
  all per-image F1, object Dice and object Hausdorff are in `per_image.csv`.
- Source GlaS manifest SHA256 and four immutable SSL checkpoint hashes
  are recorded in `verification.json`.
- No changes to SSL backbone/projector, training, old segmentation or
  pre-existing downstream benchmark results.
- All four full runs completed across three GPU servers, after 8 unit
  tests and three server GPU smoke tests.
- **Non-equivalence caveat:** The metric family follows GlaS's challenge
  definitions; the Python implementation explicitly prevents duplicate
  TP matching and assigns image-diagonal penalties for empty predictions.
  The source official MATLAB evaluator has **not** been executed on these
  predictions. Do not present them as official leaderboard scores. Cite
  GlaS challenge Sirinukunwattana et al. (MedIA 2017) and indicate that
  these are *challenge-style* Python scores.

Files:
`summary.csv` (eight rows),
`paired.csv` (four matched comparisons),
`per_image.csv` (320 sample-metric records),
`verification.json` (audited hashes, split & metric provenance),
`scripts/run_glas_instance.py` (training, native inference, postprocess),
`scripts/aggregate_glas_instance.py` (independent result audit).

Server outputs: `SSL_runs/glas_instance_20261020/results/<method>/epoch_<epoch>`.
Each contains the final trained `instance_head.pt`, `selection.json`,
`test_started.json` and full `result.json`. Smoke output stays under
`SSL_runs/glas_instance_20261020/smoke`.
