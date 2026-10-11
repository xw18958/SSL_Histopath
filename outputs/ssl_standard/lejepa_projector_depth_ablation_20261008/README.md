# Controlled LeJEPA Projector Depth Ablation: Final Results

All 180 original downstream `test_metrics.json` payloads were consolidated and verified (3 architectures x 4 checkpoints x 15 datasets).

## Architectures

- **Shallow:** `768 -> 512 -> 2048 -> 512`
- **Baseline:** `768 -> 512 -> 2048 -> 2048 -> 512`
- **Deep:** `768 -> 512 -> 2048 -> 2048 -> 2048 -> 512`

Only the number of internal 2048-wide MLP layers changes. The entrance bottleneck and normalization/activation convention are preserved.

## Test Macro-F1 (unweighted mean across 15 datasets)

| Epoch | Baseline | Shallow | Deep | Shallow vs baseline | Deep vs baseline |
|---:|---:|---:|---:|---:|---:|
| 100 | 71.5867% | 72.6808% | 71.5688% | +1.0941 pp | -0.0179 pp |
| 150 | 74.6324% | 72.4460% | 71.7638% | -2.1863 pp | -2.8686 pp |
| 200 | 74.7039% | 74.4887% | 72.7137% | -0.2153 pp | -1.9902 pp |
| 250 | 75.1105% | 73.7637% | 74.7247% | -1.3468 pp | -0.3858 pp |

At epoch 250, baseline reaches **75.1105%**, shallow **73.7637%**, and deep **74.7247%** mean Macro-F1.
The best epoch-250 model by dataset is baseline on 6 datasets, shallow on 4, and deep on 5.

## Protocol and provenance

- **Seed:** 20260903; pretraining: 7,901 PanNuke images per epoch.
- **LeJEPA:** SIGReg lambda = 0.02, training batch size = 128.
- **Schedule:** 300-epoch learning-rate schedule for all variants; ablations stop after epoch 250. The completed baseline trained 300 epochs, but only matched epochs 100/150/200/250 are compared.
- **Evaluation:** 15 dataset-specific downstream classification probes; use stored test Macro-F1, not validation Macro-F1.
- **Important:** single pretraining seed per architecture. Dataset-specific differences and mean differences do not establish a causal shielding mechanism or statistical significance.

## Files

- `classification_results.csv`: all 180 rows, including accuracy, balanced accuracy, macro-F1, weighted F1, test image counts.
- `all_results.json`: full original `test_metrics.json` payloads for all 180 evaluations, including experiment and dataset metadata.
- `summary.json`: numerical means, baseline deltas, dataset wins, and training/commit provenance.
- No model checkpoints, datasets, or prediction images are committed. Original checkpoints and other run artifacts remain in server storage.
