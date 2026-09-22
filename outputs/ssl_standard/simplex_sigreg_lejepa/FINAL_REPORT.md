# Standard SSL Report — simplex_sigreg_lejepa

- SSL images: **6305**, unlabeled, from split **train**
- PanNuke split: **6305 / 798 / 798** train/val/test
- Representation: **mean_patch_tokens**
- Validation selection: **linear_val_macro_f1**
- Test protocol: **one evaluation after encoder/probe selection**

## Tuning

- Search strategy: **sequential_greedy**
- Selected learning rate: **0.0005**
- Selected simplex components K: **64**
- Fixed simplex sigma: **1.0**
- Source/recommended LR included: **True** (`0.0005`)
- Best tuning validation macro-F1: **0.465287**

## SSL pretraining

- Epochs completed: **300 / 300**
- Stop reason: **max_epochs**
- Selected epoch: **270**
- Best validation linear macro-F1: **0.8284941543295563**

## Downstream test

- Accuracy: **0.869674**
- Balanced accuracy: **0.869674**
- Macro-F1: **0.869544**
- Weighted F1: **0.869544**
- Test images: **798**

## Method source metadata

```json
{
  "baseline_implementation": "galilai-group/lejepa and released stable-pretraining recipe",
  "baseline_method": "lejepa",
  "baseline_sigreg_target": "standard_normal",
  "deliberate_adaptations": [
    "replace only SIGReg target N(0,I) with a fixed equal-weight regular-simplex isotropic GMM",
    "derive center distance as d=2*sigma and simplex scale C=2*sigma^2*(K-1)/K",
    "keep sigma fixed at 1.0 in the initial K study",
    "global resolution 256 instead of 224 for the shared backbone",
    "local resolution 96 instead of README 98 for an exact 3x3 patch-32 grid",
    "mean final patch tokens for the shared encoder readout",
    "warmup fraction 0.10 because README gives warmup form but no universal length"
  ],
  "implementation": "LeJEPA standard pipeline with a fixed regular-simplex isotropic Gaussian-mixture SIGReg target",
  "sigreg_target": "equal_weight_regular_simplex_isotropic_gmm",
  "simplex_components_tuned": true,
  "simplex_sigma_initial_study": 1.0,
  "simplex_spacing_rule": "d_equals_2_sigma",
  "source_global_scale": [
    0.3,
    1.0
  ],
  "source_local_scale": [
    0.05,
    0.3
  ],
  "source_peak_lr": 0.0005,
  "source_precision": "bf16",
  "source_schedule": "linear warmup plus cosine annealing; final LR = initial LR / 1000",
  "source_sigreg_lambda_released": 0.02,
  "source_views": "2_global_plus_6_local",
  "source_weight_decay_vit": 0.05
}
```
