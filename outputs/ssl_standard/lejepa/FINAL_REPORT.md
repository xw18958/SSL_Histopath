# Standard SSL Report — lejepa

- SSL images: **7901**, unlabeled
- Representation: **mean_patch_tokens**
- Validation selection: **linear_val_macro_f1**
- Test protocol: **one evaluation after encoder/probe selection**

## Tuning

- Selected learning rate: **0.0001**
- Source/recommended LR included: **True** (`0.0005`)
- Best tuning validation macro-F1: **0.530532**

## SSL pretraining

- Epochs completed: **280 / 300**
- Stop reason: **validation_plateau**
- Selected epoch: **230**
- Best validation linear macro-F1: **0.8989585818876437**

## Downstream test

- Accuracy: **0.866397**
- Balanced accuracy: **0.866397**
- Macro-F1: **0.866324**
- Weighted F1: **0.866324**
- Test images: **247**

## Method source metadata

```json
{
  "deliberate_adaptations": [
    "global resolution 256 instead of 224 for the shared backbone",
    "local resolution 96 instead of README 98 for an exact 3x3 patch-32 grid",
    "mean final patch tokens for the shared encoder readout",
    "warmup fraction 0.10 because README gives warmup form but no universal length"
  ],
  "implementation": "galilai-group/lejepa and released stable-pretraining recipe",
  "reference": "README GOTO hyperparameters",
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
