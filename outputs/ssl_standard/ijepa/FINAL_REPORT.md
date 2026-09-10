# Standard SSL Report — ijepa

- SSL images: **7901**, unlabeled
- Representation: **mean_patch_tokens**
- Validation selection: **linear_val_macro_f1**
- Test protocol: **one evaluation after encoder/probe selection**

## Tuning

- Selected learning rate: **0.0003**
- Source/recommended LR included: **True** (`0.001`)
- Best tuning validation macro-F1: **0.471034**

## SSL pretraining

- Epochs completed: **110 / 300**
- Stop reason: **validation_plateau**
- Selected epoch: **30**
- Best validation linear macro-F1: **0.5058726927698299**

## Downstream test

- Accuracy: **0.384615**
- Balanced accuracy: **0.384615**
- Macro-F1: **0.378139**
- Weighted F1: **0.378139**
- Test images: **247**

## Method source metadata

```json
{
  "deliberate_adaptations": [
    "256 crop instead of 224 for the shared backbone",
    "patch-32 gives an 8x8 token grid",
    "predictor depth 4 instead of ViT-H source depth 12",
    "target block minimum 4 for the coarse 8x8 grid",
    "predictor heads 12 inherited from the shared encoder"
  ],
  "evaluation_encoder": "target_encoder_ema",
  "evaluation_pooling": "average_patch_tokens",
  "implementation": "facebookresearch/ijepa",
  "reference_config": "configs/in1k_vith14_ep300.yaml",
  "source_ema": [
    0.996,
    1.0
  ],
  "source_ema_schedule": "linear",
  "source_final_lr": 1e-06,
  "source_final_weight_decay": 0.4,
  "source_peak_lr": 0.001,
  "source_pred_dim": 384,
  "source_start_lr": 0.0002,
  "source_warmup_epochs_of_300": 40,
  "source_weight_decay": 0.04
}
```
