# DINOv3 standardized PanNuke baseline

This adds DINOv3 to the same controlled SSL protocol already used for I-JEPA and LeJEPA.

## Fair-comparison rule

DINOv3 uses the **same freshly initialized PLIP/CLIP ViT-B/32 architecture** as the existing standardized baselines. No DINOv3 or PLIP pretrained weights are loaded. The shared protocol is unchanged: 7,901 unlabeled PanNuke images, seed `20260903`, batch size 128, at most 300 SSL epochs, validation-only encoder selection, mean final patch-token representation, the same linear-probe search, and one test evaluation only after encoder/probe selection.

The DINOv3-specific training objective is the base pretraining recipe: DINO self-distillation + iBOT masked-patch prediction + KoLeo regularization. Gram anchoring is intentionally not mixed into this initial baseline because the official DINOv3 recipe introduces it as a separate later stage.

Method-specific source behavior retained here includes 2 global + 8 local crops, DINO/iBOT separate projection heads, Sinkhorn-Knopp teacher assignments, 50% iBOT mask sampling with 10%-50% patch masking, EMA teacher evaluation, DINOv3 teacher temperature/momentum schedules, and the DINOv3 asymmetric blur/solarization augmentations.

## Run

```bash
python scripts/run_ssl_standard.py pipeline --method dinov3
```

This performs the same three-stage path as the existing standardized methods:

1. 20-epoch / 3-candidate learning-rate tuning using validation macro-F1 only.
2. Full SSL pretraining with the standard validation/early-stopping checkpoint selection.
3. Frozen-feature linear probing and exactly one final test evaluation.

Outputs are written under `outputs/ssl_standard/dinov3/`.
