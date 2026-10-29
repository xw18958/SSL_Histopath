# I-JEPA matched histopathology baseline (29 Oct 2026)

The predictor remains 4 layers, as explicitly requested. This is an
architecture-matched adaptation, not an unaltered original paper reproduction.

## Code corrections
- Standard I-JEPA reads all masking and predictor parameters from YAML.
  Current 384-dimensional 4-layer 12-head predictor is unchanged.
- I-JEPA source fallback can relax exclusion constraints on an 8x8 grid.
  Per-image resampling rejects context overlapping target blocks, preserving
  target block scale/aspect/count, and the training step asserts no leakage.
- CPU tests cover geometry, reproducibility, and predictor shape.
  GPU smoke verifies target pixel perturbation does not change student
  visible-context features, teacher frozen, finite gradients, and EMA update.

## Experimental protocol
- Shared randomly initialised CLIP ViT-B/32, 256 px, patch32.
- Seed 20260903; batch128; BF16.
- Tuning: three sequential 20-epoch LR trials at 1e-4, 3e-4, 1e-3,
  using 6305 PanNuke train images and development validation only.
- Final SSL: full 7901 unlabeled PanNuke images, 250 epochs and
  fixed checkpoints at epochs 100/150/200/250; publish 100/200 comparisons.
- Frozen EMA target encoder with 64 patch mean-pooling for downstream.
- Existing downstream adapter branch is inherited unchanged, evaluating
  classification, ARCH/PathCap image-text retrieval, patch retrieval,
  trainable 1/5/10-shot, GlaS/EBHI/BCSS segmentation and GlaS instance.
- The epoch/data budget is matched; FLOPs and view count differ by method.

## Execution
Export IJEPA_CODE_ROOT, IJEPA_RUN_ROOT, IJEPA_DATA_ROOT, IJEPA_MODEL_ROOT,
IJEPA_PYTHON and IJEPA_GPU, then run scripts/ijepa_cvpr_campaign.sh.
The runner verifies CUDA smoke, tunes LR, and trains fixed 250 epochs.
It refuses to overwrite partial or existing pretraining results.
Downstream requires validated 100/200 checkpoints.
