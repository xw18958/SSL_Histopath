# DINOv3 source audit and fair histopathology SSL preflight

Status: code, configuration and downstream adapters prepared.
**NO new DINOv3 tuning or full pretraining has been started.**

## Official sources reviewed

- Paper: https://arxiv.org/html/2508.10104 (Sec. 3.2 and 4.2)
- Meta code: https://github.com/facebookresearch/dinov3
- Specifically: dinov3/train/ssl_meta_arch.py,
  dinov3/train/param_groups.py,
  dinov3/loss/dino_clstoken_loss.py,
  dinov3/loss/ibot_patch_loss.py,
  dinov3/layers/dino_head.py,
  dinov3/data/augmentations.py and
  dinov3/configs/ssl_default_config.yaml.

## Controlled comparisons (MAE / LeJEPA / Simplex K=8 / DINOv3)

| Dimension | DINOv3 setting identical to baselines |
|---|---|
| Encoder | Randomly initialised CLIP/PLIP ViT-B/32, 768-D, 12 layers |
| Pretraining images | PanNuke, 7,901 unlabelled images, same fixed source IDs |
| Development | 6,305 train and combined 1,596 internal development |
| Seed / runs | Seed 20260903, one final SSL run |
| Batch | 128 source images per step; BF16 and TF32 |
| Full budget | 250 epochs, no early stopping, no development test for final training |
| Fixed checkpoints | 100, 150, 200 and 250; 100/200 primary |
| LR tuning budget | Three sequential trials of 20 epochs each, val macro-F1 only |
| Frozen readout | 768-D mean final patch tokens; DINO uses EMA teacher |
| Classification | 11 selected external datasets, same frozen linear probes |
| Image/text retrieval | ARCH and PathCap, same PLIP frozen text protocol |
| Image/image retrieval | Same patch dataset manifest, train gallery/test query, mAP |
| Trainable few-shot | 1/5/10 shots, 100 fixed support subsets, accuracy |
| Segmentation | GlaS and EBHI semantic, official five-class BCSS, GlaS native instance |

Previously excluded datasets remain excluded from CLASSIFICATION:
PCam, EBHI-SEG, WSSS4LUAD, Osteosarcoma, IPATH. EBHI-SEG
remains in the separately requested SEGMENTATION evaluation.

## DINOv3 source-specific algorithm retained

- DINO class-token self-distillation, masked iBOT patch distillation,
  KoLeo on student global CLS tokens, with Meta's loss scales.
- 2 global 256px views, 8 local 96px views; source crop scales and
  asymmetric augmentation (blur, solarization, jitter, grayscale).
- Separate 65,536-prototype DINO and iBOT heads; 3-layer MLPs
  with hidden dim 2048, bottleneck 256.
- Source Sinkhorn-Knopp teacher assignments, EMA teacher schedule,
  student temperature 0.1, teacher temperature 0.04->0.07,
  iBOT mask fraction 0.1-0.5 on half the global views.
- AdamW betas (0.9, 0.999), weight-decay 0.04->0.4,
  gradient clipping 3.0, 1-epoch head last-layer freeze.
- CORRECTED: the previous DINOv3 implementation omitted
  source layerwise LR decay 0.9 and patch-embedding LR multiplier 0.2.
  These are now mapped to CLIP encoder.layers and patch_embedding,
  and preserved on each schedule update. Existing MAE/LeJEPA/Simplex
  param groups have default LR/WD multiplier 1 (unchanged).
- CORRECTED tuning: the official raw 0.001 learning rate remains
  one candidate; include its sqrt_wrt_1024 batch-size scaling at
  batch 128 (0.0003535533905932738) and the earlier 0.0001
  candidate, under the same three-trial development selection budget.

## What counts as DINOv3 here

The benchmark is the DINOv3 BASE-STAGE objective.
The DINOv3 paper introduces Gram anchoring as a later, separate
refinement stage with an earlier EMA Gram teacher. The official
default source config has gram.use_loss=false, and this code does not
perform that later Gram refinement. The results must not be called a
full reproduction of the complete paper.

DINOv3's native original models use a different architecture,
often ViT patch-16 / RoPE with orders of magnitude more data and
compute. Adopting them here would destroy the controlled shared
ViT-B/32 comparison. The shared patch-32 encoder gives only 3x3
local patch grids. This is a limitation, not an implementation bug.
Likewise, preserving 65,536 prototypes is source-faithful, but
may not be optimal for 7,901 pathology images; do not silently
change K or patch size.

The paper's largest models use long constant optimizer schedules;
the released ssl_default_config.yaml uses warmup/cosine source
schedules. This controlled baseline follows the released default
recipe adapted to the fixed 250-epoch budget, with the source
warmup fractions.

## Verification

- DINOv3 + MAE + classification/segmentation regression suite: 56 tests passed.
- CUDA one-step (synthetic batch 2) training smoke: PASS with original
  65,536-prototype heads and 2 global + 8 local views.
  Loss 17.07349, finite gradient norm 1.796945, 28 optimizer
  groups, peak allocated VRAM 2.184 GiB.
- Identical fixed-seed initial shared backbone: all 199 tensors equal
  (DINOv3 student.base vs a fresh baseline backbone).
- No full 128-image per-step throughput timing was performed;
  synthetic batch-2 VRAM does not predict batch-128 VRAM.
- No held-out test images accessed in this audit.

## Prepared launch workflow (NOT EXECUTED)

The new scripts follow the same immutable MAE workflow:
- First run the three-candidate development tuning using
  scripts/run_ssl_standard.py tune --method dinov3.
- scripts/approved_dinov3_pretrain_after_tuning.sh strictly gates
  final 250-epoch training on the finished selected LR.
- scripts/approved_dinov3_downstreams_after_pretrain.sh gates
  the same downstream tasks on verified 100/200 checkpoints.
- scripts/run_dinov3_selected_downstreams.py and
  scripts/patch_benchmark_dinov3.py reuse existing task code.
- These are not running, and existing DINOv3, MAE, LeJEPA,
  Simplex outputs and checkpoints are not overwritten.


## Follow-up source-parity correction (28 October 2026)

- DINOv3 last-layer projection-head freeze is now measured using the actual
  loader step count, **50 development steps** (6,305 / 128) and **62 final-data
  steps** (7,901 / 128). The training loop passes `len(loader)` to DINOv3;
  MAE, LeJEPA and Simplex are not modified.
- Gaussian blur now matches the **literal released Meta DINOv3 code**:
  `dinov3/data/transforms.py::GaussianBlur`, using a **9-pixel kernel** and
  torchvision `RandomApply(p=1-nominal_p)`. Although unusual, this official
  code means global1=0%, global2=90%, local=50% actual blur rates, rather
  than the nominal 100%/10%/50%. This interpretation is explicitly pinned
  by regression tests. It is not being presented as DINOv2 augmentation.
- These changes should be validated on CUDA (full batch 128) before tuning,
  followed by the same three 20-epoch LR trials, 250-epoch full-data training,
  100/200 checkpoints and MAE-identical external downstream protocols.
