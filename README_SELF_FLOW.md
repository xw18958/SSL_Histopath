# Self-Flow-Pixel-B/32 PanNuke Baseline

This implementation adds a fair Self-Flow baseline for the existing B0 and I-JEPA PanNuke SSL experiments. Existing B0/I-JEPA source, configs, checkpoints, and outputs are not modified.

## What is implemented

`Self-Flow-Pixel-B/32` is explicitly a **pixel-space adaptation** of Self-Flow rather than the released latent ImageNet checkpoint/training setup.

Fixed comparison choices:

- SSL data: the same 7,901 unlabeled PanNuke images used by B0/I-JEPA.
- Input: 256x256 RGB; no labels during SSL.
- No pretrained VAE/tokenizer and no class conditioning.
- Flow domain: RGB `[0,1] -> [-1,1]`.
- Patch size: 32x32 -> 64 raw pixel-patch tokens.
- Hidden width: 768; 12 attention heads; MLP ratio 4.
- Transformer depth: **8**, not 12. Self-Flow's per-token adaLN adds substantial parameters per block; 8 blocks produce 94.08M trainable parameters, close to the existing B0/I-JEPA student+predictor totals. The smoke test computes the exact ratios from the local baseline code and requires Self-Flow to be within 10% of both.
- Fixed 2D sinusoidal positional embeddings.
- Self-Flow per-token timestep conditioning and adaLN-Zero blocks.
- Dual-Timestep Scheduling: independent `t,s ~ Uniform(0,1)`, exact 25% token mask, teacher timestep `min(t,s)`, same Gaussian noise for student and teacher.
- Rectified-flow target: `v = noise - x0` for `x_tau = (1-tau)x0 + tau*noise`.
- Self-distillation: projected student layer 2 predicts raw EMA-teacher layer 6 features with cosine loss. Layers 2/6 are the proportional 8-layer mapping of the released 8/20 layers in the 28-layer Self-Flow model.
- Loss: `L = L_flow + 0.5 * L_rep`.
- EMA: cosine 0.996 -> 1.0, matching the existing B0/I-JEPA comparison pipeline.
- Gradient clip: 1.0, matching the released Self-Flow training detail.
- Training horizon/effective batch/optimizer schedules/checkpoint monitor: same benchmark settings as the existing duration runs.
- Checkpoint selection: validation macro-F1 only, using **clean t=0 final-layer tokens -> mean pool**, matching the existing shared representation monitor. Test is never used for checkpoint selection.
- Final probe: same LR/weight-decay grid, patience, balanced PanNuke split, final-layer mean-pooled 768-D features, and one final test evaluation.

The released Self-Flow repository provides inference/model code but not the full training script. Therefore `t,s ~ Uniform(0,1)`, `representation_weight=0.5`, the pixel-space input, and the 8-layer capacity match are documented benchmark adaptation choices rather than claims about an unreleased official training configuration.

## Files

- `src/pannuke_ssl/self_flow.py` - model, pixel patchification, Dual-Timestep Scheduling, and objective.
- `src/pannuke_ssl/self_flow_training.py` - 300-epoch validation-selected pretraining run.
- `src/pannuke_ssl/self_flow_probe.py` - frozen final linear probe.
- `configs/self_flow_pixel_b32_duration_pilot.yaml` - locked pretraining configuration.
- `configs/self_flow_pixel_b32_duration_final_probe.yaml` - locked final-probe configuration.
- `scripts/smoke_test_self_flow_pixel_b32.py` - shape/loss/gradient/EMA/capacity sanity checks.
- `scripts/pretrain_self_flow_pixel_b32.py` - pretraining entry point.
- `scripts/self_flow_pixel_b32_duration_final_probe.py` - final probe entry point.

## Run order

From `/raid1/xwan0900/SSL_proj`:

```bash
python scripts/smoke_test_self_flow_pixel_b32.py \
  --config configs/self_flow_pixel_b32_duration_pilot.yaml
```

The smoke test must pass before training. It writes:

```text
outputs/self_flow_pixel_b32_duration_pilot/smoke_test.json
```

Then run pretraining:

```bash
python scripts/pretrain_self_flow_pixel_b32.py \
  --config configs/self_flow_pixel_b32_duration_pilot.yaml
```

Important outputs:

```text
outputs/self_flow_pixel_b32_duration_pilot/
  checkpoints/best.pt
  pretrain_metrics.csv
  monitor_metrics.csv
  duration_selection.json
  pretrain_summary.json
```

Finally run the frozen linear probe once:

```bash
python scripts/self_flow_pixel_b32_duration_final_probe.py \
  --config configs/self_flow_pixel_b32_duration_final_probe.yaml
```

Important final outputs:

```text
outputs/self_flow_pixel_b32_duration_final_probe/
  probe_leaderboard.csv
  probe_metrics.csv
  test_per_class_metrics.csv
  test_confusion_matrix.csv
  linear_probe_summary.json
```

Both the duration run and final probe refuse to overwrite completed output files.
