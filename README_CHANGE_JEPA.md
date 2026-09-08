# Change-JEPA degradation-dynamics SSL experiment

This experiment is isolated from B0, B0-Delta, B1, and the existing I-JEPA implementation.

## Objective

Reuse B0's calibrated adjacent degradation transitions:

```text
x_s = more-degraded image
x_t = adjacent less-degraded image
d = x_t - x_s                         in [-1, 1]
x_delta = (d + 1) / 2                 in [0, 1]
```

The mapping is exactly invertible by `d = 2*x_delta - 1`, so sign, magnitude, channel-wise change, and spatial structure are preserved.

The student encodes the full degraded context image `x_s`. The EMA teacher encodes `x_delta` using the same fresh PLIP ViT-B/32 architecture and the repository's normal image preprocessing. Teacher patch tokens are feature-wise layer-normalized, matching the existing repository I-JEPA target normalization.

For each of the 64 spatial patches, the predictor constructs a JEPA-style degradation query:

```text
shared learned query
+ fixed 2-D patch position
+ action embedding (defocus/resolution)
+ source degradation state embedding
+ delta-s embedding
```

The 64 projected context tokens and 64 query tokens are concatenated and passed through the 2-layer, 384-D Transformer predictor. Only the 64 query outputs are retained and projected to the 768-D teacher target space.

```text
P(S(x_s), action, source_state, delta_s, position) -> T_ema(x_delta)
```

The SSL loss is Smooth-L1 prediction loss only. VICReg is not used in this controlled experiment.

The first run deliberately retains B0's adjacent transitions, so `delta_s = -0.25` for every sample. This isolates the new change-image target and JEPA-style query mechanism before testing multi-step degradation dynamics.

## Training and early stopping

The run keeps B0's seed, 7,901 unlabeled PanNuke images, PLIP ViT-B/32 student, batch size 128, AdamW settings, LR/weight-decay schedules, EMA schedule, 300-epoch maximum schedule, and validation representation monitor.

The monitor runs every 10 epochs. A checkpoint is replaced only when validation linear macro-F1 improves by more than 0.005. Training stops after **3 consecutive monitor evaluations** without such an improvement, i.e. **30 epochs of patience**. The counter resets whenever a new checkpoint is selected. The test split is not used for SSL checkpoint selection.

## Weights & Biases

Change-JEPA requires W&B logging during SSL training. The API key is intentionally **not stored in GitHub**. Set it in the training shell:

```bash
export WANDB_API_KEY='<your W&B API key>'
```

Optional naming controls:

```bash
export WANDB_PROJECT='SSL_Histopath'
export WANDB_RUN_NAME='change-jepa-pilot'
# export WANDB_ENTITY='<your W&B team/entity>'   # only if needed
```

Each SSL epoch is logged as one W&B step. Logged training values include Smooth-L1 prediction loss, prediction cosine, target RMS, mean signed-change magnitude, student embedding std, gradient norm, learning rate, weight decay, EMA momentum, optimizer step, throughput, peak GPU memory, and all defocus/resolution transition-specific losses. On monitor epochs, W&B also receives k-NN and linear-probe validation metrics, representation-health diagnostics, selected best validation macro-F1/epoch, patience count, and the early-stop decision.

## Run commands

From `/raid1/xwan0900/SSL_proj`:

```bash
source /raid1/xwan0900/venvs/ftkp_cu128/bin/activate
python -m pip install -e .

# Set W&B authentication in this shell before training.
export WANDB_API_KEY='<your W&B API key>'

# 1. CPU core tests
pytest -q tests/test_change_jepa_core.py

# 2. Real-data/GPU objective + gradient smoke test
python scripts/smoke_change_jepa.py \
  --config configs/change_jepa_duration_pilot.yaml

# 3. SSL training; max 300 epochs, early stop after 3 failed monitors
python scripts/run_change_jepa_duration_pilot.py \
  --config configs/change_jepa_duration_pilot.yaml

# 4. Freeze selected student and extract CLEAN train/validation features
python scripts/change_jepa_duration_final_probe.py extract \
  --config configs/change_jepa_duration_final_probe.yaml

# 5. Tune the frozen linear probe on train/validation only
python scripts/change_jepa_duration_final_probe.py tune \
  --config configs/change_jepa_duration_final_probe.yaml

# 6. ONE-TIME test evaluation after all selection is frozen
python scripts/change_jepa_duration_final_probe.py test \
  --config configs/change_jepa_duration_final_probe.yaml
```

The final evaluation discards the teacher and predictor, freezes only the selected student encoder, feeds clean images, mean-pools the 64 patch tokens, and uses the same balanced PanNuke train/validation/test split and linear-probe hyperparameter grid as B0.
