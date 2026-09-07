# B0-Delta diagnostic experiment

B0-Delta is a controlled diagnostic of the completed B0 degradation-conditioned
SSL baseline. It keeps the B0 data, degradation sampler, fresh PLIP ViT-B/32
student, EMA teacher, B0 predictor, VICReg control, optimizer/schedules,
validation checkpoint rule, and final linear-probe pipeline fixed. The configured
maximum training horizon remains 300 epochs, but B0-Delta now uses validation-
based early stopping to avoid continuing a clearly stagnant run.

The only SSL target change is:

```text
B0:
P(S(x_s), s, action, delta) -> T(x_t)

B0-Delta:
P(S(x_s), s, action, delta) -> T(x_t) - T(x_s)
```

`x_s` is the more-degraded source and `x_t` is the adjacent less-degraded
target. The subtraction is performed in FP32 using the same EMA teacher for
both endpoints. The target is detached. No masking, cropping, pretrained
teacher, or architecture change is introduced.

## Why this experiment exists

It tests one hypothesis only: B0 may be dominated by representation information
already shared between the degraded and cleaner views. B0-Delta removes that
shared component from the supervision and directly predicts the
degradation-induced teacher residual.

This is a diagnostic experiment, not the final proposed method.

## Locked B0-matched protocol

The config intentionally matches the completed B0 duration experiment except
for the explicitly requested early-stopping rule:

- seed: 20260903
- 7,901 unlabeled PanNuke images
- 256x256 input, PLIP ViT-B/32, 64 x 768 patch tokens
- B0 predictor: dim 384, depth 2, 6 heads, no dropout
- adjacent degradation transitions only: 1.00->0.75, 0.75->0.50,
  0.50->0.25, 0.25->0.00
- batch/effective batch: 128/128
- AdamW
- LR: 1e-4 -> 1e-6, 10% warmup
- weight decay: 0.04 -> 0.40
- EMA: 0.996 -> 1.0
- gradient clipping: 5
- VICReg control: lambda=0.10, covariance weight=0.04, variance target=1.0
- validation representation monitor every 10 epochs
- selected SSL checkpoint: strict >=0.005 improvement in validation linear
  macro-F1, same checkpoint criterion as B0
- maximum SSL horizon: 300 epochs
- early stopping: stop after 5 consecutive monitor evaluations without a
  checkpoint-qualifying improvement = 50 epochs of patience

The early-stopping counter resets to zero whenever a new checkpoint is selected.
The earliest possible stop is epoch 50. For example, if epoch 40 is the last
selected checkpoint, non-improvements at epochs 50, 60, 70, 80 and 90 stop the
run at epoch 90 while retaining the epoch-40 checkpoint. Test data are never
used for this decision.

The trainer refuses configs that drift from these values.

## Run order

From `/raid1/xwan0900/SSL_proj`:

```bash
source /raid1/xwan0900/venvs/ftkp_cu128/bin/activate
python -m pip install -e .

# 1. CPU unit tests
pytest -q tests/test_b0_delta_core.py

# 2. Real-data/GPU one-batch objective smoke test
python scripts/smoke_b0_delta.py \
  --config configs/b0_delta_duration_pilot.yaml

# 3. Validation-selected SSL run, maximum 300 epochs with 50-epoch patience
#    (NO test evaluation)
python scripts/run_b0_delta_duration_pilot.py \
  --config configs/b0_delta_duration_pilot.yaml

# 4. Lock selected encoder and extract CLEAN train/validation features
python scripts/extract_b0_delta_duration_probe_features.py \
  --config configs/b0_delta_duration_final_probe.yaml

# 5. Exercise the exact probe loop for two epochs, still NO test access
python scripts/smoke_b0_delta_duration_final_probe.py \
  --config configs/b0_delta_duration_final_probe.yaml

# 6. Tune the frozen linear probe on train/validation only
python scripts/tune_b0_delta_duration_final_probe.py \
  --config configs/b0_delta_duration_final_probe.yaml

# 7. ONE-TIME test evaluation after all selection is frozen
python scripts/test_b0_delta_duration_final_probe_once.py \
  --config configs/b0_delta_duration_final_probe.yaml
```

Do not run step 7 more than once. The final-probe code creates an exclusive
`test_started.json` marker before decoding the test split.

## Outputs

SSL:

```text
outputs/b0_delta_duration_pilot/
  checkpoints/best.pt
  resolved_config.json
  pretrain_metrics.csv
  pretrain_summary.json
  duration_selection.json
  monitor_metrics.csv
  pretrain_losses.png
  embedding_std.png
  residual_diagnostics.png
  representation_validation_curves.png
  representation_health.png
```

`duration_selection.json` and `pretrain_summary.json` record `epochs_run`,
`stopped_early`, `early_stop_epoch`, the five-evaluation patience, and the
selected validation checkpoint.

Final probe:

```text
outputs/b0_delta_duration_pilot_final_probe/
  encoder_lock.json
  train_val_features.npz
  feature_extraction_audit.json
  smoke_test.json
  probe_leaderboard.csv
  all_trial_metrics.csv
  selected_probe_metrics.csv
  selection.json
  best_linear_probe.pt
  test_started.json
  test_features.npz
  test_predictions.npz
  test_per_class_metrics.csv
  test_confusion_matrix.csv
  test_confusion_matrix.png
  linear_probe_summary.json
```

## Residual diagnostics

`pretrain_metrics.csv` adds diagnostics that do not affect optimization or
checkpoint selection:

- `zero_prediction`: Smooth-L1 loss for predicting an all-zero residual
- `residual_skill = 1 - prediction / zero_prediction`
- `residual_cosine`: cosine similarity between predicted and teacher residual
- `target_delta_rms`: RMS magnitude of the teacher residual
- transition-specific prediction/zero/skill values for both defocus and
  resolution

These are necessary because a residual target can be numerically small. A low
raw prediction loss alone is not evidence that the residual was learned.

## Comparison rule

For the first diagnostic, compare B0-Delta against the already completed B0
using validation representation curves and the same frozen clean-image linear
probe. Do not retune the B0-Delta SSL hyperparameters before seeing this
controlled result. In particular, `lambda_reg=0.10` is intentionally retained
for the first run. The only training-control difference from the completed B0
run is the requested validation-based early stopping; the maximum schedule and
all optimization hyperparameters remain fixed.
