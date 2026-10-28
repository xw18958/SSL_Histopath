#!/usr/bin/env bash
# Approved dependency gate: start 250-epoch DINOv3 ONLY after all three 20-epoch
# sequential tuning trials complete successfully on gpu1-jinman-2.
set -euo pipefail
S="$DINOV3_CODE_ROOT"
RUN="$DINOV3_RUN_ROOT"
PY="$DINOV3_PYTHON"
TUNE_HOST=xwan0900@gpu1-jinman-2.it.usyd.edu.au
TUNE_DIR=/raid1/xwan0900/dinov3_cvpr_20261028/ssl_standard/runs/ssl-dinov3-cvpr20261028/tuning
RUNDIR="$RUN/ssl_standard/runs/ssl-dinov3-cvpr20261028"
mkdir -p "$RUN/logs"
test ! -f "$RUNDIR/pretrain_full/run_summary.json" || { echo "ABORT: pretraining already completed"; exit 9; }
test ! -f "$RUNDIR/pretrain_full/pretrain_metrics.csv" || { echo "ABORT: pretraining already started"; exit 9; }
echo "[DINOv3] $(date -Is) Wait for successful three-candidate upstream tuning"
while :; do
  if ssh -T -oBatchMode=yes -oConnectTimeout=12 "$TUNE_HOST" "test -f '$TUNE_DIR/best_hyperparameters.yaml' && test -f '$TUNE_DIR/tuning_summary.json'"; then
    break
  fi
  if ssh -T -oBatchMode=yes -oConnectTimeout=12 "$TUNE_HOST" 'test -f /raid1/xwan0900/dinov3_cvpr_20261028/logs/tuning.pid && kill -0 "$(cat /raid1/xwan0900/dinov3_cvpr_20261028/logs/tuning.pid)" 2>/dev/null'; then
    sleep 90
  else
    echo "STOP: Tuning not completed and no live tuner"; exit 7
  fi
done
TEMP="$RUN/logs/tune_selected_$(date +%s).yaml"
ssh -T -oBatchMode=yes -oConnectTimeout=12 "$TUNE_HOST" "cat '$TUNE_DIR/best_hyperparameters.yaml'" > "$TEMP"
LR=$("$PY" - "$TEMP" <<'PY'
import sys,yaml
s=yaml.safe_load(open(sys.argv[1]))
assert s["method"]=="dinov3"
assert s["selection"]["test_used"] is False
assert s["selection"]["search_strategy"]=="single_parameter"
value=float(s["selected"]["learning_rate"])
assert value in (0.0001,0.0003535533905932738,0.001),value
print(f"{value:.10g}")
PY
)
echo "[DINOv3] $(date -Is) Tuning COMPLETE. Selected LR=$LR"
printf 'selected_peak_lr: %s\nsource_tuning_host: %s\n' "$LR" "$TUNE_HOST" > "$RUN/logs/pretrain_handoff.yaml"
export SSL_PROJECT_ROOT="$S"
export SSL_DATA_ROOT="$DINOV3_DATA_ROOT"
export SSL_MODEL_ROOT="$DINOV3_MODEL_ROOT"
export SSL_RUN_ROOT="$RUN"
export PYTHONPATH="$S/src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="$DINOV3_GPU"
"$PY" "$S/scripts/run_ssl_standard.py" pretrain --method dinov3 \
  --run-id ssl-dinov3-cvpr20261028 --learning-rate "$LR"
test -f "$RUNDIR/pretrain_full/run_summary.json"
for epoch in 100 150 200 250; do
  test -s "$RUNDIR/pretrain_full/checkpoints/epoch_$epoch.pt"
done
date -Is > "$RUN/logs/DINOV3_PRETRAIN_COMPLETED.ok"
echo "[DINOv3] $(date -Is) Final 250 epochs completed and checkpoints verified"
