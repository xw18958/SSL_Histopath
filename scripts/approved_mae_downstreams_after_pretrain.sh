#!/usr/bin/env bash
# Approved MAE downstream campaign, strictly gated on completed 250-epoch
# fixed-checkpoint pretraining; inherits existing frozen-head protocols.
set -euo pipefail
S="$MAE_CODE_ROOT"
RUN="$MAE_RUN_ROOT"
PY="$MAE_PYTHON"
ROLE="$MAE_ROLE"
PEER=xwan0900@gpu1-358-0.research.sydney.edu.au
REMOTE=/home/xwan0900/mae_cvpr_20261027
RUNDIR="$REMOTE/ssl_standard/runs/ssl-mae-cvpr20261027"
mkdir -p "$RUN/logs"
if test "$ROLE" = gpu1; then
  echo "[DOWNSTREAM] $(date -Is) Wait for local 250-epoch training completion"
  while ! test -f "$RUN/logs/MAE_PRETRAIN_COMPLETED.ok"; do
    if ! test -f "$RUN/logs/pretrain_gate.pid" ||
        ! kill -0 "$(cat "$RUN/logs/pretrain_gate.pid")" 2>/dev/null; then
      echo "STOP: local pretraining failed or no longer running"; exit 8
    fi
    sleep 90
  done
  CKPT="$RUN/ssl_standard/runs/ssl-mae-cvpr20261027/pretrain_full/checkpoints"
elif test "$ROLE" = gpu2; then
  echo "[DOWNSTREAM] $(date -Is) Wait for remote 250-epoch pretraining"
  while ! ssh -T -oBatchMode=yes -oConnectTimeout=12 "$PEER" "test -f '$REMOTE/logs/MAE_PRETRAIN_COMPLETED.ok'"; do
    if ! ssh -T -oBatchMode=yes -oConnectTimeout=12 "$PEER" \
         'test -f /home/xwan0900/mae_cvpr_20261027/logs/pretrain_gate.pid && kill -0 "$(cat /home/xwan0900/mae_cvpr_20261027/logs/pretrain_gate.pid)" 2>/dev/null'; then
      echo "STOP: upstream pretraining failed or missing"; exit 8
    fi
    sleep 90
  done
  while ! test -f "$RUN/logs/GPU2_BENCHMARK_DATA_READY.ok"; do
    if ! test -f "$RUN/logs/benchmark_data_sync.pid" ||
         ! kill -0 "$(cat "$RUN/logs/benchmark_data_sync.pid")" 2>/dev/null; then
      echo "STOP: benchmark data stage failed"; exit 8
    fi
    sleep 90
  done
  test -f "$RUN/logs/GPU2_ENV_READY.ok"
  test -f "$RUN/logs/GPU2_DATA_READY.ok"
  CKPT="$RUN/checkpoints"
  mkdir -p "$CKPT"
  for epoch in 100 200; do
    echo "STAGE APPROVED CHECKPOINT epoch=$epoch"
    rsync -a --partial "$PEER:$RUNDIR/pretrain_full/checkpoints/epoch_$epoch.pt" "$CKPT/epoch_$epoch.pt"
    expected=$(ssh -T -oBatchMode=yes "$PEER" "sha256sum '$RUNDIR/pretrain_full/checkpoints/epoch_$epoch.pt'" | cut -d' ' -f1)
    actual=$(sha256sum "$CKPT/epoch_$epoch.pt" | cut -d' ' -f1)
    if test "$actual" != "$expected"; then echo "Checkpoint SHA mismatch e$epoch"; exit 11; fi
    printf '%d %s\n' "$epoch" "$actual" >> "$RUN/logs/checkpoints_verified.sha256"
  done
else
  echo "Invalid role: $ROLE"; exit 2
fi
test -s "$CKPT/epoch_100.pt"
test -s "$CKPT/epoch_200.pt"
export SSL_PROJECT_ROOT="$S"
export SSL_DATA_ROOT="$MAE_DATA_ROOT"
export SSL_MODEL_ROOT="$MAE_MODEL_ROOT"
export SSL_RUN_ROOT="$RUN"
export MAE_SEG_MANIFEST_ROOT="$RUN/manifests/segmentation"
export PYTHONPATH="$S/src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MAE_CKPT_ROOT="$CKPT"
export SSL_MAE_CKPT_ROOT="$CKPT"
echo "[DOWNSTREAM] $(date -Is) Checkpoints validated; executing $ROLE roles"
if test "$ROLE" = gpu1; then
  CUDA_VISIBLE_DEVICES=0 "$PY" "$S/scripts/run_mae_selected_downstreams.py" classification \
    --ckpt-root "$CKPT" --out-root "$RUN/downstream_results" --gpu 0
  CUDA_VISIBLE_DEVICES=0 "$PY" "$S/scripts/run_mae_selected_downstreams.py" arch \
    --ckpt-root "$CKPT" --out-root "$RUN/downstream_results" --gpu 0
else
  # Two independent GPU streams. Failures are retained in separate role logs.
  CUDA_VISIBLE_DEVICES=0 "$PY" "$S/scripts/run_mae_selected_downstreams.py" patch-fewshot \
    --ckpt-root "$CKPT" --out-root "$RUN/downstream_results" --gpu 0 \
    > "$RUN/logs/gpu2_patch_fewshot.log" 2>&1 &
  patch_pid=$!
  set +e
  CUDA_VISIBLE_DEVICES=1 "$PY" "$S/scripts/run_mae_selected_downstreams.py" pathcap \
    --ckpt-root "$CKPT" --out-root "$RUN/downstream_results" --gpu 1 \
    > "$RUN/logs/gpu2_pathcap.log" 2>&1
  pathcap_status=$?
  if test "$pathcap_status" = 0; then
    CUDA_VISIBLE_DEVICES=1 "$PY" "$S/scripts/run_mae_selected_downstreams.py" segmentation \
      --ckpt-root "$CKPT" --out-root "$RUN/downstream_results" --gpu 1 \
      > "$RUN/logs/gpu2_segmentation.log" 2>&1
    segmentation_status=$?
  else
    segmentation_status=2
  fi
  wait "$patch_pid"
  patch_status=$?
  set -e
  printf 'pathcap=%s segmentation=%s patch_fewshot=%s\n' \
    "$pathcap_status" "$segmentation_status" "$patch_status" > "$RUN/logs/gpu2_role_status.txt"
  if test "$pathcap_status" != 0 || test "$segmentation_status" != 0 || test "$patch_status" != 0; then
    echo "One or more GPU2 downstream tasks failed; see role logs"; exit 20
  fi
fi
date -Is > "$RUN/logs/MAE_DOWNSTREAM_COMPLETED.ok"
echo "[DOWNSTREAM] $(date -Is) All approved $ROLE downstream tasks complete"
