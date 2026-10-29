#!/usr/bin/env bash
set -euo pipefail
# Raise only the OS descriptor limit; preserve all evaluation hyperparameters.
ulimit -n 65535
S="$IJEPA_CODE_ROOT"
RUN="$IJEPA_RUN_ROOT"
PY="$IJEPA_PYTHON"
ROLE="$IJEPA_ROLE"
SRC=xwan0900@gpu1-jinman-2.it.usyd.edu.au
REMOTE=/raid1/xwan0900/ijepa_cvpr_20261029_runs
CKPT_SRC="$REMOTE/ssl_standard/runs/ssl-ijepa-cvpr20261029/pretrain_full/checkpoints"
CKPT="$RUN/checkpoints"
mkdir -p "$RUN/logs" "$CKPT"
export SSL_PROJECT_ROOT="$S"
export SSL_DATA_ROOT="$IJEPA_DATA_ROOT"
export SSL_MODEL_ROOT="$IJEPA_MODEL_ROOT"
export SSL_RUN_ROOT="$RUN"
export IJEPA_SEG_MANIFEST_ROOT=/home/xwan0900/mae_cvpr_20261027/manifests/segmentation
export IJEPA_CKPT_ROOT="$CKPT"
export SSL_IJEPA_CKPT_ROOT="$CKPT"
export PYTHONPATH="$S/src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
while ! ssh -T -oBatchMode=yes -oConnectTimeout=12 "$SRC" "test -f '$REMOTE/logs/IJEPA_PRETRAIN_COMPLETED.ok'"; do
  if ! ssh -T -oBatchMode=yes -oConnectTimeout=12 "$SRC"     "test -f '$REMOTE/logs/campaign.pid' && kill -0 \$(cat '$REMOTE/logs/campaign.pid') 2>/dev/null"; then
    echo "Upstream I-JEPA campaign not running and not marked complete" >&2
    exit 8
  fi
  sleep 60
done
for epoch in 100 200; do
  srcfile="$CKPT_SRC/epoch_$epoch.pt"
  destfile="$CKPT/epoch_$epoch.pt"
  if [ ! -f "$destfile" ]; then
    rsync -a --partial -e "ssh -oBatchMode=yes" "$SRC:$srcfile" "$destfile"
  fi
  expected=$(ssh -T -oBatchMode=yes "$SRC" "sha256sum '$srcfile'" | awk '{print $1}')
  actual=$(sha256sum "$destfile" | awk '{print $1}')
  [ "$expected" = "$actual" ] || { echo "SHA mismatch epoch=$epoch" >&2; exit 11; }
  printf '%d %s\n' "$epoch" "$actual" >> "$RUN/logs/checkpoints_verified.sha256"
done
for e in 100 200; do
  "$PY" - "$CKPT/epoch_$e.pt" "$e" <<'PYINNER'
import sys,torch
x=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
assert x['framework']=='ssl_standard_v1'
assert x['method_name']=='ijepa'
assert x['epoch']==int(sys.argv[2])
assert x['config']['method']['predictor']['depth']==4
PYINNER
done
if [ "$ROLE" = "gpu1" ]; then
  CUDA_VISIBLE_DEVICES=0 "$PY" "$S/scripts/run_ijepa_selected_downstreams.py" classification     --ckpt-root "$CKPT" --out-root "$RUN/downstream_results" --gpu 0
  CUDA_VISIBLE_DEVICES=0 "$PY" "$S/scripts/run_ijepa_selected_downstreams.py" arch     --ckpt-root "$CKPT" --out-root "$RUN/downstream_results" --gpu 0
elif [ "$ROLE" = "gpu2" ]; then
  CUDA_VISIBLE_DEVICES=0 "$PY" "$S/scripts/run_ijepa_selected_downstreams.py" patch-fewshot     --ckpt-root "$CKPT" --out-root "$RUN/downstream_results" --gpu 0     > "$RUN/logs/gpu2_patch_fewshot.log" 2>&1 &
  patchpid=$!
  set +e
  CUDA_VISIBLE_DEVICES=1 "$PY" "$S/scripts/run_ijepa_selected_downstreams.py" pathcap     --ckpt-root "$CKPT" --out-root "$RUN/downstream_results" --gpu 1     > "$RUN/logs/gpu2_pathcap.log" 2>&1
  stat1=$?
  if [ "$stat1" = 0 ]; then
    CUDA_VISIBLE_DEVICES=1 "$PY" "$S/scripts/run_ijepa_selected_downstreams.py" segmentation       --ckpt-root "$CKPT" --out-root "$RUN/downstream_results" --gpu 1       > "$RUN/logs/gpu2_segmentation.log" 2>&1
    stat2=$?
  else
    stat2=2
  fi
  wait "$patchpid"
  stat3=$?
  set -e
  printf 'pathcap=%s segmentation=%s patch_fewshot=%s\n' "$stat1" "$stat2" "$stat3" > "$RUN/logs/downstream_role_status.txt"
  [ "$stat1" = 0 ] && [ "$stat2" = 0 ] && [ "$stat3" = 0 ] || exit 20
else
  echo "Unsupported role $ROLE" >&2
  exit 2
fi
date -Is > "$RUN/logs/IJEPA_DOWNSTREAM_COMPLETED.ok"
