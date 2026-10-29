#!/usr/bin/env bash
set -euo pipefail
CODE="$IJEPA_CODE_ROOT"
RUN="$IJEPA_RUN_ROOT"
PY="$IJEPA_PYTHON"
: "$CODE" "$RUN" "$PY"
export SSL_PROJECT_ROOT="$CODE"
export SSL_DATA_ROOT="$IJEPA_DATA_ROOT"
export SSL_MODEL_ROOT="$IJEPA_MODEL_ROOT"
export SSL_RUN_ROOT="$RUN"
export PYTHONPATH="$CODE/src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="$IJEPA_GPU"
TUNE_ID=ssl-ijepa-lr-tuning-20261029
TRAIN_ID=ssl-ijepa-cvpr20261029
TUNE="$RUN/ssl_standard/runs/$TUNE_ID/tuning"
TRAIN="$RUN/ssl_standard/runs/$TRAIN_ID/pretrain_full"
mkdir -p "$RUN/logs"
cd "$CODE"
"$PY" scripts/smoke_ijepa_cvpr_alignment.py
if [ ! -f "$TUNE/best_hyperparameters.yaml" ]; then
  if [ -e "$TUNE/tuning_summary.json" ] || [ -d "$TUNE/stage_1_learning_rate" ]; then
    echo "Incomplete tuning exists; stop instead of mixing partial trials" >&2
    exit 3
  fi
  "$PY" scripts/run_ssl_standard.py tune --method ijepa --run-id "$TUNE_ID"
fi
LR=$("$PY" - "$TUNE" <<'PYINNER'
import json, pathlib, sys, yaml
t=pathlib.Path(sys.argv[1])
s=json.loads((t/"tuning_summary.json").read_text())
b=yaml.safe_load((t/"best_hyperparameters.yaml").read_text())
assert s["method"]=="ijepa" and not s["test_used"]
assert s["search_strategy"]=="single_parameter"
assert s["executed_trial_count"]==3
assert all(x["status"]=="completed" for x in s["rows"])
assert sorted(round(float(x["learning_rate"]),7) for x in s["rows"])==[0.0001,0.0003,0.001]
assert all(x["best_validation_linear_macro_f1"] is not None for x in s["rows"])
assert not b["selection"]["test_used"]
lr=float(b["selected"]["learning_rate"])
assert lr in (0.0001,0.0003,0.001),lr
print(f"{lr:.10g}")
PYINNER
)
printf 'selected_learning_rate: %s
' "$LR" > "$RUN/logs/selected_lr.yaml"
if [ -e "$TRAIN/pretrain_metrics.csv" ] || [ -e "$TRAIN/run_summary.json" ]; then
  echo "I-JEPA pretraining already started; refusing overwrite" >&2
  exit 4
fi
"$PY" scripts/run_ssl_standard.py pretrain --method ijepa --run-id "$TRAIN_ID" --learning-rate "$LR" --max-epochs 250 --checkpoint-epochs 100,150,200,250
"$PY" - "$TRAIN" <<'PYINNER'
import json, pathlib, sys, torch
p=pathlib.Path(sys.argv[1])
r=json.loads((p/"run_summary.json").read_text())
assert r["epochs_completed"]==250 and r["stop_reason"]=="max_epochs",r
assert r["ssl_images_per_epoch"]==7901
assert not r["labels_loaded_for_ssl"] and not r["test_used_for_ssl_selection"]
for e in (100,150,200,250):
    f=p/"checkpoints"/f"epoch_{e}.pt"
    assert f.is_file(),f
    c=torch.load(f,map_location="cpu",weights_only=False)
    assert c["epoch"]==e and c["method_name"]=="ijepa",e
    assert c["config"]["method"]["predictor"]["depth"]==4
print("Verified I-JEPA 100/150/200/250 epoch checkpoints")
PYINNER
date -Is > "$RUN/logs/IJEPA_PRETRAIN_COMPLETED.ok"
