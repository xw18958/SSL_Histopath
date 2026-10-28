# DINOv3 fair comparison campaign — execution record

**Snapshot:** 28 October 2026, Australia/Sydney.
**Methods:** DINOv3 against fixed LeJEPA, Simplex-LeJEPA K=8 and MAE.
**Source correction:** [PR #10](https://github.com/xw18958/SSL_Histopath/pull/10);
tested source commit `a4397fc2def68b900697422bc7beb45691ccb09a`;
main merge commit `79fb19c5a69e3ddc71b7028a95e7bec55fb314ca`.

## Completed outcome and verification

- Sequential 3x20-epoch LR trials completed, selected **0.0001** using PanNuke
  validation Macro-F1 only (29.12% at the best evaluation epoch).
- Full 250-epoch pretraining completed on 7,901 unlabeled PanNuke images.
  Source checkpoints are retained at epochs 100/150/200/250.
- The post-training pipeline finished all **22/22 classification**, **4/4
  ARCH/PathCap image-text retrieval**, **22/22 patch retrieval**, **22/22
  1/5/10-shot few-shot runs**, **6/6 semantic segmentation**, and **2/2 GlaS
  instance segmentation** outputs for the fixed 100/200 checkpoints.
- First downstream launch hit Linux file descriptor / CUDA pinned memory
  resource errors and a missing BreakHis patch manifest. Recovery used the
  same frozen checkpoints and original benchmark manifests, raised the
  file-descriptor limit, disabled pin-memory transfers, and skipped completed
  tests. The interrupted epoch-200 GaHISSDB classification was finished using
  the *previously frozen probe* without retraining or test-based selection
  ([PR #12](https://github.com/xw18958/SSL_Histopath/pull/12)).
- Final cross-method source audit: **PASS 124/124 metric rows**, including
  baseline MAE result SHA, dataset manifests, exact retrieval protocol/split
  hashes, few-shot support subset and row identifiers, segmentation manifest
  hashes, test-isolation flags and GlaS official Test A/B aggregation.
- The full four-method tables, per-task summary, 124-row comparison CSV, source
  SHA-256 audit and aggregation script are committed in
  [benchmark_reports/dinov3_cvpr_20261028](../benchmark_reports/dinov3_cvpr_20261028/README.md)
  by [PR #13](https://github.com/xw18958/SSL_Histopath/pull/13).

The following sections are an **immutable launch-time record**, not the
current run status.

## Validation before launch

- `gpu1-358-0`: 44 DINOv3/MAE/LeJEPA/standard SSL regression tests passed.
- `gpu1-jinman-2`: 12 DINOv3 tests passed.
- Both servers: synthetic **full batch=128** DINOv3 optimizer step passed
  with 2 global and 8 local crops and both 65,536-prototype heads;
  16.513 GiB CUDA peak allocated memory.
- Source code staged from the same tested GitHub revision, without touching
  the existing MAE, LeJEPA, or Simplex output directories.

## Launched jobs and gate dependencies

| Server | Directory | Job | Launch status as of 17:24 Sydney |
|---|---|---|---|
| `gpu1-jinman-2` | `/raid1/xwan0900/dinov3_cvpr_20261028` | 3 sequential 20-epoch LR trials | **Running** |
| `gpu1-358-0` | `/home/xwan0900/dinov3_cvpr_20261028` | 250-epoch full-data pretraining | **Gate waiting on tuning** |
| `gpu1-358-0` | same | Classification + ARCH retrieval | **Gate waiting on pretraining** |
| `gpu2-358-0` | `/home/xwan0900/dinov3_cvpr_20261028` | Patch retrieval + few-shot on GPU 0; PathCap + segmentation on GPU 1 | **Gate waiting on pretraining and verified checkpoint transfer** |

Log files under `logs/`:
`tuning.log`, `tuning.pid`, `pretrain_gate.log`,
`pretrain_gate.pid`, `downstream_gate.log`,
`downstream_gate.pid`. Check live process state and log tail before reporting
any stage as completed. Gate scripts fail closed if upstream jobs disappear
before producing success markers.

The selected LR is sourced from
`tuning/best_hyperparameters.yaml` and is validated against the approved
three candidates. Full SSL training is 250 epochs, all 7,901 images, no early
stopping. Four checkpoints (100/150/200/250) are saved; primary downstream
evaluations use only epochs 100 and 200.

The new GPU2 campaign reuses **unchanged** MAE-validated external
segmentation manifests via a read-only directory symlink, and copies
the existing validated environment/data readiness markers. It writes all
DINOv3 results into separate directories.

## Initial real-data progress

First tuning epoch processed 6,305 PanNuke samples without numerical
failure (loss approximately 16.897, peak GPU memory about 17.22 GiB).
This is a training-integrity observation, **not** a downstream result.

## Final deliverables when the launched jobs finish

Record LR selection, training logs and checkpoint SHA-256 manifest.
Report fixed 100/200-epoch classification, image/text retrieval,
patch retrieval, 1/5/10-shot trainable few-shot and semantic/instance
segmentation metrics using the MAE-matched protocols, then prepare
four-method LeJEPA / Simplex K=8 / MAE / DINOv3 tables.
No evaluation numbers are asserted at campaign launch.
