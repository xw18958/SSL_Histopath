# DINOv3 fair comparison campaign — execution record

**Snapshot:** 28 October 2026, Australia/Sydney.
**Methods:** DINOv3 against fixed LeJEPA, Simplex-LeJEPA K=8 and MAE.
**Source correction:** [PR #10](https://github.com/xw18958/SSL_Histopath/pull/10);
tested source commit `a4397fc2def68b900697422bc7beb45691ccb09a`;
main merge commit `79fb19c5a69e3ddc71b7028a95e7bec55fb314ca`.

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
