# Simplex K campaign: ready, not started

This campaign adds three independent single-GPU SSL runs, K=8, 16, and 32. Each run and its learning-rate schedule end at epoch **250**, with downstream snapshots at **100, 150, 200, and 250**. LeJEPA/Simplex warmup occupies the first 25 epochs, followed by cosine decay through the final epoch. Every epoch uses the same 7,901 unlabeled PanNuke sources, batch 128, 2 global and 6 local views, BF16/TF32 settings, initialization, LR 0.0005, and sigma 1. Only the target component count differs among the three new runs. Existing checkpoints and result files, including the extra saved checkpoint, remain intact.

Each of the 12 saved checkpoints receives all 15 existing frozen-feature classification protocols and ARCH, IPATH, and PathCap retrieval: **180 classification + 36 retrieval = 216 evaluations**. Classification and retrieval use their existing TRAIN/VAL tuning, final maximum 50 epochs with patience 8, and exclusive TEST markers. Retrieval uses the corrected caption-aware loss and relevance rules. Its sample remains 3,267 TRAIN / 700 VAL / 700 TEST pairs per dataset. Manifests and selected image bytes are identical across servers.

| Worker | First job | Subsequent work |
|---|---|---|
| gpu1-jinman-2, CUDA 0 | K=32 SSL | Shared downstream queue after its own SSL run finishes |
| gpu1-358-0, CUDA 0 | K=16 SSL | Shared downstream queue after its own SSL run finishes |
| gpu2-358-0, CUDA 1 | K=8 SSL | Shared downstream queue after its own SSL run finishes |
| gpu2-358-0, CUDA 0 | Downstream as checkpoints arrive | Shared downstream queue throughout |

Workers use one GPU each. Checkpoints are copied to the persistent results master as they appear, checked by SHA-256, and dispatched to a free worker. Every checkpoint is a bundle of 18 evaluations. A classification encoder is loaded once for its pending datasets. Completed results are skipped only after their checkpoint identity is checked. Results are copied to the master after every bundle; completion requires all 216 result files. Failed jobs and partial TEST attempts remain available for diagnosis, and are not automatically retried.

## Prepared deployment

The machine-local inventory is `/raid1/xwan0900/SSL_runs/simplex_k_ablation_20261007/machines.json` on Jinman. The persistent master results root is its containing directory. Machine paths and Python environments are recorded there rather than committed as scientific configuration.

Campaign code is isolated from the existing checkouts and completed results. Jinman and the second server have persistent campaign checkouts; the third server currently uses the temporary runtime/data/code under `/dev/shm/xwan0900_ssl_pathcap_20261007`. This temporary copy disappears on reboot. The launch check refuses missing inputs or stale readiness. Preserve the master inventory, restore those inputs from Jinman if necessary, and rerun fast smokes and byte verification before launch. The selected inputs total 21,894,848,311 bytes in 43,347 files. The source inventory preserves manifest paths through directory aliases, including IPATH/Images.

Existing K=64/LeJEPA SSL, correct classification results, and corrected retrieval results are retained. No original experiment is rerun or overwritten. Already completed PathCap results are committed separately under `outputs/ssl_standard/pathcap_retrieval_20261007`; those are existing results, not new campaign outputs. Temporary smoke checkpoints, logs, runtime copies, and feature working files are outside the repository.

## Checks and commands

`scripts/run_simplex_k_ablation.py` supports explicit `prepare`, `preflight`, `smoke`, `pretrain`, and `downstream` actions. Set the machine environment from the inventory before using an individual worker. The campaign definition is `configs/ssl_standard/simplex_k_ablation.yaml`.

On Jinman, inspect readiness without starting any job:

```bash
cd /raid1/xwan0900/SSL_campaigns/simplex_k_ablation_20261007_code
/raid1/xwan0900/venvs/ftkp_cu128/bin/python scripts/run_simplex_k_cluster.py check --machines /raid1/xwan0900/SSL_runs/simplex_k_ablation_20261007/machines.json
/raid1/xwan0900/venvs/ftkp_cu128/bin/python scripts/run_simplex_k_cluster.py status --machines /raid1/xwan0900/SSL_runs/simplex_k_ablation_20261007/machines.json
```

Only an explicitly requested future launch should use:

```bash
/raid1/xwan0900/venvs/ftkp_cu128/bin/python scripts/run_simplex_k_cluster.py launch --machines /raid1/xwan0900/SSL_runs/simplex_k_ablation_20261007/machines.json
```

`launch` runs the coordinator detached and records its PID and log. It does not depend on an open SSH session. A previous campaign start blocks another launch. `status.json`, `coordinator.log`, and each worker job log record progress or failures; final CSV tables and `all_results.json` are written after complete coverage. A failure stops new dispatches while preserving already running jobs and outputs.

Fast verification on 2026-10-07 included 41 tests covering the schedule prefix, stop/checkpoint loop, legacy defaults, the entire four-worker queue, output identity, data aliases, and existing downstream/retrieval behavior. Two real SSL updates at batch 128 passed on each GPU with identical original encoder and projector initialization. Small TRAIN/VAL-only head fits and frozen-component checks passed all 15 classification and three retrieval datasets. Four real checkpoint-transfer roundtrips and detached-job success/failure/exclusivity checks passed. The initial cold third-server smoke hit a pinned-memory transfer error before an update; subsequent checks passed without changing the scientific settings. A missing IPATH directory alias was corrected before the final downstream smoke. No TEST image was decoded in these smokes, and full training remains unstarted.
