# Simplex K campaign

This campaign adds three independent single-GPU SSL runs, K=8, 16, and 32. Each run and its learning-rate schedule end at epoch **250**, with downstream snapshots at **100, 150, 200, and 250**. LeJEPA/Simplex warmup occupies the first 25 epochs, followed by cosine decay through the final epoch. Every epoch uses the same 7,901 unlabeled PanNuke sources, batch 128, 2 global and 6 local views, BF16/TF32 settings, initialization, LR 0.0005, and sigma 1. Only the target component count differs among the three new runs. Existing checkpoints and result files, including the extra saved checkpoint, remain intact.

Each of the 12 saved checkpoints receives all 15 existing frozen-feature classification protocols and ARCH, IPATH, and PathCap retrieval: **180 classification + 36 retrieval = 216 evaluations**. Classification and retrieval use their existing TRAIN/VAL tuning, final maximum 50 epochs with patience 8, and exclusive TEST markers. Retrieval uses the corrected caption-aware loss and relevance rules. Its sample remains 3,267 TRAIN / 700 VAL / 700 TEST pairs per dataset. Manifests and selected image bytes are identical across servers.

| Worker | First job | Subsequent work |
|---|---|---|
| gpu1-jinman-2, CUDA 0 | K=32 SSL | Shared downstream queue after its own SSL run finishes |
| gpu1-358-0, CUDA 0 | K=16 SSL | Shared downstream queue after its own SSL run finishes |
| gpu2-358-0, CUDA 0 | K=8 SSL | Shared downstream queue after its own SSL run finishes |
| gpu2-358-0, CUDA 1 | Downstream as checkpoints arrive | Shared downstream queue throughout |

Workers use one GPU each. Checkpoints are copied to the persistent results master as they appear, checked by SHA-256, and dispatched to a free worker. Every checkpoint is a bundle of 18 evaluations. A classification encoder is loaded once for its pending datasets. Completed results are skipped only after their checkpoint identity is checked. Per-dataset results and TEST attempt markers are copied to the master during every coordinator poll; completion requires all 216 result files. Failed jobs and partial TEST attempts remain available for diagnosis, and are not automatically retried.

## Prepared deployment

The machine-local inventory is `/raid1/xwan0900/SSL_runs/simplex_k_ablation_20261007/machines.json` on Jinman. The persistent master results root is its containing directory. Machine paths and Python environments are recorded there rather than committed as scientific configuration.

Campaign code, job state, checkpoints, and results use persistent storage on all servers. The third server reads Jinman’s unchanged runtime, datasets, and models through a read-only SSHFS mount at `/home/xwan0900/SSL_remote`, with automatic connection recovery. Its code is under `/home/xwan0900/SSL_campaigns`, and its scientific outputs and job state are under `/home/xwan0900/SSL_runs` and `/home/xwan0900/SSL_job_state`. The old RAM deployment was lost and is no longer used. The launch check refuses missing inputs or stale readiness. The selected inputs total 21,894,848,311 bytes in 43,347 files. The source inventory preserves manifest paths through directory aliases, including IPATH/Images.

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

`launch` runs the coordinator detached and records its PID and log. It does not depend on an open SSH session. A previous campaign start blocks another launch. `status.json`, `coordinator.log`, and each worker job log record progress or failures; final CSV tables and `all_results.json` are written after complete coverage. A transport failure backs off only the affected worker, up to five minutes between connection attempts; reachable workers continue. Scientific job failures remain visible and are never automatically rerun. Job launch acknowledgements are idempotent, and dispatch intent and completed bundle identities are journaled durably. Resume adopts existing jobs and skips completed bundles; `launch-resume` starts that coordinator detached.

Fast verification on 2026-10-07 included 41 tests covering the schedule prefix, stop/checkpoint loop, legacy defaults, the entire four-worker queue, output identity, data aliases, and existing downstream/retrieval behavior. Two real SSL updates at batch 128 passed on each GPU with identical original encoder and projector initialization. Small TRAIN/VAL-only head fits and frozen-component checks passed all 15 classification and three retrieval datasets. Four real checkpoint-transfer roundtrips and detached-job success/failure/exclusivity checks passed. The initial cold third-server smoke hit a pinned-memory transfer error before an update; subsequent checks passed without changing the scientific settings. A missing IPATH directory alias was corrected before the final downstream smoke. No TEST image was decoded in these smokes. The campaign was subsequently launched. On 2026-10-07, K=32 completed all four bundles and K=16 completed SSL; a third-server SSH timeout stopped the original coordinator. Recovery preserves K=32 results and K=16 checkpoints. K=8 is restarted from its original initialization because no state survived the lost RAM deployment. K=16 evaluations interrupted without retained results are rerun; valid completed results are retained.

Recovery verification includes simulated connection loss during queue draining, adoption of existing completed/running jobs, transport-error classification, unchanged scientific loader settings, and a real duplicate detached-job launch that executes its payload only once. Restored third-server inputs are byte verified against the original ledger, and TRAIN/VAL-only GPU smokes cover the original batch size and all downstream datasets. Third-server pinning remains disabled via the recorded machine-local `SSL_PIN_MEMORY=0`; batch size, sampling, data splits, augmentation, optimizer, and 250-epoch schedule are unchanged.

Recovery moves only the lost K=8 run to the now-idle Jinman GPU for faster training and durable local storage. The machine inventory records `pretrain_sources` separately from the original allocation and preserves the actual source of K=16 and K=32 checkpoints. Third-server GPUs finish the pending K=16 bundles and join the shared queue for new K=8 snapshots. A fresh K=8 batch-128 smoke on Jinman passed with the original encoder/projector hashes. Server 3 exposes `org.freedesktop.login1.Manager.RemoveIPC=true`; lingering has been enabled for xwan0900 to retain its user service lifetime. Primary scientific files are outside shared memory regardless of that setting.
