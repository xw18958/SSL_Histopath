# Multi-server experiment workflow

The standard SSL pipeline is server-independent. All experiment code, protocol configs, dataset manifests, and registered experiment definitions are committed to Git. A server supplies only machine-local roots through `.ssl_server.env`.

## One-time server setup

Copy `configs/ssl_standard/server.env.example` to `.ssl_server.env` and set:

- `SSL_DATA_ROOT`: local dataset root
- `SSL_MODEL_ROOT`: local model/config root
- `SSL_RUN_ROOT`: local run-storage root
- `SSL_RESULTS_MASTER`: optional canonical result backup, local or `user@host:/path`

`.ssl_server.env` is ignored by Git. Do not commit server paths.

## Canonical experiments

`configs/ssl_standard/experiment_registry.yaml` contains the run IDs and method-defining overrides. The current primary runs are:

- `ssl-lejepa-s20260903`
- `ssl-simplex-k64-s20260903`

`ssl-simplex-k16-s20260903` is a sensitivity run.

List them with:

```bash
python scripts/run_registered_experiment.py --list
```

## Before a job

Run preflight on the server that will execute it:

```bash
python scripts/preflight_server.py --experiment ssl-lejepa-s20260903 --action pretrain
```

Preflight checks the Git commit/source tree, local roots, backbone, canonical manifests, GPU, output collision, and required checkpoints for evaluation jobs.

## Running

Use the registry wrapper rather than hand-editing configs:

```bash
python scripts/run_registered_experiment.py pretrain --experiment ssl-lejepa-s20260903
```

Outputs use the same relative layout on every server:

```text
$SSL_RUN_ROOT/ssl_standard/runs/<run-id>/
  run_metadata.json
  pretrain_full/
  downstream_datasets/
  image_retrieval_datasets/
```

Each run has an immutable `run_metadata.json` plus one JSON record per invocation under `executions/`. Every pretrain/downstream/retrieval invocation records the Git commit, hostname, runtime roots, seed, method/K/sigma, LR, package versions, and GPU identity, so one run can safely span multiple servers.

## Moving work between servers

Push completed/intermediate run artifacts to the configured results master:

```bash
python scripts/sync_run.py push --run-id ssl-lejepa-s20260903 --execute
```

On another server, pull them:

```bash
python scripts/sync_run.py pull --run-id ssl-lejepa-s20260903 --execute
```

The sync excludes mutable `last.pt` and `best.pt` SSL checkpoints but includes immutable milestone checkpoints (`epoch_100.pt`, `epoch_150.pt`, `epoch_200.pt`, `epoch_250.pt`, `epoch_300.pt`), metadata, logs, metrics, and downstream outputs.

A downstream job therefore needs only the same Git commit, local dataset copies, the committed manifest, and the corresponding milestone checkpoints. It does not need to run on the server that performed SSL pretraining.

## Canonical dataset manifests

Classification manifests live in `manifests/ssl_standard/dataset_manifests/` and are committed to Git. They contain relative records plus portable `${SSL_DATA_ROOT}` root specifications, never server-specific absolute paths. This ensures every server evaluates exactly the same samples.

Do not regenerate a committed manifest during routine experiments. Regeneration is a protocol change and should be reviewed and committed deliberately.
