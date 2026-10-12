#!/usr/bin/env python3
"""Three-server/four-GPU frozen patch benchmark orchestrator.

Runs only after explicit '--run --confirm-full-run'.  Default --plan makes no
changes and never starts the 150-job evaluation.  GPU admission is based solely
on currently free VRAM (not utilization percentage).

Run on the configured primary host with SSH access to the other hosts.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import queue
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from patch_benchmark import (
    EPOCHS, EXTERNAL_DATASETS, METHODS, ROOT, checkpoint_path,
    compare_and_aggregate, load_manifest, sha_file, dump_json,
)

OUT = ROOT / "outputs/patch_benchmark_v1"
MIN_FREE_MIB = 14000
SOURCE_HASH_FILES = (
    "scripts/patch_benchmark.py", "scripts/patch_benchmark_reader.py",
    "src/pannuke_ssl/ssl_framework/external_datasets.py",
    "src/pannuke_ssl/ssl_framework/downstream.py",
    "src/pannuke_ssl/ssl_framework/trainer.py",
    "src/pannuke_ssl/ssl_methods/registry.py",
    "src/pannuke_ssl/ssl_methods/lejepa_standard.py",
    "src/pannuke_ssl/ssl_methods/simplex_sigreg_lejepa_standard.py",
    "configs/ssl_standard/external_probe_datasets.yaml",
)
@dataclass(frozen=True)
class Worker:
    name: str
    host: str
    gpu: int
    repo: str
    python: str
    out: str
    env: dict[str, str]


# Config lives outside the repository; see configs/patch_benchmark/cluster.example.json.
CONFIG_PATH = os.environ.get("PATCH_BENCHMARK_CLUSTER_CONFIG")
if CONFIG_PATH:
    config = json.loads(Path(CONFIG_PATH).expanduser().read_text())
    SSH_USER = config["ssh_user"]
    WORKERS = tuple(Worker(**worker) for worker in config["workers"])
    if len(WORKERS) != 4 or len({w.host for w in WORKERS}) != 3:
        raise ValueError("Expected four GPU workers on three distinct hosts")
    MASTER, GPU1, GPU2 = WORKERS[0].host, WORKERS[1].host, WORKERS[2].host
else:
    WORKERS = ()
    SSH_USER = MASTER = GPU1 = GPU2 = ""

_lock = threading.Lock()


def run(command, *, timeout=1800, check=True):
    p = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    if check and p.returncode:
        raise RuntimeError(f"Failed ({p.returncode}): {command}\nstdout:{p.stdout[-1400:]}\nstderr:{p.stderr[-1400:]}")
    return p


def remote(worker, script, *, timeout=1800, check=True):
    if worker.host == MASTER:
        return run(["bash", "-lc", script], timeout=timeout, check=check)
    return run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
                SSH_USER + "@" + worker.host, script], timeout=timeout, check=check)


def q(s):
    return shlex.quote(str(s))


def exec_prefix(worker):
    vars_ = " ".join(f"{key}={q(val)}" for key, val in worker.env.items())
    return vars_ + " " + q(worker.python) if vars_ else q(worker.python)


def benchmark_command(worker, action, method=None, epoch=None, dataset=None):
    opts = [action, "--out", worker.out]
    if method is not None:
        opts += ["--method", method]
    if epoch is not None:
        opts += ["--epoch", str(epoch)]
    if dataset is not None:
        opts += ["--dataset", dataset]
    if action == "extract-evaluate":
        opts += ["--device", f"cuda:{worker.gpu}", "--variant", "main"]
    if action == "preflight":
        opts += ["--hash-checkpoints"]
    argv = " ".join(q(x) for x in opts)
    return "cd " + q(worker.repo) + " && " + exec_prefix(worker) + " scripts/patch_benchmark.py " + argv


def _rsync_host(worker, src, dst, *, timeout=1800):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if worker.host == MASTER:
        if Path(src).resolve() != dst.resolve():
            raise RuntimeError("Master worker output root mismatch")
        return
    hostsource = f"{SSH_USER}@{worker.host}:{src}"
    run(["rsync", "-a", "--partial", "--", hostsource, str(dst)], timeout=timeout)


def copy_result(worker, method, epoch, dataset):
    relpaths = [
        f"features/full/main/{method}/epoch_{epoch}/{dataset}.npz",
        f"results/full/main/{method}/epoch_{epoch}/{dataset}.json",
    ]
    for path in relpaths:
        _rsync_host(worker, str(Path(worker.out) / path), OUT / path)


def one_task(worker, method, epoch, dataset, attempts=0):
    free = free_vram(worker)
    if free < MIN_FREE_MIB:
        raise RuntimeError(f"{worker.name} has {free}MiB free; requires {MIN_FREE_MIB}MiB")
    path = OUT / f"results/full/main/{method}/epoch_{epoch}/{dataset}.json"
    if path.is_file():
        print(f"SKIP verified existing result: {method}/{epoch}/{dataset}", flush=True)
        return
    task_id = f"{method}_epoch{epoch}_{dataset}"
    log = OUT / "cluster_logs" / worker.name / f"{task_id}.log"
    with _lock:
        print(f"START {worker.name}: {task_id}; VRAM free {free}MiB", flush=True)
    try:
        p = remote(worker, benchmark_command(worker, "extract-evaluate",
                                              method, epoch, dataset), timeout=7200)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(p.stdout + "\n" + p.stderr)
        copy_result(worker, method, epoch, dataset)
        if not path.is_file():
            raise FileNotFoundError(f"Result missing after sync: {path}")
        payload = json.loads(path.read_text())
        if (payload["method"], payload["epoch"], payload["dataset"],
                payload["split_variant"]) != (method, epoch, dataset, "main"):
            raise ValueError("Synced result has incorrect task identity")
        with _lock:
            print(f"OK    {worker.name}: {task_id}", flush=True)
    except Exception as exc:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(str(exc) + "\n")
        raise


def free_vram(worker):
    p = remote(worker,
               "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits",
               timeout=30)
    vals = {}
    for line in p.stdout.splitlines():
        parts = [v.strip() for v in line.split(",")]
        if len(parts) >= 2:
            vals[int(parts[0])] = int(parts[1])
    if worker.gpu not in vals:
        raise RuntimeError("GPU index not present: " + worker.name)
    return vals[worker.gpu]


def ensure_protocol():
    for ds in EXTERNAL_DATASETS:
        load_manifest(ds, "main")
    repaired = load_manifest("breakhis_8subtype", "main")
    if repaired.manifest.get("remaining_duplicate_content_groups_cross_split") != 0:
        raise RuntimeError("Corrected BreaKHis split has not passed pixel dedup audit")
    audits = OUT / "audit.json"
    if not audits.is_file():
        raise FileNotFoundError("Missing mandatory main audit")
    by = {r["dataset"]: r for r in json.loads(audits.read_text())}
    if set(by) != set(EXTERNAL_DATASETS):
        raise RuntimeError("Incomplete 15-dataset main leakage audit")
    for ds in EXTERNAL_DATASETS:
        if by[ds]["split_version"] != "main" or by[ds]["identity_split_overlap_count"]:
            raise RuntimeError("Audit has an incorrect split/identity overlap: " + ds)
    for ds, item in by.items():
        if (item["pixel_duplicate_cross_split_count"] != 0 or
                item.get("image_split_disjoint") is not True or
                item.get("conflicting_label_groups_in_train_or_test") != 0):
            raise RuntimeError("Exact-content overlap scan missing or failed: " + ds)


def prepare_remote():
    """Copy ONLY new benchmark scripts and versioned manifest; no unrelated edits."""
    for host in (GPU1,):
        base = SSH_USER + "@" + host
        run(["ssh", base, "mkdir -p " + q(str(Path(WORKERS[1].repo) / "manifests/patch_benchmark_v1"))])
        srcs = [ROOT / "scripts/patch_benchmark.py",
                ROOT / "scripts/patch_benchmark_breakhis.py",
                ROOT / "scripts/patch_benchmark_filter.py",
                ROOT / "scripts/patch_benchmark_reader.py"]
        run(["scp", "-q", *map(str, srcs),
             base + ":" + str(Path(WORKERS[1].repo) / "scripts") + "/"])
        manifests = sorted((ROOT / "manifests/patch_benchmark_v1").glob("*_image_disjoint_v1.json"))
        if len(manifests) != 3:
            raise RuntimeError("Three corrected image-level manifests required before remote preparation")
        run(["scp", "-q", *map(str, manifests),
             base + ":" + str(Path(WORKERS[1].repo) / "manifests/patch_benchmark_v1") + "/"])


def preflight_hosts():
    refs = {}
    free_by_worker = {worker.name: free_vram(worker) for worker in WORKERS}
    for worker in WORKERS:
        if free_by_worker[worker.name] < MIN_FREE_MIB:
            raise RuntimeError("Insufficient free VRAM on " + worker.name)
    for worker in (WORKERS[0], WORKERS[1], WORKERS[2]):
        free = free_by_worker[worker.name]
        cmd = benchmark_command(worker, "preflight", dataset="all")
        result = remote(worker, cmd, timeout=180)
        obj = json.loads(result.stdout)
        if len(obj["checkpoints"]) != 10 or len(obj["datasets"]) != 15:
            raise RuntimeError("Incomplete checkpoint/dataset preflight")
        source_cmd = "cd " + q(worker.repo) + " && sha256sum " + " ".join(q(x) for x in SOURCE_HASH_FILES)
        source_hashes = [line.split()[0] for line in remote(worker, source_cmd, timeout=90).stdout.splitlines()]
        if len(source_hashes) != len(SOURCE_HASH_FILES):
            raise RuntimeError("Source module hash preflight incomplete: " + worker.host)
        obj["source_module_sha256"] = source_hashes
        refs[worker.host] = obj
        print(f"PRECHECK {worker.host}: ten checkpoints, fifteen datasets, "
              f"GPU free={free}MiB", flush=True)
    baseline = refs[MASTER]
    for host, data in refs.items():
        if data["source_module_sha256"] != baseline["source_module_sha256"]:
            raise RuntimeError("Benchmark source tree differs across servers: " + host)
        for dataset, item in baseline["datasets"].items():
            if data["datasets"][dataset]["sha256"] != item["sha256"]:
                raise RuntimeError("Manifest hash mismatch: " + host + " " + dataset)
        for name, item in baseline["checkpoints"].items():
            if data["checkpoints"][name]["size_bytes"] != item["size_bytes"] or (
                data["checkpoints"][name]["sha256"] != item["sha256"]):
                raise RuntimeError("Checkpoint bytes differ: " + host + " " + name)
    return refs


def execute_full():
    jobs = queue.Queue()
    for epoch in EPOCHS:
        for dataset in EXTERNAL_DATASETS:
            for method in METHODS:
                jobs.put((method, epoch, dataset, 0))
    failures = []

    def worker_loop(worker):
        while True:
            try:
                method, epoch, dataset, attempts = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                one_task(worker, method, epoch, dataset, attempts=attempts)
            except Exception as exc:
                # An error in a single host does not discard any completed
                # task. Retry once on an available worker, then report.
                with _lock:
                    print(f"ERROR {worker.name} {method}/{epoch}/{dataset}: {exc}",
                          flush=True)
                if attempts < 1:
                    jobs.put((method, epoch, dataset, attempts + 1))
                else:
                    with _lock:
                        failures.append((method, epoch, dataset, str(exc)))
            finally:
                jobs.task_done()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(WORKERS)) as pool:
        futures = [pool.submit(worker_loop, worker) for worker in WORKERS]
        for f in futures:
            f.result()
    dump_json(OUT / "cluster_logs/parallel_run_status.json",
              {"failures": failures, "remaining_in_queue": jobs.qsize()})
    if failures or not jobs.empty():
        raise RuntimeError(f"Parallel run incomplete: failures={failures[:5]}")
    compare_and_aggregate(OUT, EXTERNAL_DATASETS, variant="main")
    print("FULL RUN COMPLETE and paired aggregation independently verified.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="Print dry-run work plan only")
    parser.add_argument("--prepare", action="store_true",
                        help="Only deploy dedicated benchmark files to gpu1-358")
    parser.add_argument("--preflight", action="store_true",
                        help="Validate exact inputs and GPU memory; no full run")
    parser.add_argument("--run", action="store_true", help="Execute full parallel evaluation")
    parser.add_argument("--confirm-full-run", action="store_true",
                        help="Explicit guard required with --run")
    args = parser.parse_args()
    if not WORKERS:
        parser.error("Set PATCH_BENCHMARK_CLUSTER_CONFIG to a private cluster JSON file")
    tasks = len(METHODS) * len(EPOCHS) * len(EXTERNAL_DATASETS)
    print(f"BENCHMARK: {tasks} distinct method/checkpoint/dataset jobs on "
          f"{len(WORKERS)} GPUs across 3 servers", flush=True)
    for w in WORKERS:
        print(f"{w.name}: {w.host} GPU:{w.gpu} output:{w.out}", flush=True)
    if args.prepare:
        prepare_remote()
    if args.preflight or args.run:
        ensure_protocol()
        preflight_hosts()
    if args.run:
        if not args.confirm_full_run:
            parser.error("--run requires --confirm-full-run to prevent an accidental full launch")
        execute_full()
    elif not (args.prepare or args.preflight):
        print("PLAN ONLY. No evaluation started.", flush=True)


if __name__ == "__main__":
    main()
