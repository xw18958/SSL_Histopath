"""Multi-server coordinator for one PPC-LeJEPA pretrain and four downstream bundles.

The coordinator never starts from status/check/prepare. Only the explicit launch
action starts the detached coordinator, and launch is blocked by fresh readiness
reports on every worker.
"""
from __future__ import annotations

import argparse
import base64
import csv
import fcntl
import json
import shlex
import subprocess
import time
from pathlib import Path

import yaml

from ssl_campaign_job import atomic, digest

ROOT = Path(__file__).resolve().parents[1]
SSH = [
    "ssh",
    "-n",
    "-oBatchMode=yes",
    "-oConnectTimeout=15",
    "-oServerAliveInterval=30",
    "-oServerAliveCountMax=3",
]
TRANSFER_SSH = [part for part in SSH if part != "-n"]


def settings(path: Path, pretrain_worker: str):
    machines = json.loads(path.read_text())
    campaign = yaml.safe_load((ROOT / "configs/ssl_standard/ppc_lejepa_campaign.yaml").read_text())
    workers = machines["workers"]
    if pretrain_worker not in workers:
        raise ValueError(f"Unknown PPC pretrain worker {pretrain_worker!r}")
    if not workers:
        raise ValueError("PPC campaign requires at least one worker")
    if (
        [float(x) for x in campaign["lambda_candidates"]] != [0.01, 0.05, 0.10]
        or float(campaign["smoke_lambda"]) != 0.05
        or float(campaign["standard_lejepa_lambda"]) != 0.0
        or float(campaign["epsilon"]) != 1e-8
        or campaign["stop_epoch"] != 250
        or campaign["schedule_epochs"] != 250
        or campaign["checkpoint_epochs"] != [100, 150, 200, 250]
    ):
        raise ValueError("PPC functional-drift campaign settings changed")
    return machines, campaign


def env(worker):
    value = {
        "CUDA_VISIBLE_DEVICES": str(worker["gpu"]),
        "SSL_PROJECT_ROOT": worker["project"],
        "SSL_DATA_ROOT": worker["data"],
        "SSL_MODEL_ROOT": worker["models"],
        "SSL_RUN_ROOT": worker["run_root"],
        "SSL_PPC_RUN_ROOT": worker["run_root"],
        "SSL_PPC_SCRATCH": worker["scratch"],
        "SSL_WORKER_NAME": worker["name"],
        "PYTHONPATH": worker["project"]
        + "/src"
        + (":" + worker["extra_pythonpath"] if worker.get("extra_pythonpath") else ""),
        "OMP_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
    }
    if "pin_memory" in worker:
        value["SSL_PIN_MEMORY"] = str(worker["pin_memory"])
    return value


def command(worker, args):
    return "cd " + shlex.quote(worker["project"]) + " && " + shlex.join(
        ["env", *(key + "=" + value for key, value in env(worker).items()), worker["python"], *args]
    )


def call(worker, args, *, timeout=120):
    cmd = command(worker, args)
    process = subprocess.run(
        ["bash", "-c", cmd] if worker.get("local") else SSH + ["xwan0900@" + worker["host"], cmd],
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if process.returncode:
        raise RuntimeError(
            f"{worker['name']}: {process.stderr[-2000:]} {process.stdout[-2000:]}"
        )
    return json.loads(process.stdout.strip().splitlines()[-1])


def job(worker, action, job_root=None, spec=None):
    args = ["scripts/ssl_campaign_job.py", action]
    if job_root:
        args += ["--root", str(job_root)]
    if spec:
        args += [
            "--spec-base64",
            base64.b64encode(json.dumps(spec).encode()).decode(),
        ]
    return call(worker, args)


def launch_job(worker, action, epoch=None):
    name = action + (f"_e{epoch}" if epoch is not None else "")
    args = [worker["python"], "-u", "scripts/run_ppc_lejepa.py", action]
    if epoch is not None:
        args += ["--epoch", str(epoch)]
    spec = {
        "project": worker["project"],
        "command": args,
        "env": env(worker),
        "job_root": worker["scratch"] + "/jobs/" + name,
    }
    job(worker, "launch", spec=spec)
    return {"root": spec["job_root"], "action": action, "epoch": epoch}


def bundle_queue(campaign):
    return [int(epoch) for epoch in campaign["checkpoint_epochs"]]


def audit(worker):
    return call(
        worker,
        ["scripts/ppc_campaign_readiness.py", "--report", worker["scratch"] + "/readiness.json"],
    )


def check(machines, campaign, pretrain_worker):
    reports = {name: audit(worker) for name, worker in machines["workers"].items()}
    if any(report["status"] != "PASS" for report in reports.values()):
        raise RuntimeError("A PPC worker is not ready")
    if len({report["source_sha256"] for report in reports.values()}) != 1:
        raise RuntimeError("PPC workers have different code")
    if len({report["input_ledger_sha256"] for report in reports.values()}) != 1:
        raise RuntimeError("PPC workers have different input inventories")
    if len({json.dumps(report["model_files"], sort_keys=True) for report in reports.values()}) != 1:
        raise RuntimeError("PPC PLIP assets differ across workers")
    if len(
        {
            (report["python_version"].split()[0], report["torch_version"])
            for report in reports.values()
        }
    ) != 1:
        raise RuntimeError("PPC Python/PyTorch releases differ across workers")

    for name, report in reports.items():
        smoke = report["smoke"]
        if (
            report["worker"] != name
            or smoke["method"] != "ppc_lejepa"
            or float(smoke["ppc_lambda"]) != float(campaign["smoke_lambda"])
            or float(smoke["ppc_epsilon"]) != float(campaign["epsilon"])
            or smoke["initial_encoder_sha256"] != campaign["initial_encoder_sha256"]
            or smoke["initial_projector_sha256"] != campaign["initial_projector_sha256"]
            or smoke["initial_reference_projector_sha256"] != smoke["initial_projector_sha256"]
            or not smoke["reference_projector_unchanged"]
            or not smoke["reference_projector_frozen"]
            or not smoke["reference_bn_eval"]
            or not smoke["ppc_projector_gradient_positive"]
            or not smoke["ppc_encoder_gradient_zero"]
            or not smoke["ordinary_lejepa_encoder_gradient_positive"]
            or smoke["test_touched"]
        ):
            raise RuntimeError(f"Incorrect PPC smoke identity: {name}")

    return {
        "status": "PASS",
        "workers": reports,
        "pretrain_worker": pretrain_worker,
        "training_started": False,
        "tuning_trials": 3,
        "final_ssl_runs": 1,
        "checkpoint_bundles": 4,
        "evaluations": 72,
        "classification": 60,
        "retrieval": 12,
    }


def pull(worker, relative, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = worker["run_root"] + "/" + relative
    if worker.get("local"):
        if Path(source).resolve() != destination.resolve():
            subprocess.run(["rsync", "-a", "--checksum", source, str(destination)], check=True)
    else:
        subprocess.run(
            [
                "rsync",
                "-a",
                "--checksum",
                "-e",
                shlex.join(TRANSFER_SSH),
                "xwan0900@" + worker["host"] + ":" + source,
                str(destination),
            ],
            check=True,
        )


def push_file(worker, source, relative):
    target = worker["run_root"] + "/" + relative
    if worker.get("local") and Path(source).resolve() == Path(target).resolve():
        return
    call(worker, ["scripts/ppc_campaign_readiness.py", "--mkdir", str(Path(target).parent)])
    temporary = target + ".transfer"
    destination = temporary if worker.get("local") else "xwan0900@" + worker["host"] + ":" + temporary
    subprocess.run(
        ["rsync", "-a", "--checksum", "-e", shlex.join(TRANSFER_SSH), str(source), destination],
        check=True,
    )
    call(
        worker,
        [
            "scripts/ppc_campaign_readiness.py",
            "--install-checkpoint",
            temporary,
            "--destination",
            target,
            "--sha256",
            digest(Path(source)),
        ],
    )


def collect(worker, epoch, master, campaign):
    for task, datasets in (
        ("downstream_datasets", campaign["classification_datasets"]),
        ("image_text_retrieval", campaign["retrieval_datasets"]),
    ):
        for dataset in datasets:
            relative = f"ppc_lejepa/{task}/{dataset}/epoch_{epoch}/"
            pull(worker, relative, master / relative)
    relative = f"ppc_lejepa/completion/epoch_{epoch}.json"
    pull(worker, relative, master / relative)
    done = json.loads((master / relative).read_text())
    if (done["epoch"], done["evaluations"]) != (epoch, 18):
        raise RuntimeError("Incomplete PPC downstream bundle")


def verify_complete(master, campaign):
    classifications = []
    retrievals = []
    for epoch in bundle_queue(campaign):
        done = json.loads(
            (master / f"ppc_lejepa/completion/epoch_{epoch}.json").read_text()
        )
        checkpoint = master / f"ppc_lejepa/pretrain_full/checkpoints/epoch_{epoch}.pt"
        checkpoint_hash = digest(checkpoint)
        if done["checkpoint_sha256"] != checkpoint_hash:
            raise RuntimeError("PPC completion checkpoint mismatch")

        for dataset in campaign["classification_datasets"]:
            result = json.loads(
                (
                    master
                    / f"ppc_lejepa/downstream_datasets/{dataset}/epoch_{epoch}/test_metrics.json"
                ).read_text()
            )
            metadata = result["encoder_metadata"]
            if metadata["checkpoint_sha256"] != checkpoint_hash or not result["test_evaluated_once"]:
                raise RuntimeError("PPC classification result identity mismatch")
            classifications.append({"epoch": epoch, "dataset": dataset, **result["test"]})

        for dataset in campaign["retrieval_datasets"]:
            result = json.loads(
                (
                    master
                    / f"ppc_lejepa/image_text_retrieval/{dataset}/epoch_{epoch}/test_retrieval_metrics.json"
                ).read_text()
            )
            if result["checkpoint_sha256"] != checkpoint_hash:
                raise RuntimeError("PPC retrieval result identity mismatch")
            retrievals.append({"epoch": epoch, "dataset": dataset, **result["test"]})

    if (len(classifications), len(retrievals)) != (60, 12):
        raise RuntimeError("Missing PPC results")
    atomic(
        {"classification": classifications, "retrieval": retrievals},
        master / "all_results.json",
    )
    for name, rows in (
        ("classification_results", classifications),
        ("retrieval_results", retrievals),
    ):
        with (master / (name + ".csv")).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def start(machines, campaign, pretrain_worker):
    master = Path(machines["master_root"])
    master.mkdir(parents=True, exist_ok=True)
    with (master / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (master / "campaign_started.json").exists():
            raise RuntimeError("PPC campaign already started; preserve outputs and diagnose before restart")

        ready = check(machines, campaign, pretrain_worker)
        atomic(ready, master / "launch_readiness.json")
        with (master / "campaign_started.json").open("x") as handle:
            json.dump(
                {
                    "started_at": time.time(),
                    "plan": campaign,
                    "pretrain_worker": pretrain_worker,
                },
                handle,
                indent=2,
            )

        workers = machines["workers"]
        source = workers[pretrain_worker]
        running = {pretrain_worker: launch_job(source, "tune")}
        completed = []
        available = set()
        backed = set()
        tuning_finished = False
        pretrain_finished = False

        try:
            while len(completed) != 4 or not pretrain_finished:
                for name, running_job in list(running.items()):
                    worker = workers[name]
                    state = job(worker, "status", running_job["root"])
                    if state["state"] in ("failed", "lost", "missing"):
                        raise RuntimeError(f"{name} {running_job}: {state}")
                    if state["state"] == "complete":
                        if running_job["action"] == "tune":
                            tuning_summary = call(
                                worker,
                                [
                                    "scripts/ppc_campaign_readiness.py",
                                    "--json-file",
                                    worker["run_root"] + "/ppc_lejepa/tuning/tuning_summary.json",
                                ],
                            )
                            if (
                                tuning_summary["search_strategy"] != "sequential_greedy"
                                or tuning_summary["lambda_candidates"] != [0.01, 0.05, 0.10]
                                or tuning_summary["executed_trial_count"] != 3
                                or tuning_summary["test_used"]
                                or float(tuning_summary["selected_value"]) not in (0.01, 0.05, 0.10)
                            ):
                                raise RuntimeError("Wrong PPC sequential-greedy tuning result")
                            for filename in (
                                "tuning_summary.json",
                                "tuning_summary.csv",
                                "best_hyperparameters.yaml",
                            ):
                                pull(
                                    worker,
                                    "ppc_lejepa/tuning/" + filename,
                                    master / "ppc_lejepa/tuning" / filename,
                                )
                            tuning_finished = True
                            running[name] = launch_job(worker, "pretrain")
                            continue
                        if running_job["action"] == "pretrain":
                            summary = call(
                                worker,
                                [
                                    "scripts/ppc_campaign_readiness.py",
                                    "--json-file",
                                    worker["run_root"]
                                    + "/ppc_lejepa/pretrain_full/run_summary.json",
                                ],
                            )
                            if (
                                summary["epochs_completed"] != 250
                                or summary["schedule_epochs"] != 250
                                or summary["saved_checkpoint_epochs"] != [100, 150, 200, 250]
                                or not summary.get("ppc_reference_unchanged")
                                or not summary.get("ppc_reference_frozen")
                                or float(summary.get("ppc_lambda", -1.0)) not in (0.01, 0.05, 0.10)
                            ):
                                raise RuntimeError("Wrong PPC final pretraining metadata")
                            for filename in (
                                "run_summary.json",
                                "resolved_config.json",
                                "pretrain_metrics.csv",
                            ):
                                pull(
                                    worker,
                                    "ppc_lejepa/pretrain_full/" + filename,
                                    master / "ppc_lejepa/pretrain_full" / filename,
                                )
                            pretrain_finished = True
                        else:
                            collect(worker, running_job["epoch"], master, campaign)
                            completed.append(running_job["epoch"])
                        del running[name]

                for epoch in bundle_queue(campaign):
                    if epoch in backed:
                        continue
                    relative = f"ppc_lejepa/pretrain_full/checkpoints/epoch_{epoch}.pt"
                    info = call(
                        source,
                        [
                            "scripts/ssl_campaign_job.py",
                            "inspect",
                            "--file",
                            source["run_root"] + "/" + relative,
                        ],
                    )
                    if info["exists"]:
                        local = master / relative
                        pull(source, relative, local)
                        if digest(local) != info["sha256"]:
                            raise RuntimeError("PPC checkpoint transfer mismatch")
                        backed.add(epoch)
                        available.add(epoch)

                for name, worker in workers.items():
                    if name in running:
                        continue
                    tasks = [
                        epoch
                        for epoch in bundle_queue(campaign)
                        if epoch in available
                        and epoch not in completed
                        and not any(
                            job_info["action"] == "downstream" and job_info["epoch"] == epoch
                            for job_info in running.values()
                        )
                    ]
                    if tasks:
                        epoch = tasks[0]
                        relative = f"ppc_lejepa/pretrain_full/checkpoints/epoch_{epoch}.pt"
                        push_file(worker, master / relative, relative)
                        tuning_relative = "ppc_lejepa/tuning/best_hyperparameters.yaml"
                        push_file(worker, master / tuning_relative, tuning_relative)
                        running[name] = launch_job(worker, "downstream", epoch)

                atomic(
                    {
                        "state": "running",
                        "tuning_finished": tuning_finished,
                        "pretraining_finished": pretrain_finished,
                        "checkpoint_bundles_backed_up": len(backed),
                        "completed_bundles": len(completed),
                        "completed_evaluations": 18 * len(completed),
                        "total_evaluations": 72,
                        "running": running,
                        "updated_at": time.time(),
                    },
                    master / "status.json",
                )
                time.sleep(30)

            verify_complete(master, campaign)
            atomic(
                {
                    "state": "complete",
                    "tuning_trials": 3,
                    "final_ssl_runs": 1,
                    "checkpoint_bundles": 4,
                    "evaluations": 72,
                    "classification": 60,
                    "retrieval": 12,
                    "finished_at": time.time(),
                },
                master / "status.json",
            )
        except Exception as error:
            atomic(
                {
                    "state": "failed",
                    "error": str(error),
                    "running": running,
                    "completed_bundles": completed,
                    "updated_at": time.time(),
                },
                master / "status.json",
            )
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "check", "launch", "start", "status"])
    parser.add_argument("--machines", type=Path, required=True)
    parser.add_argument("--pretrain-worker", default="jinman-g0")
    args = parser.parse_args()
    machines, campaign = settings(args.machines, args.pretrain_worker)

    if args.action == "prepare":
        for worker in machines["workers"].values():
            call(worker, ["scripts/run_ppc_lejepa.py", "prepare"])
        result = {
            "status": "PREPARED",
            "training_started": False,
            "tuning_trials": 3,
            "final_ssl_runs": 1,
            "evaluations": 72,
        }
    elif args.action == "check":
        result = check(machines, campaign, args.pretrain_worker)
        atomic(result, Path(machines["master_root"]) / "ready_to_start.json")
    elif args.action == "status":
        path = Path(machines["master_root"]) / "status.json"
        result = json.loads(path.read_text()) if path.exists() else {"state": "not_started"}
    elif args.action == "launch":
        master = Path(machines["master_root"])
        if (master / "campaign_started.json").exists():
            raise RuntimeError("PPC campaign already started")
        check(machines, campaign, args.pretrain_worker)
        with (master / "launch_requested.json").open("x") as handle:
            json.dump({"requested_at": time.time()}, handle)
        controller = machines["workers"][args.pretrain_worker]
        with (master / "coordinator.log").open("ab") as log:
            child = subprocess.Popen(
                [
                    controller["python"],
                    "-u",
                    str(Path(__file__).resolve()),
                    "start",
                    "--machines",
                    str(args.machines.resolve()),
                    "--pretrain-worker",
                    args.pretrain_worker,
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        result = {
            "state": "launch_requested",
            "pid": child.pid,
            "log": str(master / "coordinator.log"),
        }
        atomic(result, master / "launcher.json")
    else:
        start(machines, campaign, args.pretrain_worker)
        result = {"state": "complete"}

    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
