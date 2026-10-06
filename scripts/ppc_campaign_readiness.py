"""Readiness gate for the PPC-LeJEPA functional-drift campaign."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from ssl_campaign_inputs import sha, verify
from ssl_campaign_job import atomic

ROOT = Path(__file__).resolve().parents[1]


def source():
    files = {}
    for folder in ("src", "scripts", "configs", "manifests"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in (".py", ".yaml", ".json", ".sha256"):
                files[path.relative_to(ROOT).as_posix()] = sha(path)
    files["pannuke19_metadata.csv"] = sha(ROOT / "pannuke19_metadata.csv")
    return files


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def data_stats(ledger, data):
    stats = []
    for row in ledger["files"]:
        path = data / row["path"]
        stat = path.stat()
        if stat.st_size != row["bytes"]:
            raise RuntimeError("Input size changed: " + row["path"])
        stats.append([row["path"], stat.st_size, stat.st_mtime_ns])
    return fingerprint(stats)


def models():
    root = Path(os.environ["SSL_MODEL_ROOT"]) / "plip_model"
    return {
        name: sha(root / name)
        for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
    }


def record(report):
    import torch
    from pannuke_ssl.ssl_framework.ppc_campaign import plan, preflight

    scratch = Path(os.environ["SSL_PPC_SCRATCH"])
    ledger = json.loads((scratch / "inputs.json").read_text())
    data = Path(os.environ["SSL_DATA_ROOT"])
    checked = verify(ledger, ROOT, data)
    flight = preflight()
    smoke = json.loads((scratch / "smoke" / os.environ["SSL_WORKER_NAME"] / "report.json").read_text())
    p = plan()
    if (
        smoke["status"] != "PASS"
        or smoke["test_touched"]
        or smoke["method"] != "ppc_lejepa"
        or float(smoke["ppc_lambda"]) != float(p["smoke_lambda"])
        or float(smoke["ppc_epsilon"]) != float(p["epsilon"])
        or smoke["initial_projector_sha256"] != p["initial_projector_sha256"]
        or smoke["initial_reference_projector_sha256"] != smoke["initial_projector_sha256"]
        or not smoke["reference_projector_unchanged"]
        or not smoke["reference_projector_frozen"]
        or not smoke["reference_bn_eval"]
        or not smoke["trainable_projector_changed"]
        or not smoke["ppc_projector_gradient_positive"]
        or not smoke["ppc_encoder_gradient_zero"]
        or not smoke["ordinary_lejepa_encoder_gradient_positive"]
    ):
        raise RuntimeError("PPC functional smoke did not establish the required invariants")

    current = source()
    result = {
        "status": "PASS",
        "worker": os.environ["SSL_WORKER_NAME"],
        "source_sha256": fingerprint(current),
        "source_files": current,
        "input_ledger_sha256": fingerprint(ledger),
        "input_verification": checked,
        "data_stat_sha256": data_stats(ledger, data),
        "model_files": models(),
        "smoke": smoke,
        "preflight": flight,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "training_started": False,
    }
    atomic(result, report)
    return result


def check(report):
    result = json.loads(report.read_text())
    scratch = Path(os.environ["SSL_PPC_SCRATCH"])
    ledger = json.loads((scratch / "inputs.json").read_text())
    if result["status"] != "PASS" or result["source_sha256"] != fingerprint(source()):
        raise RuntimeError("Verified PPC source changed; repeat functional smoke")
    if result["input_ledger_sha256"] != fingerprint(ledger):
        raise RuntimeError("PPC input inventory changed")
    if result["data_stat_sha256"] != data_stats(ledger, Path(os.environ["SSL_DATA_ROOT"])):
        raise RuntimeError("PPC data changed; repeat byte verification")
    if result["model_files"] != models():
        raise RuntimeError("PPC model assets changed")

    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA GPU must be visible per PPC worker")
    if sys.version != result["python_version"] or torch.__version__ != result["torch_version"]:
        raise RuntimeError("PPC runtime changed")
    free, _ = torch.cuda.mem_get_info()
    if free < 20 * 2**30:
        raise RuntimeError(f"Insufficient free GPU memory: {free / 2**30:.1f} GiB")
    result["free_gpu_memory_gib"] = free / 2**30
    result.pop("source_files", None)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--mkdir", type=Path)
    parser.add_argument("--json-file", type=Path)
    parser.add_argument("--install-checkpoint", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--sha256")
    args = parser.parse_args()

    if args.mkdir:
        args.mkdir.mkdir(parents=True, exist_ok=True)
        result = {"status": "PASS"}
    elif args.json_file:
        result = json.loads(args.json_file.read_text())
    elif args.install_checkpoint:
        if sha(args.install_checkpoint) != args.sha256:
            raise RuntimeError("Transferred PPC checkpoint checksum mismatch")
        if args.destination.exists() and sha(args.destination) != args.sha256:
            raise RuntimeError("Refuse overwrite of a different PPC checkpoint")
        args.destination.parent.mkdir(parents=True, exist_ok=True)
        args.install_checkpoint.replace(args.destination)
        result = {"status": "PASS", "sha256": args.sha256}
    elif args.record:
        if args.report is None:
            parser.error("--record requires --report")
        result = record(args.report)
    else:
        if args.report is None:
            parser.error("--report is required")
        result = check(args.report)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
