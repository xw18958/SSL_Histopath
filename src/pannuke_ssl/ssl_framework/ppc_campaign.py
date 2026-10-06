"""PPC-LeJEPA functional projector-drift campaign on the frozen 250/250 protocol."""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import torch
import yaml

from pannuke_ssl.config import load_yaml
from pannuke_ssl.utils import atomic_json_dump, seed_everything
from pannuke_ssl.ssl_methods.registry import build_method
from .config import apply_tuned_hyperparameters, load_standard_config, load_tuning_spec
from .data_validation import build_ssl_loader, module_sha
from .downstream import _run_downstream
from .external_datasets import EXTERNAL_DATASETS, load_external_manifest
from .image_text_datasets import IMAGE_TEXT_DATASETS, load_image_text_manifest
from .retrieval_protocol import RETRIEVAL_VERSION, file_sha256
from .retrieval_repair import run_test, run_train_val
from .trainer import _set_schedule, load_checkpoint, train_ssl
from .tuning import run_tuning

ROOT = Path(__file__).resolve().parents[3]


def plan():
    p = load_yaml(ROOT / "configs/ssl_standard/ppc_lejepa_campaign.yaml")
    if [float(x) for x in p["lambda_candidates"]] != [0.01, 0.05, 0.10]:
        raise ValueError("PPC lambda search must be exactly [0.01, 0.05, 0.10]")
    if float(p["smoke_lambda"]) != 0.05 or float(p["standard_lejepa_lambda"]) != 0.0:
        raise ValueError("PPC smoke/default or standard-LeJEPA lambda changed")
    if float(p["epsilon"]) != 1e-8:
        raise ValueError("PPC epsilon must remain 1e-8")
    if (p["stop_epoch"], p["schedule_epochs"]) != (250, 250):
        raise ValueError("PPC must use 250 training epochs and a 250-epoch LR schedule")
    if p["checkpoint_epochs"] != [100, 150, 200, 250]:
        raise ValueError("PPC checkpoints must be 100/150/200/250")
    if p["ssl_batch_size"] != 128 or float(p["ssl_learning_rate"]) != 0.0005:
        raise ValueError("PPC frozen SSL optimization settings changed")
    if tuple(p["classification_datasets"]) != EXTERNAL_DATASETS:
        raise ValueError("PPC campaign must cover all frozen classification datasets")
    if tuple(p["retrieval_datasets"]) != IMAGE_TEXT_DATASETS:
        raise ValueError("PPC campaign must cover all frozen retrieval datasets")
    if p["seed"] != 20260903:
        raise ValueError("PPC initialization seed changed")
    spec = load_tuning_spec("ppc_lejepa")
    if spec["search"]["strategy"] != "sequential_greedy" or spec["search"]["order"] != ["ppc_lambda"]:
        raise ValueError("PPC tuning must use the repository sequential-greedy protocol")
    return p


def config(*, ppc_lambda: float | None = None):
    p = plan()
    c = load_standard_config("ppc_lejepa")
    c["training"].update(
        max_epochs=p["stop_epoch"],
        schedule_epochs=p["schedule_epochs"],
        checkpoint_epochs=list(p["checkpoint_epochs"]),
    )
    c["method"]["optimizer"]["peak_lr"] = float(p["ssl_learning_rate"])
    if ppc_lambda is not None:
        c = apply_tuned_hyperparameters(c, {"ppc_lambda": float(ppc_lambda)})
    c["campaign"] = {
        "name": p["campaign"],
        "lambda_candidates": list(p["lambda_candidates"]),
        "expected_initial_encoder_sha256": p["initial_encoder_sha256"],
        "expected_initial_projector_sha256": p["initial_projector_sha256"],
    }
    return c


def matrix():
    p = plan()
    return [
        {"epoch": epoch, "task": task, "dataset": dataset}
        for epoch in p["checkpoint_epochs"]
        for task, datasets in (
            ("classification", p["classification_datasets"]),
            ("retrieval", p["retrieval_datasets"]),
        )
        for dataset in datasets
    ]


def run_root():
    return Path(os.environ["SSL_PPC_RUN_ROOT"]) / "ppc_lejepa"


def tuning_file():
    return run_root() / "tuning" / "best_hyperparameters.yaml"


def selected_lambda():
    path = tuning_file()
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing PPC sequential-greedy tuning result: {path}. "
            "Run PPC tuning before final pretraining."
        )
    payload = yaml.safe_load(path.read_text())
    selected = payload.get("selected", {})
    value = float(selected["ppc_lambda"])
    if value not in (0.01, 0.05, 0.10):
        raise RuntimeError(f"Unexpected selected PPC lambda: {value}")
    if payload.get("selection", {}).get("test_used") is not False:
        raise RuntimeError("PPC lambda selection must not use TEST")
    return value


def final_config():
    return config(ppc_lambda=selected_lambda())


def preflight():
    c = config(ppc_lambda=plan()["smoke_lambda"])
    p = plan()
    manifest_root = Path(c["manifests"]["root"])
    from .data_validation import validated_pannuke_split

    rows, index = validated_pannuke_split(c)
    if len(rows) != 7901 or len(index) != 7901:
        raise AssertionError("PPC final pretraining must resolve all 7901 PanNuke images")

    manifests = {}
    for dataset in p["classification_datasets"]:
        manifest = load_external_manifest(dataset, manifest_root)
        manifests[dataset] = manifest.manifest_sha256
    for dataset in p["retrieval_datasets"]:
        manifest = load_image_text_manifest(dataset, manifest_root)
        if manifest["split_counts"] != {"train": 3267, "val": 700, "test": 700}:
            raise AssertionError(f"Unexpected retrieval split for {dataset}")
        manifests[dataset] = manifest["manifest_sha256"]

    return {
        "status": "PASS",
        "method": "ppc_lejepa",
        "ppc_lambda_candidates": list(p["lambda_candidates"]),
        "ppc_smoke_lambda": float(p["smoke_lambda"]),
        "ppc_epsilon": float(p["epsilon"]),
        "tuning_strategy": "sequential_greedy",
        "tuning_order": ["ppc_lambda"],
        "tuning_uses_external_test": False,
        "pannuke_images": 7901,
        "training_epochs": 250,
        "schedule_epochs": 250,
        "checkpoint_epochs": [100, 150, 200, 250],
        "manifest_hashes": manifests,
        "evaluations": len(matrix()),
        "test_decoded": False,
    }


def prepare():
    c = config(ppc_lambda=plan()["smoke_lambda"])
    root = run_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "campaign_config.json"
    if path.exists() and json.loads(path.read_text()) != c:
        raise RuntimeError("Existing PPC campaign configuration differs")
    atomic_json_dump(c, path)
    atomic_json_dump({"plan": plan(), "evaluations": matrix()}, root.parent / "campaign_plan.json")
    return {"status": "PREPARED", "root": str(root), "training_started": False, "evaluations": 72}


def tune():
    c = config(ppc_lambda=plan()["smoke_lambda"])
    c.setdefault("runtime", {})["run_root"] = str(run_root())
    result = run_tuning(c)
    if result["search_strategy"] != "sequential_greedy":
        raise RuntimeError("PPC tuning did not use sequential_greedy")
    if result["lambda_candidates"] != [0.01, 0.05, 0.10] or result["test_used"]:
        raise RuntimeError("PPC tuning protocol drifted")
    return result


def _initial_hashes(c):
    seed_everything(int(c["seed"]))
    method = build_method(c, torch.device("cpu"))
    result = {
        "encoder": module_sha(method.encoder),
        "projector": module_sha(method.model.projector),
        "reference_projector": module_sha(method.reference_projector),
    }
    del method
    return result


def pretrain():
    c = final_config()
    expected = plan()
    hashes = _initial_hashes(c)
    if hashes["encoder"] != expected["initial_encoder_sha256"]:
        raise RuntimeError("PPC initial encoder initialization mismatch")
    if hashes["projector"] != expected["initial_projector_sha256"]:
        raise RuntimeError("PPC initial projector initialization mismatch")
    if hashes["reference_projector"] != hashes["projector"]:
        raise RuntimeError("PPC reference projector is not the true initialization snapshot")
    return train_ssl(
        c,
        run_root() / "pretrain_full",
        use_all_data=True,
        validate=False,
        early_stop=False,
        checkpoint_epochs=c["training"]["checkpoint_epochs"],
        schedule_epochs=c["training"]["schedule_epochs"],
    )


def evaluate(epoch):
    p = plan()
    epoch = int(epoch)
    if epoch not in p["checkpoint_epochs"]:
        raise ValueError("Unplanned PPC checkpoint")

    c = final_config()
    root = run_root()
    checkpoint = root / "pretrain_full/checkpoints" / f"epoch_{epoch}.pt"
    checkpoint_hash = file_sha256(checkpoint)
    selected = selected_lambda()
    metadata = {
        "method": "ppc_lejepa",
        "ppc_lambda": selected,
        "ppc_epsilon": 1e-8,
        "checkpoint_sha256": checkpoint_hash,
        "schedule_epochs": 250,
    }
    pending = []
    results = []

    for dataset in p["classification_datasets"]:
        out = root / "downstream_datasets" / dataset / f"epoch_{epoch}"
        done = out / "test_metrics.json"
        if done.exists():
            result = json.loads(done.read_text())
            if result.get("encoder_metadata") != metadata or result["encoder_epoch"] != epoch:
                raise RuntimeError("Completed PPC classification identity mismatch")
            results.append(result)
        else:
            pending.append((dataset, out))

    if pending:
        seed_everything(int(c["seed"]))
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        method = build_method(c, torch.device("cuda"))
        state = load_checkpoint(method, checkpoint, torch.device("cuda"))
        checkpoint_meta = state.get("method_metadata", {})
        if state["epoch"] != epoch or state["method_name"] != "ppc_lejepa":
            raise RuntimeError("PPC checkpoint method/epoch mismatch")
        if float(checkpoint_meta.get("ppc_lambda", -1.0)) != selected:
            raise RuntimeError("PPC checkpoint lambda metadata mismatch")
        if not checkpoint_meta.get("ppc_reference_unchanged"):
            raise RuntimeError("PPC frozen reference changed in checkpoint")
        before = module_sha(method.encoder)
        for dataset, out in pending:
            results.append(
                _run_downstream(
                    c,
                    method.encoder,
                    out,
                    encoder_epoch=epoch,
                    encoder_metadata=metadata,
                    external_dataset=load_external_manifest(dataset, Path(c["manifests"]["root"])),
                )
            )
        if module_sha(method.encoder) != before:
            raise AssertionError("PPC classification changed frozen encoder weights")
        del method, state
        torch.cuda.empty_cache()

    for dataset in p["retrieval_datasets"]:
        out = root / "image_text_retrieval" / dataset / f"epoch_{epoch}"
        done = out / "test_retrieval_metrics.json"
        if done.exists():
            result = json.loads(done.read_text())
            if (
                result["checkpoint_sha256"] != checkpoint_hash
                or result["encoder_epoch"] != epoch
                or result["evaluation_protocol_version"] != RETRIEVAL_VERSION
            ):
                raise RuntimeError("Completed PPC retrieval identity mismatch")
        else:
            cache = Path(os.environ.get("SSL_PPC_SCRATCH", "/tmp/ssl_ppc_lejepa")) / "feature_work"
            run_train_val(c, checkpoint, out, dataset=dataset, cache_root=cache)
            result = run_test(c, checkpoint, out, dataset=dataset, cache_root=cache)
        results.append(result)

    if len(results) != 18:
        raise AssertionError("Each PPC checkpoint must produce 18 downstream evaluations")
    completion = {
        "method": "ppc_lejepa",
        "ppc_lambda": selected,
        "epoch": epoch,
        "evaluations": 18,
        "classification": 15,
        "retrieval": 3,
        "checkpoint_sha256": checkpoint_hash,
    }
    atomic_json_dump(completion, root / "completion" / f"epoch_{epoch}.json")
    return {"status": "PASS", **completion}


def functional_smoke():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("PPC functional smoke requires exactly one visible CUDA GPU")

    p = plan()
    c = config(ppc_lambda=p["smoke_lambda"])
    seed_everything(int(c["seed"]))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = bool(c["training"]["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(c["training"]["tf32"])
    device = torch.device("cuda")
    method = build_method(c, device)

    initial_encoder = module_sha(method.encoder)
    initial_projector = module_sha(method.model.projector)
    initial_reference = module_sha(method.reference_projector)
    if initial_encoder != p["initial_encoder_sha256"]:
        raise AssertionError("PPC smoke initial encoder mismatch")
    if initial_projector != p["initial_projector_sha256"]:
        raise AssertionError("PPC smoke initial projector mismatch")
    if initial_reference != initial_projector:
        raise AssertionError("PPC reference projector does not match g_phi at initialization")
    if any(parameter.requires_grad for parameter in method.reference_projector.parameters()):
        raise AssertionError("PPC reference projector is not frozen")
    if method.reference_projector.training:
        raise AssertionError("PPC reference projector must remain in eval mode")

    loader = build_ssl_loader(c, method, use_all_data=True)
    optimizer = method.build_optimizer()
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if any(id(parameter) in optimizer_ids for parameter in method.reference_projector.parameters()):
        raise AssertionError("PPC reference projector entered the optimizer")

    batch = next(iter(loader))
    g = batch["global_views"].to(device, non_blocking=True)
    l = batch["local_views"].to(device, non_blocking=True)
    gs = [g[:, i] for i in range(g.shape[1])]
    ls = [l[:, i] for i in range(l.shape[1])]
    if len(gs) != 2 or len(ls) != 6 or g.shape[0] != 128:
        raise AssertionError("PPC smoke must preserve batch128 and 2G+6L")

    method.train_mode()
    reference_bn_before = {
        name: (module.running_mean.detach().clone(), module.running_var.detach().clone(), module.num_batches_tracked.detach().clone())
        for name, module in method.reference_projector.named_modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    }

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        global_features = method.model._features(torch.cat(gs, dim=0))
        local_features = method.model._features(torch.cat(ls, dim=0))
        all_features = torch.cat((global_features, local_features), dim=0)
        initial_regularizer = method.ppc_regularizer_from_features(all_features)
    if float(initial_regularizer.detach()) > 1e-10:
        raise AssertionError(f"PPC regularizer must start near zero, got {float(initial_regularizer.detach())}")

    optimizer.zero_grad(set_to_none=True)
    result = method.training_step(batch, bf16=True)
    if not torch.isfinite(result.loss):
        raise FloatingPointError("Non-finite PPC smoke total loss")
    result.loss.backward()
    encoder_grad = torch.sqrt(
        sum(
            parameter.grad.detach().float().square().sum()
            for parameter in method.encoder.parameters()
            if parameter.grad is not None
        )
    )
    if not torch.isfinite(encoder_grad) or float(encoder_grad) <= 0.0:
        raise AssertionError("Ordinary LeJEPA loss did not update the encoder")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if module_sha(method.reference_projector) != initial_reference:
        raise AssertionError("Frozen PPC reference projector changed after optimizer step")
    if module_sha(method.model.projector) == initial_projector:
        raise AssertionError("Trainable PPC projector did not change after optimizer step")

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        global_features = method.model._features(torch.cat(gs, dim=0))
        local_features = method.model._features(torch.cat(ls, dim=0))
        all_features = torch.cat((global_features, local_features), dim=0)
        regularizer = method.ppc_regularizer_from_features(all_features)
    regularizer.backward()
    projector_grad_sq = sum(
        parameter.grad.detach().float().square().sum()
        for parameter in method.model.projector.parameters()
        if parameter.grad is not None
    )
    encoder_regularizer_grads = [
        parameter.grad for parameter in method.encoder.parameters() if parameter.grad is not None
    ]
    if float(projector_grad_sq) <= 0.0:
        raise AssertionError("PPC regularizer did not produce projector gradients")
    if encoder_regularizer_grads and any(float(grad.detach().abs().max()) != 0.0 for grad in encoder_regularizer_grads):
        raise AssertionError("PPC regularizer leaked gradient into encoder")

    for name, module in method.reference_projector.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            mean, var, tracked = reference_bn_before[name]
            if (
                not torch.equal(module.running_mean, mean)
                or not torch.equal(module.running_var, var)
                or not torch.equal(module.num_batches_tracked, tracked)
            ):
                raise AssertionError("PPC reference BatchNorm statistics changed")

    metadata = method.checkpoint_metadata()
    scratch = Path(os.environ.get("SSL_PPC_SCRATCH", "/tmp/ssl_ppc_lejepa"))
    report_path = scratch / "smoke" / os.environ.get("SSL_WORKER_NAME", "worker") / "report.json"
    report = {
        "status": "PASS",
        "method": "ppc_lejepa",
        "ppc_lambda": float(p["smoke_lambda"]),
        "ppc_epsilon": float(p["epsilon"]),
        "initial_encoder_sha256": initial_encoder,
        "initial_projector_sha256": initial_projector,
        "initial_reference_projector_sha256": initial_reference,
        "initial_ppc_regularizer": float(initial_regularizer.detach()),
        "post_update_ppc_regularizer": float(regularizer.detach()),
        "reference_projector_unchanged": metadata["ppc_reference_unchanged"],
        "reference_projector_frozen": metadata["ppc_reference_frozen"],
        "reference_bn_eval": metadata["ppc_reference_bn_eval"],
        "trainable_projector_changed": metadata["ppc_trainable_changed_from_initial"],
        "ppc_projector_gradient_positive": True,
        "ppc_encoder_gradient_zero": True,
        "ordinary_lejepa_encoder_gradient_positive": True,
        "test_touched": False,
    }
    atomic_json_dump(report, report_path)
    return report
