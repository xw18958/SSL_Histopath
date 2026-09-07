"""Isolated validation-selected duration run; B0 implementations are read-only."""
from __future__ import annotations
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import build_ssl_loader, PanNukeImageDataset, loader_kwargs
from .parquet import build_source_index, preload_images, read_metadata, verify_records
from .monitor import SSLRepresentationMonitor
from .ijepa import BlockMasks, build_models, objective
from .models import update_ema
from .training import _make_optimizer, _learning_rate, _weight_decay, _cosine, _save_final_student
from .utils import seed_everything, atomic_json_dump, write_csv, plot_history


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_manifest(root):
    root = Path(root)
    paths = [root / "pannuke19_metadata.csv"]
    for directory in ("src/pannuke_ssl", "configs", "scripts", "outputs"):
        for p in (root / directory).rglob("*"):
            rel = str(p.relative_to(root))
            if p.is_file() and "__pycache__" not in rel and not p.name.startswith("._"):
                if "ijepa" not in rel.lower() and (directory != "outputs" or "b0" in rel.lower()):
                    paths.append(p)
    return {str(p.relative_to(root)): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns,
                                       "sha256": sha(p)} for p in sorted(set(paths))}


def verified_metadata(config):
    rows = read_metadata(Path(config["monitor"]["metadata_csv"]))
    assert dict(Counter(r["split"] for r in rows)) == {"train":2052, "val":247, "test":247}
    assert len({(r["fold"], r["sample_index"]) for r in rows}) == 2546
    counts = Counter((r["split"], r["class_id"]) for r in rows)
    for split, n in (("train",108), ("val",13), ("test",13)):
        assert [counts[split, i] for i in range(19)] == [n]*19
    return rows


class StrictMonitor(SSLRepresentationMonitor):
    def __init__(self, config, device):
        self.config = dict(config["monitor"])
        self.output_dir = Path(config["output_dir"])
        self.device, self.seed, self.history = device, config["seed"], []
        rows = verified_metadata(config)
        selected = [r for r in rows if r["split"] in ("train", "val")]
        index = build_source_index(Path(config["data_root"]))
        verify_records(rows, index)
        cache = preload_images(selected, index)
        assert set(cache) == {(r["fold"], r["sample_index"]) for r in selected}
        self.loaders = {s: DataLoader(PanNukeImageDataset([r for r in selected if r["split"]==s],
                                                       index, cache, include_label=True),
                                     **loader_kwargs(self.config["batch_size"], self.config["num_workers"],
                                                     shuffle=(s=="train"))) for s in ("train","val")}


def validate_config(c):
    t = c["train"]
    assert c["seed"] == 20260903
    assert (t["epochs"],t["batch_size"],t["warmup_fraction"]) == (300,128,.1)
    assert (t["learning_rate"],t["minimum_learning_rate"]) == (1e-4,1e-6)
    assert (t["weight_decay"],t["final_weight_decay"],t["ema_start"]) == (.04,.40,.996)
    assert t["bf16"] and not t.get("resume")
    assert c["model"] == {"image_size":256,"patch_size":32,"hidden_size":768,
                          "predictor_dim":384,"predictor_depth":2,"predictor_heads":6,
                          "predictor_mlp_ratio":4,"dropout":0.0}


def run(config):
    validate_config(config)
    output = Path(config["output_dir"])
    assert output.name == "ijepa_duration_pilot"
    smoke = json.loads((output / "smoke_test.json").read_text())
    assert smoke["passed"]
    assert not (output / "pretrain_metrics.csv").exists(), "Refuse to overwrite/restart a duration run"
    protected = json.loads((output / "protected_manifest.json").read_text())
    assert protected_manifest(output.parents[1]) == protected
    seed_everything(config["seed"])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device, train = torch.device("cuda"), config["train"]
    atomic_json_dump(config, output / "resolved_config.json")
    loader, _ = build_ssl_loader(config["data_root"], batch_size=128, num_workers=train["num_workers"], cache_in_ram=True)
    assert len(loader.dataset) == 7901 and not loader.dataset.include_label
    student, teacher, predictor = build_models(config, device)
    trainable = list(student.parameters()) + list(predictor.parameters())
    optimizer = _make_optimizer(trainable, 1e-4, .04)
    masks = BlockMasks(config["seed"])
    monitor = StrictMonitor(config, device)
    total_steps, step, history = 300 * len(loader), 0, []
    best_score, best_epoch = float("-inf"), 0
    started = time.perf_counter()

    def evaluate(epoch):
        nonlocal best_score, best_epoch
        row = monitor.evaluate(student, epoch)
        score = row["linear_val_macro_f1"]
        if score > best_score and score >= best_score + .005:
            best_score, best_epoch = score, epoch
            _save_final_student(output / "checkpoints/best.pt", student=student, config=config,
                                epoch=epoch, history=history,
                                selection={"metric":"validation_linear_macro_f1", "value":score,
                                           "minimum_delta":.005,"selection_split":"validation","transductive_ssl":True})
        print(json.dumps({"monitor":row,"best_epoch":best_epoch,"best_score":best_score}), flush=True)

    evaluate(0)
    for epoch in range(1,301):
        student.train(); predictor.train(); teacher.eval()
        torch.cuda.reset_peak_memory_stats()
        epoch_start, loss_sum, seen, context_sum, target_sum = time.perf_counter(), 0., 0, 0, 0
        for uint8_images in loader:
            images = uint8_images.to(device, dtype=torch.float32, non_blocking=True).div_(255.)
            context, targets = masks.sample(images.shape[0], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _, _ = objective(student, teacher, predictor, images, context, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, step {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, train["gradient_clip_norm"])
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradients")
            for group in optimizer.param_groups:
                group["lr"] = _learning_rate(train, step, total_steps)
                group["weight_decay"] = _weight_decay(train, step, total_steps)
            optimizer.step()
            momentum = _cosine(.996,1.,step,total_steps)
            update_ema(student, teacher, momentum)
            step += 1
            n = images.shape[0]
            loss_sum += float(loss.detach()) * n
            context_sum += context.shape[1] * n
            target_sum += targets.shape[2] * n
            seen += n
        seconds = time.perf_counter()-epoch_start
        row = {"epoch":epoch,"prediction":loss_sum/seen,"learning_rate":optimizer.param_groups[0]["lr"],
               "weight_decay":optimizer.param_groups[0]["weight_decay"],"ema_momentum":momentum,
               "seconds":seconds,"samples":seen,"samples_per_second":seen/seconds,
               "mean_context_tokens":context_sum/seen,"mean_tokens_per_target":target_sum/seen,
               "peak_gpu_memory_gib":torch.cuda.max_memory_allocated()/2**30}
        assert seen == 7901
        history.append(row)
        write_csv(history,output / "pretrain_metrics.csv")
        print(json.dumps(row),flush=True)
        if epoch % 10 == 0:
            evaluate(epoch)
    plot_history(history,["prediction"],output/"pretrain_losses.png","I-JEPA pretraining loss")
    plot_history(monitor.history,["linear_val_macro_f1","knn_val_macro_f1"],
                 output/"representation_validation_curves.png","I-JEPA validation-only monitoring")
    plot_history(monitor.history,["feature_embedding_std","feature_effective_rank_fraction",
                                  "feature_mean_pairwise_cosine","feature_near_constant_fraction"],
                 output/"representation_health.png","I-JEPA representation health")
    raw_best = max(monitor.history,key=lambda r:(r["linear_val_macro_f1"],-r["epoch"]))
    selection = {"selected_epoch":best_epoch,"selection_score":best_score,"horizon_epochs":300,
                 "warmup_epochs":30,"minimum_delta_macro_f1":.005,"test_split_used_for_selection":False,
                 "raw_best_monitor_epoch":raw_best["epoch"],"raw_best_monitor_score":raw_best["linear_val_macro_f1"],
                 "transductive_ssl":True,"checkpoint_sha256":sha(output/"checkpoints/best.pt")}
    atomic_json_dump(selection,output/"duration_selection.json")
    assert protected_manifest(output.parents[1]) == protected
    result = {"epochs":300,"best_epoch":best_epoch,"best_validation_linear_macro_f1":best_score,
              "total_seconds":time.perf_counter()-started,"final_metrics":history[-1],
              "protected_b0_unchanged":True,"selection":selection}
    atomic_json_dump(result,output/"pretrain_summary.json")
    return result
