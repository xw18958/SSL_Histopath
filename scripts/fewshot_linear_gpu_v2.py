#!/usr/bin/env python3
"""Frozen-backbone, GPU-batched, trainable few-shot linear probing.

Loads the *existing* patch_benchmark_v1 feature caches. No encoder/projector
updates, SSL retraining, feature extraction, or test-based model selection.
Fits independent L2-regularized multinomial logistic regressions for fixed
1/5/10-shot support subsets, with 100 support subsets per setting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

# Must precede CUDA initialization for deterministic cublas operations.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F

SHOTS = (1, 5, 10)
REPEATS = 100
SEED = 20260903
METHODS = ("lejepa", "simplex_k8", "mae", "dinov3", "ijepa")
EPOCHS = (100, 200)
MIN_FREE_MIB = 8192


def sha_json(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf8")).hexdigest()


def normalize(x):
    a = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    if not np.isfinite(a).all() or np.any(n < 1e-12):
        raise ValueError("Nonfinite or degenerate input features")
    return a / n


def selected_subsets(dataset, indices, labels):
    """Match scripts/patch_benchmark.py:support_selections exactly."""
    indices = np.asarray(indices, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    assert len(indices) == len(set(map(int, indices)))
    classes = sorted(set(map(int, labels)))
    by_class = {c: np.sort(indices[labels == c]) for c in classes}
    basis = int(sha_json([dataset, "main", SEED])[:16], 16)
    selections = {}
    for k in SHOTS:
        if any(len(z) < k for z in by_class.values()):
            raise ValueError(f"{dataset}: insufficient examples for {k}-shot")
        subs = []
        for rep in range(REPEATS):
            rng = np.random.default_rng(np.random.SeedSequence(
                [basis & 0xffffffff, basis >> 32, int(k), rep]))
            chosen = []
            for cls in classes:
                chosen.extend(map(int, rng.choice(by_class[cls], k, replace=False)))
            subs.append(chosen)
        selections[str(k)] = subs
    return selections


def read_cache(path, method, epoch, dataset):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"].item()))
        arr = {s: (np.asarray(z[s+"_features"], dtype=np.float32),
                   np.asarray(z[s+"_labels"], dtype=np.int64),
                   np.asarray(z[s+"_indices"], dtype=np.int64))
               for s in ("train", "test")}
    expected = dict(method=method, epoch=epoch, dataset=dataset, split_variant="main",
                    frozen=True, no_test_augmentation=True,
                    encoder="pre-projector backbone mean final patch tokens")
    for key, val in expected.items():
        if meta.get(key) != val:
            raise RuntimeError(f"{path}: bad metadata {key}: {meta.get(key)} != {val}")
    for name, (x,y,idx) in arr.items():
        if x.ndim != 2 or len(x) != len(y) or len(y) != len(idx):
            raise ValueError(f"{name} array shape mismatch")
        if x.shape[1] != meta["feature_dim"]:
            raise ValueError("Feature dimensionality mismatch")
        if len(set(map(int, idx))) != len(idx):
            raise ValueError("Duplicate record indices")
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite feature")
    if not meta.get("checkpoint_sha256") or not meta.get("manifest_sha256"):
        raise ValueError("Missing immutable provenance hashes")
    if set(map(int, arr["test"][1])) - set(map(int, arr["train"][1])):
        raise ValueError("Test labels absent from training set")
    if set(map(int, arr["train"][2])) & set(map(int, arr["test"][2])):
        raise ValueError("Train/test source record overlap")
    return meta, arr


def summary(x):
    x = np.asarray(x, dtype=np.float64)
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    radius = float(student_t.ppf(.975, len(x)-1) * std / np.sqrt(len(x))) if len(x)>1 else 0.0
    return dict(mean=mean, std=std, ci95_lower=mean-radius,
                ci95_upper=mean+radius, n_support_subsets=len(x))


def score_batch(targets, predictions, classes):
    """No sklearn overhead for the many independent test-set evaluations."""
    y = np.asarray(targets, dtype=np.int64)
    C = len(classes)
    result = {"accuracy": [], "balanced_accuracy": [], "macro_f1": []}
    support = np.bincount(y, minlength=C)
    if np.any(support == 0):
        raise ValueError("Some training classes absent from test set")
    for pred in predictions:
        pred = np.asarray(pred, dtype=np.int64)
        tp = np.bincount(y[pred == y], minlength=C)
        predicted = np.bincount(pred, minlength=C)
        recall = tp / support
        denom = support + predicted
        f1 = np.divide(2 * tp, denom, out=np.zeros(C, dtype=np.float64),
                       where=denom != 0)
        result["accuracy"].append(float(np.mean(pred == y)))
        result["balanced_accuracy"].append(float(recall.mean()))
        result["macro_f1"].append(float(f1.mean()))
    return result


def fit_gpu_batched(support_x, support_y, test_x, *, device, C_reg=1.0,
                    max_iter=200):
    """GPU-batched convex logistic-regression warm starts, independently
    refined by scikit-learn LBFGS for *every* support subset.

    Multinomial CE for >=3 classes; binary logistic BCE for 2 classes.
    Both have the sklearn L2 objective with no intercept penalty. sklearn
    is the definitive prediction backend, not the batch GPU stopping rule.
    """
    from sklearn.linear_model import LogisticRegression
    X = torch.tensor(support_x, dtype=torch.float32, device=device)
    Y = torch.tensor(support_y, dtype=torch.int64, device=device)
    B,N,D = X.shape
    C = int(Y.max().item())+1
    if C<2 or not np.array_equal(np.unique(support_y),np.arange(C)):
        raise ValueError("Unsupported class encoding")
    channels = 1 if C==2 else C
    W = torch.nn.Parameter(torch.zeros((B,D,channels),device=device))
    b = torch.nn.Parameter(torch.zeros((B,1,channels),device=device))
    opt = torch.optim.LBFGS([W,b],lr=1.0,max_iter=max_iter,
                            tolerance_grad=1e-8,tolerance_change=1e-11,
                            history_size=20,line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(set_to_none=True)
        logits=X.bmm(W)+b
        if C==2:
            ce=F.binary_cross_entropy_with_logits(
                logits.squeeze(-1),Y.float(),reduction="none").mean(1)
        else:
            ce=F.cross_entropy(logits.transpose(1,2),Y,reduction="none").mean(1)
        penalty=W.square().sum((1,2))*(0.5/(C_reg*N))
        loss=(ce+penalty).sum()
        loss.backward()
        return loss

    opt.step(closure)
    closure()
    with torch.no_grad():
        norms=torch.maximum(W.grad.reshape(B,-1).abs().max(1).values,
                            b.grad.reshape(B,-1).abs().max(1).values)
        worst=float(norms.max().item())
        if not np.isfinite(worst):
            raise RuntimeError("Nonfinite GPU gradients")
        gpu_coef=W.cpu().numpy()
        gpu_bias=b[:,0,:].cpu().numpy()
        if C==2:
            gpu_pred=((torch.einsum("nd,bdc->bnc",
                       torch.as_tensor(test_x,device=device),W)+b)[:,:,0]>0
                     ).long().cpu().numpy()
        else:
            gpu_pred=(torch.einsum("nd,bdc->bnc",
                      torch.as_tensor(test_x,device=device),W)+b
                     ).argmax(-1).cpu().numpy()

    authoritative=np.empty_like(gpu_pred)
    mismatch=0
    warm_failed=0
    max_n_iter=0
    for j in range(B):
        ref=LogisticRegression(C=C_reg,solver="lbfgs",
                               max_iter=4000,tol=1e-11,warm_start=True)
        if C==2:
            ref.coef_=gpu_coef[j].T.copy()
            ref.intercept_=gpu_bias[j].copy()
        else:
            ref.coef_=gpu_coef[j].T.copy()
            ref.intercept_=gpu_bias[j].copy()
        ref.classes_=np.arange(C)
        ref.n_features_in_=D
        try:
            ref.fit(support_x[j],support_y[j])
        except (ValueError,AttributeError):
            warm_failed+=1
            ref=LogisticRegression(C=C_reg,solver="lbfgs",
                                   max_iter=4000,tol=1e-11)
            ref.fit(support_x[j],support_y[j])
        max_n_iter=max(max_n_iter,int(ref.n_iter_.max()))
        authoritative[j]=ref.predict(test_x)
        mismatch+=int(np.count_nonzero(authoritative[j]!=gpu_pred[j]))
    return authoritative,worst,dict(
        gpu_fits=B,sklearn_refinements=B,
        gpu_reference_differing_test_predictions=mismatch,
        sklearn_warmstart_failures=warm_failed,
        sklearn_max_n_iter=max_n_iter)


def evaluate(args):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required; do not silently execute on CPU")
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    free_mib = free//(1024**2)
    if free_mib < MIN_FREE_MIB:
        raise RuntimeError(f"GPU {device} has only {free_mib} MiB free "
                           f"(minimum {MIN_FREE_MIB} MiB)")
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    cache = args.cache_root / "full/main" / args.method / f"epoch_{args.epoch}" / (args.dataset+".npz")
    if not cache.is_file():
        raise FileNotFoundError(cache)
    meta, arrays = read_cache(cache, args.method, args.epoch, args.dataset)
    tx, ty, ti = arrays["train"]
    qx, qy, qi = arrays["test"]
    support = selected_subsets(args.dataset, ti, ty)
    signature = sha_json(support)
    if args.reference_root:
        original = args.reference_root / "full/main" / args.method / f"epoch_{args.epoch}" / (args.dataset+".json")
        if not original.is_file():
            raise FileNotFoundError(f"Missing existing benchmark to verify support subset identity: {original}")
        old = json.loads(original.read_text())
        if old["support_subsets_sha256"] != signature:
            raise RuntimeError("Support subset IDs differ from the original training-free benchmark")
        if old["manifest_sha256"] != meta["manifest_sha256"] or old["row_order_sha256"] != meta["row_order_sha256"]:
            raise RuntimeError("Manifests differ from original evaluation")
        if old["feature_checkpoint_sha256"] != meta["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint mismatch with original evaluation")
    classes = np.sort(np.unique(ty))
    class_to_pos = {int(c): i for i,c in enumerate(classes)}
    targets = np.asarray([class_to_pos[int(c)] for c in qy], dtype=np.int64)
    tx, qx = normalize(tx), normalize(qx)
    id_to_pos = {int(i): k for k,i in enumerate(ti)}
    count = int(args.smoke_subsets) if args.smoke_subsets else REPEATS
    if not 1 <= count <= REPEATS:
        raise ValueError("Bad number of smoke subsets")
    results = {}
    for shot in SHOTS:
        st = time.monotonic()
        selections = support[str(shot)][:count]
        positions = np.array([[id_to_pos[int(i)] for i in sel] for sel in selections], dtype=np.int64)
        X = tx[positions].copy()
        Y = np.array([[class_to_pos[int(ty[i])] for i in sel] for sel in positions],
                     dtype=np.int64)
        if X.shape[1] != shot*len(classes) or any(
                not np.array_equal(np.bincount(row, minlength=len(classes)),
                                   np.full(len(classes), shot)) for row in Y):
            raise RuntimeError("Incorrect support balancing")
        preds, residual, verification = fit_gpu_batched(X,Y,qx,device=device,
                                                         max_iter=args.max_iter)
        scores = score_batch(targets, preds, classes)
        results[str(shot)] = {
            "metrics": {metric: summary(vals) for metric,vals in scores.items()},
            "per_subset": scores, "max_abs_gpu_objective_gradient": residual,
            "solver_verification": verification,
            "elapsed_seconds": round(time.monotonic()-st,3)
        }
        print(f"OK {args.dataset} {args.method} e{args.epoch} "
              f"{shot}-shot x{count}: accuracy={results[str(shot)]['metrics']['accuracy']['mean']:.5f} "
              f"grad={residual:.2e} sklearn_refined={verification['sklearn_refinements']} "
              f"gpu_disagreements={verification['gpu_reference_differing_test_predictions']}", flush=True)
    payload = dict(protocol="fewshot_linear_gpu_v2", mode="smoke" if args.smoke_subsets else "full",
                   method=args.method, epoch=args.epoch, dataset=args.dataset,
                   shots=list(SHOTS), repeats=count, classifier="multinomial logistic regression",
                   optimizer="GPU torch LBFGS + per-fit sklearn LBFGS numerical refinement", C=1.0,
                   max_iter=args.max_iter, encoder_frozen=True,
                   representation="pre-projector mean final patch tokens",
                   feature_normalization="per-image L2 only",
                   selected_support_sha256=signature,
                   manifest_sha256=meta["manifest_sha256"],
                   row_order_sha256=meta["row_order_sha256"],
                   checkpoint_sha256=meta["checkpoint_sha256"],
                   n_train=int(len(ty)), n_test=int(len(qy)),
                   classes=list(map(int,classes)), cuda_device=str(device),
                   results=results)
    mode = "smoke" if args.smoke_subsets else "full"
    dest = args.out_root / mode / args.method / f"epoch_{args.epoch}" / (args.dataset+".json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        prev = json.loads(dest.read_text())
        identity = ("selected_support_sha256","manifest_sha256","checkpoint_sha256","mode","repeats","C")
        if any(prev.get(k)!=payload.get(k) for k in identity):
            raise RuntimeError(f"Existing incompatible result: {dest}")
    with tempfile.NamedTemporaryFile("w",dir=dest.parent,delete=False) as f:
        json.dump(payload,f,indent=2,sort_keys=True,allow_nan=False)
        f.write("\n")
        tmp = f.name
    os.replace(tmp,dest)
    print(f"RESULT {dest}",flush=True)
    return payload


def self_test(gpu):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA missing")
    device = torch.device(f"cuda:{gpu}")
    rng=np.random.default_rng(4021)
    B=5; C=3; N=15; D=16
    X=normalize(rng.normal(size=(B,N,D)).astype("float32"))
    Y=np.tile(np.repeat(np.arange(C),N//C),(B,1))
    T=normalize(rng.normal(size=(45,D)).astype("float32"))
    pred,grad,verification=fit_gpu_batched(X,Y,T,device=device,max_iter=250)
    for i in range(B):
        model=LogisticRegression(C=1.0,solver="lbfgs",max_iter=2000,tol=1e-10)
        model.fit(X[i],Y[i])
        a=model.predict(T)
        if not np.array_equal(a,pred[i]):
            raise AssertionError(f"GPU vs sklearn predictions mismatch ({i}): "
                                 f"{np.mean(a==pred[i])}")
        scores=score_batch(Y[i],np.array([Y[i]]),np.arange(C))
        if not np.isclose(scores["accuracy"][0],accuracy_score(Y[i],Y[i])):
            raise AssertionError("Scoring failure")
        if not np.isclose(scores["balanced_accuracy"][0],balanced_accuracy_score(Y[i],Y[i])):
            raise AssertionError("Balanced-accuracy failure")
        if not np.isclose(scores["macro_f1"][0],f1_score(Y[i],Y[i],average="macro")):
            raise AssertionError("F1 failure")
    print(f"SELF_TEST PASS: {B} GPU warm starts refined with sklearn; gradient={grad:.3g}; details={verification}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root",type=Path)
    p.add_argument("--out-root",type=Path)
    p.add_argument("--reference-root",type=Path,default=None)
    p.add_argument("--method",choices=METHODS)
    p.add_argument("--epoch",type=int,choices=EPOCHS)
    p.add_argument("--dataset")
    p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--smoke-subsets",type=int,default=0)
    p.add_argument("--max-iter",type=int,default=200)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:
        self_test(args.gpu)
        return
    if not (args.cache_root and args.out_root and args.method and args.epoch and args.dataset):
        p.error("Evaluation requires --cache-root --out-root --method --epoch --dataset")
    evaluate(args)


if __name__=="__main__":
    main()
