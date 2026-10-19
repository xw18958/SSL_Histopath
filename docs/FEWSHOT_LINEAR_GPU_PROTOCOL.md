# Frozen-backbone trainable few-shot benchmark (LeJEPA vs Simplex K=8)

## Purpose

A **separate** evaluation of trained few-shot classifiers, not a modification
of the pre-existing prototype few-shot or patch retrieval benchmarks.

- Original SSL weights: unchanged; **no SSL retraining or fine-tuning**.
- Methods: original LeJEPA and Regular-Simplex LeJEPA K=8.
- Checkpoints: pre-existing epochs **100** and **200**.
- Datasets (11): BACH, BreaKHis, CRC-HE-7K, Endometrial, GaHISSDB,
  Kather 2016, LC25000, MHIST, Oral OSCC, RenalCell, SICAPv2.
- Predeclared exclusions: PCam, EBHI-SEG, WSSS4LUAD, Osteosarcoma.
  This 11-dataset subset must be disclosed; it is **not** the original
  15-dataset comprehensive benchmark.

## Representations and splits

Reuse *unchanged* existing cached frozen encoder features:
`outputs/patch_benchmark_v1/features/full/main/{method}/epoch_{epoch}/{dataset}.npz`.

These are mean final backbone patch tokens **before the SSL projector**,
with deterministic image preprocessing (bicubic resize short side to 256,
centered 256 crop, RGB/255). Per-image L2 normalization is the only
feature normalization. The original `main` split variant is retained,
including the recorded exact-image leakage repairs for BreaKHis,
Endometrial and Oral OSCC. No test images or test labels enter training.

For each dataset and shot count 1, 5, 10, re-create the **same original**
100 predetermined class-balanced support selections from the training split.
Match their SHA256 signatures to the saved training-free benchmark results.
The same support IDs, evaluation indices and manifest hashes are required
for both methods and both checkpoint epochs.

## Trainable downstream

Fit **a fresh L2-regularized logistic-regression classifier** separately
for every support subset, using only K labeled training examples per class.
`C=1.0` is fixed across all settings, with no validation- or test-driven
hyperparameter adjustment. Intercept is unpenalized. For binary tasks use
binary logistic loss, and for multiclass tasks use multinomial cross entropy.

The implementation first optimizes 100 independent convex problems in
parallel on an admissible CUDA GPU using full-batch Torch L-BFGS, then
**independently refines every fit with scikit-learn L-BFGS**
(`max_iter=4000`, `tol=1e-11`, GPU weights as warm starts; cold-fit
fallback if warm start cannot be initialized). The scikit-learn refinement
is the authoritative result. This avoids small numerical prediction changes
from shared GPU line searches and assures parity with the conventional
trainable linear-probe solver. GPU/reference disagreements and solver
convergence diagnostics are logged.

GPU admission uses currently available CUDA memory with minimum 8192 MiB.
In environments where the primary Blackwell GPU is incompatible with the
existing PyTorch build, use a **separate** compatible CUDA environment;
do not change the source SSL environment.

## Scoring and uncertainty

Score every held-out test image for each independently trained classifier.
Store per-subset accuracy (primary), balanced accuracy and macro-F1, plus
100-subset mean, sample SD and 95% Student-t CI. Compute paired
Simplex-minus-LeJEPA differences using identically indexed support subsets.
Aggregate across datasets by **equal dataset weight**; report dataset-level
wins and individual datasets. Repetitions here are **support-set draws**,
not independent SSL pretraining seeds. Support-resampling CIs do not establish
independent-patient or cross-dataset population significance.

## Run and audit

```bash
python scripts/fewshot_linear_gpu_v2.py --self-test --gpu 0

python scripts/fewshot_linear_gpu_v2.py \
  --cache-root outputs/patch_benchmark_v1/features \
  --reference-root outputs/patch_benchmark_v1/results \
  --out-root outputs/fewshot_linear_gpu_v2 \
  --method simplex_k8 --epoch 200 --dataset mhist --gpu 0

python scripts/fewshot_linear_aggregate_v2.py \
  --root outputs/fewshot_linear_gpu_v2 \
  --out benchmark_reports/fewshot_linear_gpu_v2
```

For a smoke test use `--smoke-subsets 3`: these are the first three
predeclared support subsets and write to a separate `smoke/` subfolder.

The verifier **fails closed** if any of the 44 expected full result files
are absent, if any metric is nonfinite, or if support/manifest/sample identity
or all-100-subset completion fails. All completed summaries must be
recomputed independently from per-subset values. For additional numerical
checks use `scripts/fewshot_linear_real_data_audit.py`, which retrains
selected real-data support subsets using fresh scikit-learn classifiers.

## Reproducibility and limits

- Checkpoint and image-split SHA256 hashes remain in the local result JSONs.
- Private per-image manifests, raw features and support identifiers are **not**
  committed to the public repository.
- Existing training-free few-shot and patch retrieval scores are unchanged.
- `benchmark_reports/fewshot_linear_gpu_v2` contains aggregate tables
  and verification summaries only.
- Histopathology patch datasets with unavailable source-group identifiers
  retain unresolved patient/slide independence and near-duplicate caveats.
- Do not interpret the 100 support subsets as independent pretraining runs.
