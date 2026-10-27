# Matched-backbone MAE baseline — pending owner approval

**Status:** implement, review, push and smoke-test only. DO NOT launch full
pretraining, hyperparameter tuning, checkpoint benchmarking or downstream
evaluation without explicit owner approval.

Original: He et al., Masked Autoencoders Are Scalable Vision Learners (CVPR 2022).
Official source: https://github.com/facebookresearch/mae and PRETRAIN.md.

## Faithful MAE components

- 75% independently and randomly masked image patches, after patch+positional embeddings.
- Encoder sees only 16/64 visible patches and CLS, NEVER mask tokens.
- Decoder gets learned mask tokens, unshuffles all 64 positions using ids_restore,
  and predicts all original RGB patch pixels.
- Decoder has 512-d embeddings, eight pre-norm transformer blocks, 16 heads,
  fourfold MLP and fixed 2D sin/cos decoder positions; discarded downstream.
- Per-patch pixel-variance normalisation (norm_pix_loss=true);
  squared reconstruction loss ONLY on masked patches.
- MAE-style random resized crop (0.2–1.0) and random horizontal flip.
- AdamW (betas 0.9,0.95), weight decay 0.05, cosine scheduler with warmup.
- All images sourced from the existing validated PanNuke pipeline, no SSL labels.

## Controlled comparison and deviations from official MAE

| Property | MAE adaptation | LeJEPA / Simplex |
|---|---|---|
| Backbone | freshly initialized CLIP ViT-B/32, 256px, 12 layers, 768 dim | SAME |
| Input patches | 8x8 (64) patch grid | SAME |
| Encoder input normalization | CLIP RGB normalization | SAME |
| Encoder positional embeddings | learned CLIP embeddings | SAME |
| Random initialization and seed | shared project setting | SAME |
| Final SSL images | 7901 PanNuke, labels not loaded | SAME |
| Snapshot epochs | 100/150/200/250 | SAME |
| Downstream readout | unmasked mean of final patch tokens | SAME |
| Frozen probes/retrieval and splits | existing shared code | SAME |
| SSL objective, views and optimizer | MAE-specific | method-specific |

The original MAE released encoder is **ViT-B/16**, with fixed 2D sine-cosine
encoder positions and MAE-specific pretraining encoder layers, not CLIP
ViT-B/32. We intentionally use a matched CLIP encoder instead to isolate
the objective. Therefore this is an MAE **algorithm adaptation**, NOT an
exact reimplementation of the original ViT-B/16 architecture. Also,
original MAE trains on ImageNet with 800–1600 epochs and large effective
batch. This project restricts training to 250 epochs on a small histopathology
set, and cannot claim equivalent official MAE benchmark accuracy.

Official base LR 0.00015 x (effective batch 128 / 256) = 0.000075,
using the published MAE scaling rule. Warmup is 5% (12.5/250 epochs)
matching 40/800 original schedule. The optional sequential 20-epoch
single-variable LR search is specified but NOT yet run.

Equal epoch and shared image set is NOT equal FLOPs: MAE sees 25% of tokens
during its encoder pass; LeJEPA sees multiple crops. Future paper tables
must disclose training GPU-hours, accelerator, views, batches, and FLOPs,
plus identical downstream splits, one metric per task and fixed checkpoint
selection. Do not report generated metrics before full verified evaluations.

## Smoke tests allowed before approval

CPU: PYTHONPATH=src pytest -q tests/test_mae_standard.py

GPU (one synthetic batch, ONE optimizer step, no training epochs):
PYTHONPATH=src python scripts/smoke_mae_standard.py

GPU (one real unlabeled PanNuke training batch):
PYTHONPATH=src python scripts/smoke_mae_standard.py --with-dataset

Use existing SSL_PROJECT_ROOT / SSL_MODEL_ROOT / SSL_DATA_ROOT / SSL_RUN_ROOT.
No run of the pretrain, pipeline, tune, downstream or cluster-launch commands.

## Approval gate

Before any full experiments: confirm the matched ViT-B/32 MAE adaptation and
its documented departures from official ViT-B/16, 250-epoch budget, source
LR=0.000075, and whether sequential LR tuning should precede full training.
After approval: choose GPUs with free VRAM, train one fixed seed, evaluate
frozen encoders at 100/200 (and optionally 150/250) on exact same downstream
protocols and compare against LeJEPA/Simplex.

## 27 October 2026 verification record

- Compared against the actual LeJEPA and Simplex method builders: PASS.
- Identical initial FreshPLIPVisionEncoder SHA-256 for all three:
  eccdcfba016917e36667d3cf4288f41869b1a3593345729e3115b638d4a8d372.
- Exact equality checked for patch size 32, 64 patch tokens, feature dimension
  768, source image count 7901, epoch budget 250, checkpoints
  100/150/200/250, seed, and shared downstream patch-mean pooling.
- gpu1-jinman-2 (RTX PRO 6000 Blackwell): 10/10 unit tests PASS;
  synthetic two-image single-step CUDA smoke PASS; real unlabeled
  PanNuke two-image single-step smoke PASS; 2.165 GiB peak allocation.
- gpu1-358-0 (RTX PRO 6000 Blackwell): 10/10 unit tests PASS;
  synthetic two-image single-step CUDA smoke PASS; 2.165 GiB peak allocation.
- gpu2-358-0 (two RTX PRO 6000 Blackwell Max-Q): SSH/VRAM inspection
  PASS; smoke not run because previously used PyTorch environment is
  missing. A suitable environment is needed before that smoke test.
- A two-image smoke VRAM result does NOT prove batch 128 fits.
- Full MAE training and downstream experiments NOT started.

To reproduce the exact backbone comparison:

    PYTHONPATH=src python scripts/audit_mae_backbone_match.py
