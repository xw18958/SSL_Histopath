# MAE comprehensive fidelity and fairness audit (27 October 2026)

**Verdict:** faithful core MAE objective using the EXACT same CLIP ViT-B/32
backbone as LeJEPA and Simplex. This is a controlled **adaptation**, NOT a
bitwise or architectural reproduction of official MAE ViT-B/16.

**References inspected**
- He et al., Masked Autoencoders Are Scalable Vision Learners, CVPR 2022:
  https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html
- Official source:
  https://github.com/facebookresearch/mae/blob/main/models_mae.py
  https://github.com/facebookresearch/mae/blob/main/main_pretrain.py
  https://github.com/facebookresearch/mae/blob/main/engine_pretrain.py
  https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
  https://github.com/facebookresearch/mae/blob/main/util/lr_sched.py
  https://github.com/facebookresearch/mae/blob/main/PRETRAIN.md
- Our repository: shared model definition, LeJEPA and Simplex adapters,
  base and method-specific configs, MAE adapter, trainer, frozen downstream
  classification and image-text retrieval pipeline.

## Exact algorithm checks

| Item | Audit result |
|---|---|
| Random 75% per-image masking | PASS: exactly 48/64 masked |
| Patch position embedding before removing masked tokens | PASS |
| Encoder sees ONLY visible 16 tokens and CLS | PASS |
| Decoder restores original order and uses learned mask tokens | PASS |
| Decoder: width512, depth8, 16 heads, MLP4, pre-LN | PASS: equivalent topology |
| Fixed decoder 2D sin/cos positional encoding | PASS: numerical official-formula parity |
| Patchify target spatial/channel order | PASS |
| Masked-only mean squared reconstruction loss | PASS |
| Per-patch mean and variance normalization | PASS |
| Input and reconstruction target use SAME normalized pixel tensor | FIXED: both CLIP-normalized |
| Random resized crop 0.2-1.0, bicubic, horizontal flip | PASS |
| AdamW betas (0.9, 0.95), weight decay 0.05 | PASS |
| Source-scaled peak LR | PASS: 0.00015 * 128/256 = 0.000075 |
| Cosine schedule with warmup | PASS, fractionally adapted |
| Full-image downstream encoder, no decoder used | PASS |

PyTorch TransformerEncoderLayer with pre-norm/GELU is topology-equivalent
to the original MAE decoder using timm Block, not source-code-identical.

## Shared-backbone verification (not just YAML matching)

The actual LeJEPA, Simplex and MAE constructors yield the SAME full initial
FreshPLIPVisionEncoder SHA-256:

    eccdcfba016917e36667d3cf4288f41869b1a3593345729e3115b638d4a8d372

All share: 256x256 input, patch32, 64 patch tokens, width768,
random seed 20260903, 7901 final PanNuke source images, checkpoint epochs
100/150/200/250 and mean final-patch-token frozen downstream pooling.

MAE visible-token training path using all 64 patches equals the original
full-image downstream encoder exactly: max absolute output error **0.0**.

Changing ALL pixels in the masked patches under the same mask does not
change the encoder features or the decoder output. There is no masked-pixel
leakage into the latent encoder.

## Noncanonical differences (must disclose in CVPR submission)

1. **Matched patch32 vs official patch16**: official ViT-B/16 with 224px
   images has 196 patches (49 visible at 75% masking); matched ViT-B/32
   with 256px images has 64 patches (only 16 visible). This substantially
   changes the masking task, target size (3072 vs 768 pixels per patch),
   and potentially representation quality. Hence this is a matched-
   backbone algorithm comparison, not official-pretraining reproduction.
2. **CLIP backbone**: same fresh learned positions, CLIP pre/post LN and
   normalization as our methods, instead of MAE's official fixed encoder
   sin/cos positions and timm ViT structure. No pretrained PLIP weights.
3. **Normalization**: official MAE uses ImageNet-normalized input for BOTH
   encoder and pixel targets; matched project uses CLIP-normalized input
   for BOTH. Original raw-RGB reconstruction-target mismatch was corrected.
4. **Epochs**: official recommended PRETRAIN.md example uses 800 epochs;
   matched comparison runs 250. MAE paper Figure 7 shows performance
   improving with longer pretraining; do not claim beating a fully trained
   official MAE using this short-data short-budget comparison.
5. **Effective batch**: official uses 4096 with accumulation/multiple GPUs;
   matched comparison uses batch128. Source base_lr=1.5e-4 gives peak
   7.5e-5 by official linear scaling, but optimization regime differs.
6. **Warmup**: official 40/800 epochs (=5%) mapped to 5% of 250
   (=12.5 epoch equivalents), rather than copying 40 literal epochs.
7. **Sampler**: existing project drop_last=False; official MAE drop_last=True.
   Retain shared data coverage across all methods, disclose remainder batch.
8. **Precision**: shared BF16 vs official CUDA AMP commonly using FP16.
9. **Views and compute**: MAE processes one augmented 256px crop, 16
   encoder-visible patch tokens/source image. LeJEPA/Simplex processes
   2 global and 6 local crops, total 2*64+6*9=182 encoder patch tokens
   per source image. Equal images/epochs is NOT equal crops/tokens/FLOPs.
   MAE decoder FLOPs must be included in whole-method compute.
10. **Hyperparameter selection**: training itself uses no label targets,
    but optional PanNuke linear-probe LR tuning uses labeled development
    train/validation data. Disclose any supervised tuning consistently,
    use sequential single-parameter tuning only, and never select from the
    final external downstream test split.

## Verified tests (smoke tests only; no experiments launched)

- 29 regression tests PASS across MAE, existing SSL standard config,\n  downstream probe tuning and Simplex. This includes 12 MAE-specific\n  CPU unit tests, official decoder positional-grid parity
  and reconstruction target-normalization regression.
- Forward-only verification: full-visible feature equality (max error 0),
  no masked-content influence on encoder or decoder.
- RTX PRO 6000 gpu1-jinman-2: real PanNuke two-sample, one optimizer
  step PASS, peak allocated 2.165 GiB.
- RTX PRO 6000 gpu1-358-0: synthetic batch128, one optimizer step PASS,
  peak allocated 3.471 GiB, reserved 3.920 GiB.
- gpu2-358-0: two RTX PRO 6000 Max-Q devices reachable, but old Python
  environment is missing; smoke blocked by environment, not model.
- Existing SSL framework checkpoint roundtrip (strict load into freshly\n  initialized MAE object) PASS; verified original encoder hash restored.\n- No pretraining epoch, tuning trial, or downstream run was launched.

## Recommended comparison, pending explicit user approval

**Primary (shared-backbone) comparison**: Same fresh ViT-B/32,
same PanNuke unlabeled images, 250 epochs, 100/200-epoch headline
checkpoints, same classification/retrieval/few-shot/segmentation
frozen-backbone evaluation protocols. Use recommended MAE masking,
normalized pixel target, decoder, RRC+flip augmentation and source-scaled
learning rate. Save 150/250 as existing pipeline already does.

**Mandatory paper reporting**: number of raw source images, epochs,
gradient steps, augmented views, encoder-visible tokens, GPU-hours,
peak VRAM, FLOPs if measurable, training schedules and held-out test
protocols. Avoid describing equal epochs as equal compute or an exact
official MAE replication.

**Choice to confirm before experiments**: run only source-scaled MAE LR,
or apply existing single-parameter sequential LR development tuning
with matched selection budget. An additional 800-epoch MAE run can be
considered later to address undertraining, but is NOT authorized now.

**STOP: No full experiments until explicit user approval.**
