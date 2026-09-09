# Self-Flow-Pixel-B/32 PanNuke experiment

## Outcome

Self-Flow-Pixel-B/32 is a pixel-space adaptation for controlled comparison, not the exact published latent-space implementation. Tuning selected LR `1e-4` and representation weight gamma `0.25`. The full run completed 295 epochs before an authorized plateau stop; epoch 10 was selected by validation-only macro-F1. Final clean t=0 frozen probing achieved 21.2601% validation macro-F1; the one-time held-out test achieved 16.4984% macro-F1 and 22.2672% accuracy.

## Smoke and controls

The real-data CUDA smoke passed before tuning and after the tuner was installed. It verified output `[2,64,768]`, finite losses/gradients, EMA update, exactly 16/64 DTS tokens, teacher never noisier than student, no labels/VAE, and parameter ratios 1.0237x B0 and 1.0270x I-JEPA. SSL used all 7,901 images without labels; test was not accessed until final probing.

Fixed design: 256x256 RGB, 32x32 raw pixel patches, 64 tokens, width 768, depth 8, 12 heads, student layer 2, teacher layer 6, mask ratio 0.25, independent uniform t/s, teacher timestep min(t,s), shared noise, rectified-flow target `noise-x0`, flow + gamma representation cosine loss, AdamW, BF16, batch/effective batch 128, LR 1e-4 to 1e-6, WD 0.04 to 0.40, EMA 0.996 to 1.0, clipping 1.0, monitor every 10 epochs. No B0/I-JEPA/Change-JEPA artifacts were modified or rerun.

## Tuning

Exactly four trials used two SSL epochs and a validation-only six-candidate probe grid capped at five epochs. Seeds were base seed plus trial number.

| Trial | LR | gamma | Seed | Val macro-F1 | Status | Runtime (s) |
|---:|---:|---:|---:|---:|:---:|---:|
| 1 | 5e-05 | 0.25 | 20260904 | 7.7263% | complete | 54.76 |
| 2 | 5e-05 | 0.50 | 20260905 | 6.4393% | complete | 46.29 |
| 3 | 1e-04 | 0.25 | 20260906 | 10.4641% | complete | 42.73 |
| 4 | 1e-04 | 0.50 | 20260907 | 10.2642% | complete | 35.07 |

Selected solely by validation macro-F1: trial 3, LR `0.0001`, gamma `0.25`. Values were transferred automatically into `configs/self_flow_pixel_b32_duration_pilot.yaml`.

## Duration selection and monitors

The max horizon was 300 epochs. At the user's explicit request, training stopped safely after completed epoch 295, after the completed epoch-290 monitor, because validation representation quality had plateaued. This is the only duration-policy deviation; schedules were unchanged. Selected epoch 10 had validation linear macro-F1 33.1300%. Raw monitor maximum was epoch 220 at 33.1744%, below the strict +0.005 selection threshold.

| Epoch | Linear val F1 | kNN val F1 | Embedding std | Effective-rank frac. | Pairwise cosine | Near-constant frac. |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 13.3781% | 34.2010% | 0.261080 | 0.002311 | 0.881574 | 0.007812 |
| 10 | 33.1300% | 45.5850% | 0.537341 | 0.004853 | 0.873356 | 0.000000 |
| 20 | 27.4789% | 45.2829% | 0.137927 | 0.005902 | 0.974544 | 0.009115 |
| 50 | 25.0192% | 36.6873% | 0.056496 | 0.004529 | 0.991666 | 0.552083 |
| 100 | 31.1912% | 43.9130% | 0.042026 | 0.003938 | 0.992453 | 0.811198 |
| 150 | 27.6945% | 37.8948% | 0.032894 | 0.002983 | 0.985427 | 0.886719 |
| 200 | 28.9855% | 37.1899% | 0.032608 | 0.002345 | 0.979140 | 0.890625 |
| 220 | 33.1744% | 39.1720% | 0.032847 | 0.002281 | 0.978219 | 0.886719 |
| 250 | 27.9368% | 39.3264% | 0.032793 | 0.002142 | 0.976103 | 0.885417 |
| 290 | 29.3165% | 40.2206% | 0.032840 | 0.002091 | 0.975628 | 0.886719 |

Complete monitor history is in `monitor_metrics.csv`; complete per-epoch training dynamics are in `pretrain_metrics.csv`. The latter contains total, flow, representation, timestep statistics, mask fraction, target RMS, LR, WD, EMA momentum, samples, throughput, and peak GPU memory. Losses and gradient-finiteness checks were finite. The existing loop did not persist a separate scalar gradient-norm column; it enforced finite `clip_grad_norm_` every optimizer update.

## Final probe and one-time test

Clean t=0 final-layer tokens were mean-pooled to 768 dimensions after BF16 encoder inference; the encoder was frozen. Feature counts were train 2,052, validation 247, and test 247. Selected probe: LR `0.003`, WD `0.0001`, epoch `17`, validation macro-F1 `21.2601%`.

| Metric | Value |
|:---|---:|
| Test macro-F1 | 16.4984% |
| Test accuracy | 22.2672% |
| Test balanced accuracy | 22.2672% |
| Test weighted F1 | 16.4984% |

The final probe entrypoint completed once after selection froze. A prior shell redirection failed before Python/data loading because the empty output directory did not exist; creating it was the only fix. No second test decode occurred.

## Comparison with saved baselines

| Method | Trainable params | Selected epoch | Monitor val F1 | Probe val F1 | Test macro-F1 | Test accuracy | SSL time (s) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| B0 | existing | 10 | 33.0250% | 34.6569% | 26.6514% | 28.3401% | 2335.8 |
| I-JEPA | existing | 210 | 81.8801% | 81.5902% | 72.5690% | 72.4696% | 815.4 |
| Self-Flow-Pixel-B/32 | 94,080,000 | 10 | 33.1300% | 21.2601% | 16.4984% | 22.2672% | 1248.2 |

Self-Flow probe validation macro-F1 is 13.3968 pp below B0 and 60.3301 pp below I-JEPA; test macro-F1 is 10.1529 pp below B0 and 56.0706 pp below I-JEPA. These are single-seed, transductive comparisons and descriptive test deltas, not significance claims.

## Outputs and failures

- Tuning: `outputs/tuning/self_flow/leaderboard.csv`, `tuning_summary.json`, `selected_self_flow_config.json`.
- SSL: `outputs/self_flow_pixel_b32_duration_pilot/` including `best.pt`, metrics, curves, selection and stop metadata.
- Probe/test: `outputs/self_flow_pixel_b32_duration_final_probe/` including probe leaderboard, frozen features, predictions, metrics and confusion matrix.

The first full-training launch hit a Self-Flow-only validation guard that still fixed gamma at 0.5; it was corrected to permit only the specified tuned values, with no training steps in the failed launch. The first probe shell launch failed before Python because its output directory did not exist; it was created and the exact entrypoint then completed once.

Source was synchronized to `a9665544f141a5ebaa5c5953b9ac323335dad644`. Existing baseline outputs/checkpoints remain untouched.
