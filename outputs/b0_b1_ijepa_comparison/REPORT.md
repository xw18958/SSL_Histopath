# B0, B1, and I-JEPA: matched transductive comparison

All three rows are **single-seed, transductive frozen linear-probe results**. SSL used all **7,901 PanNuke images unlabeled**, including downstream validation and test images. The test split was not used for SSL duration selection, final-probe selection, or any checkpoint decision; it was evaluated once after each method's selection was frozen.

## Main results

| Test metric | B0 | B1 | I-JEPA | B1 − B0 (pp) | I-JEPA − B1 (pp) |
|---|---:|---:|---:|---:|---:|
| Accuracy | 28.3401% | 22.6721% | 72.4696% | −5.6680 | +49.7976 |
| Balanced accuracy | 28.3401% | 22.6721% | 72.4696% | −5.6680 | +49.7976 |
| Macro-F1 | 26.6514% | 21.7398% | 72.5690% | −4.9116 | +50.8292 |
| Weighted-F1 | 26.6514% | 21.7398% | 72.5690% | −4.9116 | +50.8292 |

Correct test predictions were B0 **70/247**, B1 **56/247**, and I-JEPA **179/247**. Each class has 13 test images, so balanced accuracy equals accuracy and weighted-F1 equals macro-F1 for these balanced test metrics.

For reference, the previously completed I-JEPA−B0 difference is **+44.1296 pp accuracy** and **+45.9176 pp macro-F1**. The three-way result shows B1 below B0 on this frozen-probe endpoint, while I-JEPA remains substantially above both.

## Duration and probe selection

| Selection item | B0 | B1 | I-JEPA |
|---|---:|---:|---:|
| Selected SSL epoch | 10 | 30 | 210 |
| SSL schedule | 300 epochs, 30 warmup | 300 epochs, 30 warmup | 300 epochs, 30 warmup |
| Duration-monitor validation macro-F1 | 33.0250% | 34.8160% | 81.8801% |
| Probe LR | 0.001 | 0.01 | 0.01 |
| Probe weight decay | 0 | 0 | 0 |
| Selected probe epoch | 17 | 19 | 6 |
| Final-probe validation macro-F1 | 34.6569% | 26.8504% | 81.5902% |

The selected SSL epochs are method-specific choices under the same 300-epoch horizon and validation-based strict selection rule; they are not universal optimal durations. B1's duration audit selected epoch 30 with monitor score 0.3481603188 and found no need for extension. B0 selected epoch 10 with score 0.3302503885. I-JEPA selected epoch 210 with score 0.8188010135; its raw best monitor was later in the schedule, but the saved strict selection artifact retains epoch 210.

## Matched protocol

The resolved configurations and summaries record the common comparison scaffold: seed 20260903; fresh PLIP ViT-B/32 encoders; 256×256 inputs; 64 patch tokens of width 768; 7,901 unlabeled SSL images; batch size 128; BF16; AdamW; LR 1e-4→1e-6; weight decay 0.04→0.40; EMA 0.996→1.0; gradient clipping at 5; 300 epochs; and 30 warmup epochs. Final probing froze each selected encoder, mean-pooled FP32 patch tokens, standardized using training features only, and selected among LR {0.001, 0.003, 0.01} × weight decay {0, 0.0001} by validation macro-F1.

The objectives are not identical. B0 is the completed degradation-conditioned endpoint-prediction baseline. B1 retains its predictor capacity and conditioning but parameterizes each adjacent endpoint through residual discrete velocity, Z_hat_t = Z_s + (t - s) v_phi(Z_s, s, action, t - s), with t - s = -0.25. I-JEPA is the separate matched objective baseline using masked student context, positional target queries, and teacher patch targets. Thus this report compares the implemented methods under a matched downstream/training scaffold; it is not evidence that the only difference among all rows is the objective, nor is it an official-scale I-JEPA reproduction.

## Verification and one-time testing

Each method's final-probe summary records one test evaluation and 247 test samples. Each workflow used separate train/validation feature extraction and validation-only probe selection before test access. B0, B1, and I-JEPA all used the saved prediction artifacts for their reported test metrics; no test rerun is part of this report. B1's protected manifest contains 129 entries and records B0/I-JEPA unchanged after B1 completion.

The underlying method reports preserve the per-class tables and confusion matrices: [B0 report](../b0_duration_pilot_final_probe/REPORT.md), [B1 report](../b1_duration_pilot_final_probe/REPORT.md), and the pre-existing [I-JEPA versus B0 report](../ijepa_b0_comparison/REPORT.md). Numeric comparison values here come from the saved linear_probe_summary.json files and saved selection/duration JSON artifacts; no encoder, image loader, probe, training, or guarded test entrypoint was run to create this report.

## Interpretation limits

These are descriptive, single-seed, transductive results on this PanNuke split and this specific lightweight implementation. They do not establish statistical significance, universal optimal epochs, inductive generalization, or the performance of all degradation-dynamics SSL methods. In particular, B1's result should be interpreted as evidence about this residual discrete-velocity formulation under the completed B0-matched protocol, not as a general conclusion about degradation-based self-supervision. SSL exposure to downstream validation/test images was unlabeled and transductive; it must be stated explicitly in any use of these numbers.
