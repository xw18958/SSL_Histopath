# CONCH v1 frozen linear-probe comparison

All rows use the fixed 256px PanNuke protocol: 2,052 train / 247 validation / 247 test images, train-only feature normalization, the shared linear-probe grid, validation-only selection, and one saved test evaluation. This report only reads saved artifacts; it does not load a model or test image.

| Encoder and readout | Feature width | Test macro-F1 | Test accuracy | Δ macro-F1 vs Simplex | Δ macro-F1 vs PLIP patch mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Simplex-SIGReg-LeJEPA epoch-260 — mean final patch tokens | 768 | 0.92328 | 0.92308 | 0.00000 | +0.10751 |
| Frozen PLIP — mean final patch tokens | 768 | 0.81577 | 0.81377 | -0.10751 | 0.00000 |
| Frozen PLIP — native post-layernorm CLS (supplementary, different readout) | 768 | 0.85619 | 0.85425 | -0.06709 | +0.04041 |
| Frozen official CONCH v1 — attention pooling before contrast projection/L2 normalization | 512 | 0.74067 | 0.74899 | -0.18261 | -0.07511 |

For CONCH, the accuracy deltas are -0.17409 versus Simplex and -0.06478 versus PLIP patch mean; the corresponding macro-F1 deltas are -0.18261 and -0.07511. The JSON companion records accuracy and macro-F1 deltas for every applicable row.

CONCH uses `MahmoodLab/CONCH` revision `f9ca9f877171a28ade80228fb195ac5d79003357` and its documented linear-probe call `encode_image(..., proj_contrast=False, normalize=False)`. Its 512-D attention-pooled representation is not a CLS or patch-mean readout; the native 448px checkpoint is evaluated at the registered 256px protocol using CONCH’s official positional-resize path.

Metrics/provenance: `simplex_sigreg_lejepa/downstream/test_metrics.json`, `plip_pretrained_patch_mean/downstream/test_metrics.json`, `plip_pretrained_cls/downstream/test_metrics.json`, and `conch_v1_pretrained_attn_pool/{downstream/test_metrics.json,encoder_provenance.json,preflight.json}` under `outputs/ssl_standard/`.
