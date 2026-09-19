# Frozen pretrained PLIP linear-probe comparison

| Result | Test macro-F1 | Delta vs Simplex epoch-260 |
| --- | ---: | ---: |
| Simplex-SIGReg-LeJEPA (matched patch mean) | 0.92328 | 0.00000 |
| PLIP pretrained (matched patch mean, primary) | 0.81577 | -0.10751 |
| PLIP pretrained (native CLS, supplementary) | 0.85619 | -0.06709 |

Both PLIP rows use fixed 2,052/247/247 PanNuke splits, train-only feature normalization, the shared 768-to-19 linear-probe grid, validation-only selection, and one test decode.
