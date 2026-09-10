# I-JEPA vs LeJEPA Source-Faithful Fairness Diagnostic

I-JEPA follows facebookresearch/ijepa; LeJEPA follows the authors' released stable-pretraining 2G+6L implementation. Only shared-backbone/256-input and the chosen I-JEPA predictor depth=4 are deliberate adaptations.

| Setting | Status | Encoder params | Aux params | Pixels/source | Patch tokens/source | GPU ms/step | Wall s/step | Images/s | Peak VRAM GiB | Relative GPU | Relative wall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ijepa_4layer | completed | 87467520 | 7689984 | 131072 | 128 | 102.080 | 0.1032 | 1248.85 | 3.81 | 1.00 | 1.00 |
| lejepa_2g4l | completed | 87467520 | 6697984 | 167936 | 164 | 230.521 | 0.3261 | 525.19 | 11.08 | 2.26 | 3.16 |
| lejepa_2g6l | completed | 87467520 | 6697984 | 186368 | 182 | 270.439 | 0.3609 | 431.69 | 12.41 | 2.65 | 3.50 |

- Identical initial encoder weights: **True**
- Identical 120-step source-image sequence: **True**
- 2G+4L is workload decomposition only; 2G+6L is the released LeJEPA view recipe.