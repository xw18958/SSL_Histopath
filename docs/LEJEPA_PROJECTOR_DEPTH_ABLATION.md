# LeJEPA Projector Depth Ablation

## Verified baseline

The completed LeJEPA baseline projector is:

`Linear(768,512) -> MLP(512; [2048,2048,512])`

Expanded by torchvision's `MLP`, this is:

`768 -> 512 -> 2048 -> 2048 -> 512`

The first `768 -> 512` layer is a plain linear bottleneck. Inside the MLP, each non-final hidden layer is `Linear -> BatchNorm1d -> ReLU -> Dropout(0)`, and the final 512-dimensional layer is `Linear -> Dropout(0)`.

This was double-checked against both `origin/main` and the already-completed baseline checkpoint state dict.

## Correct controlled experiment

Keep the `768 -> 512` entrance, output dimension, layer type, BatchNorm/ReLU placement, backbone, views, augmentations, SIGReg objective, optimizer, seed, data, and downstream protocol fixed. Change only the number of 2048-wide hidden layers inside the MLP:

- shallow: `768 -> 512 -> 2048 -> 512`
- baseline: `768 -> 512 -> 2048 -> 2048 -> 512`
- deep: `768 -> 512 -> 2048 -> 2048 -> 2048 -> 512`

Thus shallow is exactly one 2048 block below baseline and deep is exactly one 2048 block above baseline.

## Matched training budget

The completed baseline used a 300-epoch learning-rate schedule and saved checkpoints at 100, 150, 200, 250, and 300 epochs.

The corrected ablations stop at epoch 250 to avoid unnecessary compute, while retaining `schedule_epochs=300`. Therefore the learning-rate trajectory through epochs 1-250 is identical to the completed baseline. Compare only the matched checkpoints:

`100, 150, 200, 250`

Seed: `20260903`.

Registered run IDs are unchanged so the incorrect outputs can be overwritten:

- `ssl-lejepa-proj-shallow-s20260903`
- `ssl-lejepa-proj-deep-s20260903`

The previous outputs from the confounded projector definitions must not be used as projector-depth evidence.
