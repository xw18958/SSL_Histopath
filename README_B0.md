# PanNuke B0 SSL

This package implements the B0 direct latent-prediction ablation on all 7,901
PanNuke images. Its PLIP ViT-B/32 encoder is constructed from the local model
configuration with random weights; the local pretrained weights are never read.
The balanced downstream protocol is transductive because its images are included
without labels in SSL pretraining.

The stages are intentionally separate:

```bash
source /raid1/xwan0900/venvs/ftkp_cu128/bin/activate
cd /raid1/xwan0900/SSL_proj
python -m pip install -e .
pytest -q
python scripts/smoke_test_b0.py
python scripts/calibrate_degradation.py --config configs/calibration.yaml
python scripts/tune.py --config configs/tune_b0.yaml
python scripts/pretrain_b0.py --config outputs/tuning/b0/selected_b0_config.json
python scripts/linear_probe.py --config configs/linear_probe.yaml
```

Calibration writes measurements and endpoint metadata but no figures. Training
and probing write independent checkpoints, machine-readable metrics, and curves.
No source image or model file is changed, and no degraded image is persisted.
Only the best SSL checkpoint (lowest epoch-average total loss) and the best linear
probe checkpoint (highest validation macro-F1) are retained. The completed SSL
checkpoint contains only the student encoder; teacher, predictor, optimizer, per-epoch,
and non-winning tuning checkpoints are discarded.
