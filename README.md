# SSL Histopathology

Self-supervised learning experiments for histopathology using the **PanNuke** dataset.

This repository contains code for preparing a fixed PanNuke classification split with balanced validation/test sets and experimenting with several SSL objectives built on a fresh PLIP/CLIP ViT-B/32 vision encoder.

## Methods

- **B0** — degradation-conditioned direct latent prediction.
- **B1** — residual discrete-velocity prediction between degradation states.
- **I-JEPA** — masked latent prediction adapted to the same vision backbone.

The degradation experiments use image quality changes such as **defocus blur** and **resolution degradation** during SSL training.

## Setup

```bash
git clone https://github.com/xw18958/SSL_Histopath.git
cd SSL_Histopath
pip install -e .
```

Main dependencies are listed in `requirements.txt`.

## Repository Structure

```text
configs/                 Experiment configurations
scripts/                 Training, evaluation, probing, and calibration scripts
src/pannuke_ssl/         Core SSL implementation
prepare_pannuke19.py     PanNuke dataset preparation
pannuke19_dataset.py     PanNuke dataset loader
pannuke19_metadata.csv   Generated PanNuke train/validation/test metadata
tests/                   Unit tests
README_B0.md             Detailed B0 experiment instructions
```

## Basic Checks

```bash
pytest -q
```

For the B0 training pipeline and example commands, see [`README_B0.md`](README_B0.md).

## Dataset

Experiments are based on **PanNuke**, a multi-organ histopathology dataset. The repository stores metadata and dataset-loading code rather than duplicating the source images.

The current standard protocol uses all 7,901 images: 6,305 train, 798 validation, and 798 test. Validation and test contain exactly 42 images from each of the 19 tissue classes; all remaining images form the training split. Regenerate `pannuke19_metadata.csv` before new runs with `prepare_pannuke19.py`; the SSL framework rejects metadata that does not match this protocol.

## Status

Research code under active development for comparing self-supervised representation learning strategies in histopathology.
