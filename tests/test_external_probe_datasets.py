import importlib.util
from pathlib import Path

import pytest
import torch
from PIL import Image

from pannuke_ssl.probe import build_probe_classifier
from pannuke_ssl.ssl_framework import load_standard_config
from pannuke_ssl.ssl_framework import external_datasets as datasets
from pannuke_ssl.ssl_framework.external_datasets import (
    EXTERNAL_DATASETS,
    ExternalDatasetConfig,
    ExternalProbeImageDataset,
    load_external_manifest,
    manifest_path,
    prepare_external_manifest,
)


def _load_script(name: str):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(name, root / "scripts" / name)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _touch_crc(root: Path, class_names: tuple[str, ...]) -> None:
    for class_name in class_names:
        directory = root / class_name
        directory.mkdir(parents=True)
        for index in range(339):
            (directory / f"{index:03d}.png").touch()


def _touch_breakhis(root: Path, class_names: tuple[str, ...]) -> None:
    for class_name in class_names:
        directory = root / "benign" / "SOB" / class_name / "SOB_B_TEST" / "40X"
        directory.mkdir(parents=True)
        for index in range(444):
            (directory / f"{index:03d}.png").touch()


@pytest.mark.parametrize(
    ("slug", "builder", "expected", "split_counts"),
    (
        ("crc_val_he_7k", _touch_crc, 339 * 9, {"train": 271 * 9, "val": 34 * 9, "test": 34 * 9}),
        ("breakhis_8subtype", _touch_breakhis, 444 * 8, {"train": 356 * 8, "val": 44 * 8, "test": 44 * 8}),
    ),
)
def test_prepared_optional_manifests_are_deterministic_and_balanced(tmp_path, monkeypatch, slug, builder, expected, split_counts):
    registry = datasets._dataset_configs()
    original = registry[slug]
    root = tmp_path / slug
    builder(root, original.class_names)
    config = ExternalDatasetConfig(slug, root, original.class_names, original.split_policy, expected)
    monkeypatch.setattr(datasets, "_dataset_configs", lambda: {**registry, slug: config})
    output = tmp_path / "outputs"
    first = prepare_external_manifest(slug, output)
    frozen = load_external_manifest(slug, output)
    assert first["split_counts"] == split_counts
    assert frozen.split_counts == split_counts
    assert all(len({counts[split] for counts in frozen.manifest["selected_class_split_counts"].values()}) == 1 for split in ("train", "val", "test"))
    assert frozen.manifest_sha256 == first["manifest_sha256"]
    with pytest.raises(FileExistsError):
        prepare_external_manifest(slug, output)


def test_manifest_checksum_rejects_mutation(tmp_path, monkeypatch):
    registry = datasets._dataset_configs()
    original = registry["crc_val_he_7k"]
    root = tmp_path / "crc"
    _touch_crc(root, original.class_names)
    config = ExternalDatasetConfig("crc_val_he_7k", root, original.class_names, original.split_policy, 339 * 9)
    monkeypatch.setattr(datasets, "_dataset_configs", lambda: {**registry, "crc_val_he_7k": config})
    prepare_external_manifest("crc_val_he_7k", tmp_path / "outputs")
    path = manifest_path("crc_val_he_7k", tmp_path / "outputs")
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        load_external_manifest("crc_val_he_7k", tmp_path / "outputs")


def test_external_rgb_preprocessing_is_fixed_to_raw_256_square(tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (320, 160), color=(12, 34, 56)).save(image_path)
    dataset = ExternalProbeImageDataset(
        [{"relative_path": "sample.png", "class_id": 1, "record_index": 7}], tmp_path
    )
    image, label, record_index = dataset[0]
    assert image.shape == (3, 256, 256)
    assert image.dtype == torch.uint8
    assert (label, record_index) == (1, 7)
    assert image[:, 128, 128].tolist() == [12, 34, 56]


@pytest.mark.parametrize("feature_dim", (512, 768))
@pytest.mark.parametrize("num_classes", (2, 8, 9, 19))
def test_dynamic_probe_head_dimensions(feature_dim, num_classes):
    classifier = build_probe_classifier(feature_dim, num_classes)
    assert classifier.weight.shape == (num_classes, feature_dim)
    assert classifier.bias.shape == (num_classes,)


def test_optional_datasets_are_not_standard_actions_or_smokes():
    standard = _load_script("run_ssl_standard.py")
    plip = _load_script("run_plip_linear_probe.py")
    conch = _load_script("run_conch_v1_linear_probe.py")
    standard._validate_action_dataset("downstream", "crc_val_he_7k")
    for action in ("tune", "pretrain", "pipeline", "report"):
        with pytest.raises(ValueError, match="downstream-only"):
            standard._validate_action_dataset(action, "crc_val_he_7k")
    with pytest.raises(ValueError, match="downstream-only"):
        plip._validate_dataset_mode("crc_val_he_7k", smoke_only=True)
    with pytest.raises(ValueError, match="downstream-only"):
        conch._validate_dataset_mode("breakhis_8subtype", preflight_only=True)


def test_all_standard_methods_keep_pannuke_default_and_resolve_optional_outputs(tmp_path):
    standard = _load_script("run_ssl_standard.py")
    assert standard.METHODS == ("ijepa", "lejepa", "simplex_sigreg_lejepa", "dinov3")
    for method in standard.METHODS:
        config = load_standard_config(method)
        root = tmp_path / method
        assert standard._downstream_output_path(root, "pannuke19") == root / "downstream"
        for slug in EXTERNAL_DATASETS:
            standard._validate_action_dataset("downstream", slug)
            assert standard._downstream_output_path(root, slug) == root / "downstream_datasets" / slug
        # The default config is intentionally still PanNuke's fast gate.
        assert config["data"]["expected_ssl_images"] == 7901
        assert config["downstream"]["train_count"] == 2052
