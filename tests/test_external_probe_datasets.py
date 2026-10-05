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
    ready_external_datasets,
    DatasetProtocolNotReadyError,
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
        for slug in ready_external_datasets():
            standard._validate_action_dataset("downstream", slug)
            assert standard._downstream_output_path(root, slug) == root / "downstream_datasets" / slug
        # The default config is intentionally still PanNuke's fast gate.
        assert config["data"]["expected_ssl_images"] == 6305
        assert config["downstream"]["train_count"] == 6305
        assert config["downstream"]["validation_count"] == 798
        assert config["downstream"]["test_count"] == 798


def test_standard_downstream_runs_all_saved_ssl_checkpoints(tmp_path, monkeypatch):
    standard = _load_script("run_ssl_standard.py")
    config = load_standard_config("lejepa")
    root = tmp_path / "lejepa"
    calls = []
    for epoch in config["training"]["checkpoint_epochs"]:
        checkpoint = root / "pretrain_full" / "checkpoints" / f"epoch_{epoch}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.touch()

    def fake_run_downstream(c, checkpoint, out, *, dataset):
        calls.append((Path(checkpoint), Path(out), dataset))
        return {"encoder_epoch": int(Path(checkpoint).stem.split("_")[-1]), "dataset": dataset}

    monkeypatch.setattr(standard, "run_downstream", fake_run_downstream)
    result = standard._run_downstream_checkpoints(config, root, "crc_val_he_7k")

    assert result["checkpoint_epochs"] == [100, 150, 200, 250, 300]
    assert [call[0].name for call in calls] == [
        "epoch_100.pt", "epoch_150.pt", "epoch_200.pt", "epoch_250.pt", "epoch_300.pt"
    ]
    assert [call[1] for call in calls] == [
        root / "downstream_datasets" / "crc_val_he_7k" / f"epoch_{epoch}"
        for epoch in (100, 150, 200, 250, 300)
    ]
    assert all(call[2] == "crc_val_he_7k" for call in calls)


def test_standard_downstream_requires_every_saved_ssl_checkpoint(tmp_path):
    standard = _load_script("run_ssl_standard.py")
    config = load_standard_config("lejepa")
    root = tmp_path / "lejepa"
    with pytest.raises(FileNotFoundError, match="epoch_100.pt"):
        standard._run_downstream_checkpoints(config, root, "crc_val_he_7k")


def test_planned_suite_distinguishes_ready_from_blocked_protocols(tmp_path):
    ready = set(ready_external_datasets())
    assert ready == {"mhist", "crc_val_he_7k", "breakhis_8subtype"}
    assert len(EXTERNAL_DATASETS) == 15
    assert "oral_oscc" in EXTERNAL_DATASETS
    assert "oral_oscc_100x" not in EXTERNAL_DATASETS
    assert "oral_oscc_400x" not in EXTERNAL_DATASETS
    assert "pcgipi_he_4class" not in EXTERNAL_DATASETS
    registry = datasets._dataset_configs()
    assert registry["oral_oscc"].balance_policy == "natural_imbalance"
    assert registry["oral_oscc"].expected_images == 1224
    assert registry["ebhi_seg_6class"].balance_policy == "natural_imbalance"
    assert registry["ebhi_seg_6class"].expected_images == 2228
    with pytest.raises(DatasetProtocolNotReadyError, match="manifest builder still needs implementation"):
        prepare_external_manifest("kather_2016", tmp_path / "outputs")


def test_balanced_8_1_1_rounding_matches_frozen_counts():
    assert datasets._split_counts_8_1_1(1000) == {"train": 800, "val": 100, "test": 100}
    assert datasets._split_counts_8_1_1(990) == {"train": 792, "val": 99, "test": 99}
    assert datasets._split_counts_8_1_1(625) == {"train": 500, "val": 63, "test": 62}
    assert datasets._split_counts_8_1_1(535) == {"train": 428, "val": 54, "test": 53}
    assert datasets._split_counts_8_1_1(339) == {"train": 271, "val": 34, "test": 34}


def test_standard_runner_rejects_blocked_dataset_with_scientific_reason():
    standard = _load_script("run_ssl_standard.py")
    with pytest.raises(ValueError, match="manifest builder still needs implementation"):
        standard._validate_action_dataset("downstream", "kather_2016")
