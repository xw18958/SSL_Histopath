from pathlib import Path

import torch

from pannuke_ssl.config import load_yaml
from pannuke_ssl.ssl_framework.image_text_datasets import _group_split
from pannuke_ssl.ssl_framework.image_text_retrieval import _retrieval_metrics, _sym_clip_loss, _train_trial


def test_image_text_protocol_uses_frozen_7_1_5_1_5_counts():
    root = Path(__file__).resolve().parents[1]
    config = load_yaml(root / "configs" / "ssl_standard" / "image_text_retrieval_datasets.yaml")
    for dataset in ("arch", "ipath"):
        values = config["datasets"][dataset]
        assert values["train_size"] == 3267
        assert values["val_size"] == 700
        assert values["test_size"] == 700
        assert values["target_total"] == 4667
        assert values["split_ratio"] == [7, 1.5, 1.5]
        protocol = values["alignment_protocol"]
        assert protocol["text_branch"] == "frozen_pretrained_plip_text_encoder_and_text_projection"
        assert protocol["ssl_image_encoder"] == "frozen_checkpoint_representation"
        assert protocol["trainable_parameters"] == "image_projection_head_only"
        assert protocol["image_projection_head"] == "linear_no_bias_ssl_dim_to_plip_512"
        assert protocol["loss"] == "clip_symmetric_cross_entropy"
        assert protocol["selection_metric"] == "val_overall_mean_recall"
        assert protocol["test_policy"] == "untouched_until_projector_and_hyperparameters_are_frozen"


def test_image_text_group_split_hits_exact_counts_without_group_leakage():
    records = []
    # Plenty of singleton and paired groups make exact 4/3 targets possible.
    for group_id, size in [("a", 2), ("b", 2), ("c", 1), ("d", 1), ("e", 1), ("f", 1), ("g", 1), ("h", 1)]:
        for index in range(size):
            records.append({"group_id": group_id, "relative_path": f"{group_id}_{index}.png", "text": group_id})
    selected, counts = _group_split(records, val_count=3, test_count=4, train_count=None, seed=20260903)
    assert counts == {"train": 3, "val": 3, "test": 4}
    split_by_group = {}
    for row in selected:
        split_by_group.setdefault(row["group_id"], set()).add(row["split"])
    assert all(len(values) == 1 for values in split_by_group.values())


def test_symmetric_clip_loss_and_retrieval_metrics_are_bidirectional():
    image = torch.eye(8)
    text = torch.eye(8)
    loss = _sym_clip_loss(image, text, 10.0)
    assert float(loss) < 0.01
    metrics = _retrieval_metrics(image, text)
    for key in ("i2t_r@1","i2t_r@5","i2t_r@10","t2i_r@1","t2i_r@5","t2i_r@10","overall_mean_recall"):
        assert metrics[key] == 1.0


def test_projector_trial_has_only_one_trainable_weight_tensor():
    generator = torch.Generator().manual_seed(5)
    images = torch.randn(40, 16, generator=generator)
    texts = torch.nn.functional.normalize(torch.randn(40, 8, generator=generator), dim=-1)
    result = _train_trial(
        images[:32], texts[:32], images[32:], texts[32:],
        lr=1e-3, wd=0.0, epochs=2, batch_size=8, seed=7,
        device=torch.device("cpu"), logit_scale=10.0,
    )
    assert list(result["state"]) == ["weight"]
    assert result["state"]["weight"].shape == (8, 16)
    assert 0.0 <= result["val_overall_mean_recall"] <= 1.0


def test_standard_retrieval_runs_all_saved_ssl_checkpoints(tmp_path, monkeypatch):
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("run_ssl_standard_retrieval", root / "scripts/run_ssl_standard.py")
    assert spec and spec.loader
    standard = importlib.util.module_from_spec(spec); spec.loader.exec_module(standard)
    config = standard.load_standard_config("lejepa")
    run_root = tmp_path / "run"
    calls=[]
    for epoch in config["training"]["checkpoint_epochs"]:
        p=run_root/"pretrain_full"/"checkpoints"/f"epoch_{epoch}.pt"; p.parent.mkdir(parents=True,exist_ok=True); p.touch()
    def fake(c, checkpoint, out, *, dataset):
        calls.append((Path(checkpoint),Path(out),dataset)); return {"dataset":dataset}
    monkeypatch.setattr(standard,"run_image_text_retrieval",fake)
    result=standard._run_image_text_retrieval_checkpoints(config,run_root,"arch")
    assert result["checkpoint_epochs"] == [100,150,200,250]
    assert [x[0].name for x in calls] == [f"epoch_{e}.pt" for e in (100,150,200,250)]
    assert [x[1] for x in calls] == [run_root/"image_text_retrieval"/"arch"/f"epoch_{e}" for e in (100,150,200,250)]
    assert all(x[2] == "arch" for x in calls)
