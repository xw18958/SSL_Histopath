from pathlib import Path

from pannuke_ssl.config import load_yaml
from pannuke_ssl.ssl_framework.image_text_datasets import _group_split


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
