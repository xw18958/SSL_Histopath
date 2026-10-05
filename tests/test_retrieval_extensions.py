from pathlib import Path

import torch

from pannuke_ssl.ssl_framework.image_retrieval import class_retrieval_metrics
from pannuke_ssl.ssl_framework.image_text_datasets import _group_split


def test_class_retrieval_excludes_self_and_scores_perfect_nearest_neighbour():
    features = torch.tensor([[1.0, 0.0], [0.99, 0.01], [-1.0, 0.0], [-0.99, -0.01]])
    labels = torch.tensor([0, 0, 1, 1])
    metrics, rows = class_retrieval_metrics(features, labels, ks=(1, 2), chunk_size=2)
    assert len(rows) == 4
    assert metrics["precision@1"] == 1.0
    assert metrics["recall@1"] == 1.0
    assert metrics["hit_rate@1"] == 1.0
    assert metrics["precision@2"] == 0.5
    assert metrics["recall@2"] == 1.0
    assert metrics["mAP"] == 1.0


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
