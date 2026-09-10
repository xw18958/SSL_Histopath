from __future__ import annotations

import torch

from scripts.b0_ln_gradient_audit import (
    _category,
    _cosine_category,
    tensor_state_hash,
)


def test_gradient_ratio_categories_match_requested_thresholds() -> None:
    assert _category(0.2499) == "weak"
    assert _category(0.25) == "moderate"
    assert _category(0.75) == "comparable"
    assert _category(1.5) == "comparable"
    assert _category(1.5001) == "regularizer-dominant"


def test_gradient_cosine_categories_match_requested_thresholds() -> None:
    assert _cosine_category(0.3001) == "meaningfully aligned"
    assert _cosine_category(0.3) == "mostly orthogonal / weak relation"
    assert _cosine_category(-0.3) == "mostly orthogonal / weak relation"
    assert _cosine_category(-0.3001) == "meaningfully conflicting"


def test_state_hash_is_ordered_and_dtype_shape_sensitive() -> None:
    first = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3], dtype=torch.int64)}
    second = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3], dtype=torch.int64)}
    assert tensor_state_hash(first) == tensor_state_hash(second)
    assert tensor_state_hash({"b": first["b"], "a": first["a"]}) != tensor_state_hash(first)
    assert tensor_state_hash({"a": torch.tensor([[1.0, 2.0]]), "b": first["b"]}) != tensor_state_hash(first)
