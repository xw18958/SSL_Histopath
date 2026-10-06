"""Versioned caption-aware retrieval and strict PLIP asset validation."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

RETRIEVAL_VERSION = "caption_aware_v2_20261006"
REFERENCE_CAPTIONS = ["a normal tissue", "a cancerous tissue"]
REFERENCE_TOKENS = [
    [49406, 320, 5967, 15368, 49407],
    [49406, 320, 12690, 879, 15368, 49407],
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_plip_assets(path: Path):
    path = Path(path)
    names = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
    missing = [name for name in names if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete PLIP assets at {path}: missing {missing}")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.vocab_size != 49408:
        raise RuntimeError(f"Invalid PLIP tokenizer vocabulary: {tokenizer.vocab_size}, expected 49408")
    tokens = tokenizer(REFERENCE_CAPTIONS, padding=False, truncation=True, max_length=77)["input_ids"]
    if tokens != REFERENCE_TOKENS:
        raise RuntimeError("PLIP tokenizer does not match the known reference tokenization")
    identity = {name: file_sha256(path / name) for name in names}
    identity["vocab_size"] = tokenizer.vocab_size
    return tokenizer, identity


def caption_groups(texts: list[str]) -> torch.Tensor:
    # Preserve the exact stored text: no fuzzy, case, punctuation, or prefix merging.
    lookup = {}
    ids = []
    for text in texts:
        if not text.strip():
            raise ValueError("Empty retrieval caption")
        ids.append(lookup.setdefault(text, len(lookup)))
    return torch.tensor(ids, dtype=torch.long)


def unique_caption_indices(groups: torch.Tensor):
    if groups.ndim != 1 or groups.numel() == 0:
        raise ValueError("Caption groups must be a nonempty one-dimensional tensor")
    unique, inverse = torch.unique(groups, sorted=True, return_inverse=True)
    indices = torch.arange(groups.numel(), device=groups.device)
    first = torch.full((len(unique),), groups.numel(), device=groups.device, dtype=torch.long)
    first.scatter_reduce_(0, inverse, indices, reduce="amin", include_self=True)
    return first, inverse


def multi_positive_clip_loss(image_emb, text_emb, groups, logit_scale):
    if len(image_emb) != len(text_emb) or len(groups) != len(image_emb):
        raise ValueError("Image, caption, and group lengths differ")
    first, inverse = unique_caption_indices(groups)
    logits = logit_scale * (F.normalize(image_emb, dim=-1) @ F.normalize(text_emb[first], dim=-1).T)
    image_loss = F.cross_entropy(logits, inverse)
    positive = inverse[None, :] == torch.arange(len(first), device=groups.device)[:, None]
    log_probs = F.log_softmax(logits.T, dim=-1)
    text_loss = (-(log_probs * positive).sum(1) / positive.sum(1)).mean()
    return .5 * (image_loss + text_loss)


def _tie_averaged_hit(scores: torch.Tensor, positive: torch.Tensor, k: int):
    """Expected any-positive hit under uniform ordering inside an exact-score tie."""
    k = min(k, scores.shape[1])
    cutoff = scores.topk(k, dim=1).values[:, -1:]
    above = scores > cutoff
    tied = scores == cutoff
    hit_above = (above & positive).any(1)
    slots = k - above.sum(1)
    n = tied.sum(1).double()
    m = (tied & positive).sum(1).double()
    no_hit = torch.ones_like(n)
    for j in range(k):
        factor = ((n - m - j) / (n - j).clamp_min(1)).clamp(0, 1)
        no_hit = no_hit * torch.where(slots > j, factor, torch.ones_like(factor))
    return torch.where(hit_above, torch.ones_like(no_hit), 1 - no_hit).mean().item()


def caption_aware_metrics(image_emb, text_emb, groups):
    image_emb, text_emb, groups = image_emb.float().cpu(), text_emb.float().cpu(), groups.cpu()
    if len(image_emb) != len(text_emb) or len(groups) != len(image_emb):
        raise ValueError("Image, caption, and group lengths differ")
    if not torch.isfinite(image_emb).all() or not torch.isfinite(text_emb).all():
        raise ValueError("Nonfinite retrieval features")
    first, inverse = unique_caption_indices(groups)
    sim = F.normalize(image_emb, dim=-1) @ F.normalize(text_emb[first], dim=-1).T
    positive = inverse[:, None] == torch.arange(len(first))[None, :]
    result = {}
    for direction, scores, relevant in (("i2t", sim, positive), ("t2i", sim.T, positive.T)):
        recalls = []
        for k in (1, 5, 10):
            recall = _tie_averaged_hit(scores, relevant, k)
            result[f"{direction}_r@{k}"] = recall
            recalls.append(recall)
        result[f"{direction}_mean_recall"] = sum(recalls) / 3
    result["overall_mean_recall"] = (result["i2t_mean_recall"] + result["t2i_mean_recall"]) / 2
    return result


def caption_aware_chance(groups):
    first, inverse = unique_caption_indices(groups.cpu())
    counts = torch.bincount(inverse).double()
    n, c = len(groups), len(first)
    result = {}
    for k in (1, 5, 10):
        result[f"i2t_r@{k}"] = min(k, c) / c
        no_hit = torch.ones_like(counts)
        for j in range(min(k, n)):
            no_hit *= ((n - counts - j) / (n - j)).clamp(0, 1)
        result[f"t2i_r@{k}"] = (1 - no_hit).mean().item()
    for direction in ("i2t", "t2i"):
        result[f"{direction}_mean_recall"] = sum(result[f"{direction}_r@{k}"] for k in (1, 5, 10)) / 3
    result["overall_mean_recall"] = (result["i2t_mean_recall"] + result["t2i_mean_recall"]) / 2
    return result
