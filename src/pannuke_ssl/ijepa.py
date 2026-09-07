"""I-JEPA objective on the existing fresh CLIP vision architecture.

The requested adaptation enforces target-target disjointness as well as
context-target disjointness. Official I-JEPA only enforces the latter.
All masking occurs BEFORE encoder attention; clean downstream encoding uses
the unchanged FreshPLIPVisionEncoder.forward implementation.
"""
from __future__ import annotations

import math
import random
import torch
from torch import nn
from torch.nn import functional as F

from .models import FreshPLIPVisionEncoder, normalize_clip, make_teacher


class BlockMasks:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.positions = {}
        for h, w in ((3, 3), (3, 4), (4, 3), (4, 2), (7, 7), (8, 8)):
            self.positions[h, w] = [tuple(r * 8 + c for r in range(y, y+h)
                                          for c in range(x, x+w))
                                    for y in range(9-h) for x in range(9-w)]

    def sample(self, batch_size: int, device=None):
        # Sample scales/aspect independently; quantize rectangles on the 8x8 grid.
        area = int(64 * self.rng.uniform(.15, .20))
        aspect = self.rng.uniform(.75, 1.5)
        h, w = round(math.sqrt(area * aspect)), round(math.sqrt(area / aspect))
        side = round(math.sqrt(int(64 * self.rng.uniform(.85, 1.0))))
        contexts, targets = [], []
        for _ in range(batch_size):
            for attempt in range(1000):
                used, blocks = set(), []
                for _ in range(4):
                    candidates = [p for p in self.positions[h, w] if used.isdisjoint(p)]
                    if not candidates:
                        break
                    p = self.rng.choice(candidates)
                    blocks.append(p)
                    used.update(p)
                if len(blocks) != 4:
                    continue
                valid_contexts = [tuple(i for i in p if i not in used)
                                  for p in self.positions[side, side]]
                valid_contexts = [p for p in valid_contexts if len(p) >= 10]
                if valid_contexts:
                    contexts.append(self.rng.choice(valid_contexts))
                    targets.append(blocks)
                    break
            else:
                raise RuntimeError("Cannot sample the requested disjoint 8x8 masks")
        # Randomly trim to batch-minimum length rather than prefer upper-left tokens.
        length = min(map(len, contexts))
        contexts = [sorted(self.rng.sample(list(p), length)) for p in contexts]
        return (torch.tensor(contexts, dtype=torch.long, device=device),
                torch.tensor(targets, dtype=torch.long, device=device))


def gather_tokens(tokens, indices):
    return tokens.gather(1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))


def encode_context(encoder: FreshPLIPVisionEncoder, images, indices):
    vision = encoder.model.vision_model
    embedded = vision.embeddings(normalize_clip(images))
    # CLS has no image content before attention, so keeping it leaks no target pixels.
    visible = torch.cat((embedded[:, :1], gather_tokens(embedded[:, 1:], indices)), dim=1)
    visible = vision.pre_layrnorm(visible)
    encoded = vision.encoder(inputs_embeds=visible, return_dict=True).last_hidden_state
    return vision.post_layernorm(encoded[:, 1:])


def positional_grid(width=384):
    y, x = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
    frequency = 1 / (10000 ** (torch.arange(width // 4).float() / (width // 4)))
    components = []
    for axis in (x.flatten(), y.flatten()):
        phase = axis.float()[:, None] * frequency[None]
        components.extend((phase.sin(), phase.cos()))
    return torch.cat(components, dim=1).unsqueeze(0)


class IJEPA_Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(768, 384)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 384))
        nn.init.trunc_normal_(self.mask_token, std=.02)
        self.register_buffer("position", positional_grid())
        layer = nn.TransformerEncoderLayer(384, 6, 1536, dropout=0,
                                          activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(384)
        self.output_projection = nn.Linear(384, 768)

    def forward(self, tokens, context_indices, target_indices):
        b, blocks, target_length = target_indices.shape
        positions = self.position.expand(b, -1, -1)
        context = self.input_projection(tokens) + gather_tokens(positions, context_indices)
        context = context[:, None].expand(-1, blocks, -1, -1).reshape(b*blocks, tokens.shape[1], 384)
        queries = self.mask_token + gather_tokens(positions, target_indices.flatten(1)).reshape(b*blocks, target_length, 384)
        predicted = self.transformer(torch.cat((context, queries), dim=1))[:, -target_length:]
        return self.output_projection(self.final_norm(predicted)).reshape(b, blocks, target_length, 768)


def build_models(config, device):
    student = FreshPLIPVisionEncoder(config["plip_config_dir"], 256).to(device)
    assert student.num_patches == 64 and student.hidden_size == 768
    return student, make_teacher(student), IJEPA_Predictor().to(device)


def objective(student, teacher, predictor, images, context, targets):
    visible = encode_context(student, images, context)
    prediction = predictor(visible, context, targets)
    with torch.no_grad():
        clean_targets = F.layer_norm(teacher(images).float(), (768,))
        target = gather_tokens(clean_targets, targets.flatten(1)).reshape_as(prediction)
    return F.smooth_l1_loss(prediction.float(), target.float()), prediction, target
