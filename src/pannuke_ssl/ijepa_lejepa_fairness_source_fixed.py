"""Source-faithful I-JEPA mask fallback patch for the fairness benchmark.

This module leaves the main benchmark implementation untouched and replaces only
its I-JEPA context-mask sampler with the retry/relaxation behavior used by
facebookresearch/ijepa's official multi-block mask collator.
"""
from __future__ import annotations

import torch

from . import ijepa_lejepa_fairness_source as _source


class IJEPAOfficialMaskSampler(_source.IJEPAOfficialMaskSampler):
    """Official I-JEPA retry semantics adapted to the benchmark's fixed 8x8 grid."""

    def _sample_mask(
        self,
        size: tuple[int, int],
        acceptable: list[torch.Tensor] | None,
        min_keep: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = size
        tries = 0
        timeout = original_timeout = 20

        while True:
            # Match facebookresearch/ijepa: randint high is exclusive and block
            # dimensions are guaranteed to be strictly smaller than the grid.
            top = int(
                torch.randint(
                    0,
                    self.height - h,
                    (1,),
                    generator=self.generator,
                )
            )
            left = int(
                torch.randint(
                    0,
                    self.width - w,
                    (1,),
                    generator=self.generator,
                )
            )

            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top : top + h, left : left + w] = 1

            if acceptable is not None:
                # Official fallback: after every 20 failed attempts, enforce one
                # fewer acceptable-region constraint. This prevents impossible
                # context sampling when several target blocks cover too much of
                # the coarse 8x8 patch grid.
                n_regions = max(int(len(acceptable) - tries), 0)
                for region in acceptable[:n_regions]:
                    mask *= region

            indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
            if len(indices) > min_keep:
                complement = torch.ones((self.height, self.width), dtype=torch.int32)
                complement[top : top + h, left : left + w] = 0
                return indices, complement

            timeout -= 1
            if timeout == 0:
                tries += 1
                timeout = original_timeout


    def sample(self, batch_size: int, device: torch.device | None = None):
        """Keep official target blocks, but reject relaxed contexts overlapping any target.

        Official fallback may drop exclusion constraints after repeated failures.
        On a coarse 8x8 grid that can leak prediction targets into student context.
        Resample per image, preserving common block dimensions within each batch.
        """
        p_size = self._block_size(self.pred_scale, self.aspect)
        e_size = self._block_size(self.enc_scale, (1.0, 1.0))
        all_context, all_targets = [], []
        for _ in range(batch_size):
            for attempt in range(64):
                targets, complements = [], []
                for _ in range(self.npred):
                    target, complement = self._sample_mask(p_size, None, self.block_min)
                    targets.append(target)
                    complements.append(complement)
                context, _ = self._sample_mask(e_size, complements, self.context_min)
                target_set = set(torch.cat(targets).tolist())
                if target_set.isdisjoint(context.tolist()):
                    all_context.append(context)
                    all_targets.append(targets)
                    break
            else:
                raise RuntimeError("Unable to obtain leak-free I-JEPA mask after 64 retries")
        min_context = min(len(context) for context in all_context)
        min_target = min(len(target) for targets in all_targets for target in targets)
        contexts = torch.stack([x[:min_context] for x in all_context]).to(device=device, dtype=torch.long)
        targets = torch.stack([
            torch.stack([x[:min_target] for x in blocks]) for blocks in all_targets
        ]).to(device=device, dtype=torch.long)
        assert not (contexts[:, None, :, None] == targets[:, :, None, :]).any().item()
        return contexts, targets


# Patch the module-level symbol used by BenchmarkStepper and smoke(). No existing
# training pipeline is modified; only this isolated fairness module is affected.
_source.IJEPAOfficialMaskSampler = IJEPAOfficialMaskSampler

SETTINGS = _source.SETTINGS
comparison = _source.comparison
require_smoke = _source.require_smoke
run_setting = _source.run_setting
smoke = _source.smoke
SlicedEppsPulley = _source.SlicedEppsPulley
IJEPAFairPredictor = _source.IJEPAFairPredictor
