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
