from __future__ import annotations

import copy
import hashlib
from typing import Any

import torch
from torch import nn

from .lejepa_standard import StandardLeJEPA, Step


def _module_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        if value.dtype == torch.bfloat16:
            digest.update(value.view(torch.uint16).numpy().tobytes())
        else:
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class StandardPPCLeJEPA(StandardLeJEPA):
    """Standard LeJEPA plus a frozen-initial-projector functional drift penalty."""

    def __init__(self, c: dict[str, Any], device: torch.device) -> None:
        super().__init__(c, device)
        ppc = c["method"]["projector_plasticity"]
        self.ppc_lambda = float(ppc["lambda"])
        self.ppc_epsilon = float(ppc["epsilon"])
        if self.ppc_lambda < 0.0:
            raise ValueError("PPC lambda must be non-negative")
        if self.ppc_epsilon != 1e-8:
            raise ValueError("PPC epsilon must remain 1e-8")

        # True frozen snapshot: deepcopy immediately after standard LeJEPA
        # initialization, before any optimizer step or projector forward.
        self.reference_projector = copy.deepcopy(self.model.projector)
        self.reference_projector.requires_grad_(False)
        self.reference_projector.eval()

        self._initial_trainable_projector_sha256 = _module_sha256(self.model.projector)
        self._initial_reference_projector_sha256 = _module_sha256(self.reference_projector)
        if self._initial_trainable_projector_sha256 != self._initial_reference_projector_sha256:
            raise AssertionError("Frozen reference projector does not match trainable projector at initialization")

    @property
    def projector(self) -> nn.Module:
        return self.model.projector

    def train(self, mode: bool = True):
        result = super().train(mode)
        self.reference_projector.eval()
        return result

    def train_mode(self):
        super().train_mode()
        self.reference_projector.eval()

    def _trainable_projector_eval(self, features: torch.Tensor) -> torch.Tensor:
        """Evaluate g_phi without changing its BatchNorm running statistics."""
        was_training = self.projector.training
        self.projector.eval()
        try:
            return self.projector(features)
        finally:
            self.projector.train(was_training)

    def ppc_regularizer_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """Normalized functional drift on stop-gradient encoder features."""
        detached = features.detach()
        p = self._trainable_projector_eval(detached)
        self.reference_projector.eval()
        with torch.no_grad():
            p0 = self.reference_projector(detached)
        numerator = (p - p0).square().sum(dim=-1).mean()
        denominator = p0.square().sum(dim=-1).mean() + self.ppc_epsilon
        return numerator / denominator

    def training_step(self, batch, *, bf16: bool):
        # lambda=0 takes the exact standard LeJEPA path: no extra projector
        # forward, no altered BN behavior, and identical encoder/projector loss.
        if self.ppc_lambda == 0.0:
            base = super().training_step(batch, bf16=bf16)
            metrics = dict(base.metrics)
            metrics.update(
                {
                    "ppc_regularizer": 0.0,
                    "ppc_lambda": 0.0,
                    "total_loss": float(base.loss.detach()),
                    "ppc_encoder_grad_path": 0.0,
                }
            )
            return Step(base.loss, metrics)

        g = batch["global_views"].to(self.device, non_blocking=True)
        l = batch["local_views"].to(self.device, non_blocking=True)
        gs = [g[:, i] for i in range(g.shape[1])]
        ls = [l[:, i] for i in range(l.shape[1])]
        if len(gs) != 2 or len(ls) != 6:
            raise RuntimeError("Standard PPC-LeJEPA must remain 2G+6L")

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            # Compute standard LeJEPA encoder features once. PPC is evaluated
            # before the ordinary train-mode projector forward so the frozen
            # reference exactly matches g_phi at initialization.
            global_features = self.model._features(torch.cat(gs, dim=0))
            local_features = self.model._features(torch.cat(ls, dim=0))
            all_features = torch.cat((global_features, local_features), dim=0)

            ppc_regularizer = self.ppc_regularizer_from_features(all_features)

            # Ordinary LeJEPA objective remains unchanged and keeps the normal
            # encoder -> projector gradient path.
            projected = self.model.projector(all_features)
            batch_size = gs[0].shape[0]
            n_views = len(gs) + len(ls)
            projected = projected.view(n_views, batch_size, -1)
            center = projected[: len(gs)].mean(0)
            inv_loss = (center.unsqueeze(0) - projected).square().mean()
            sigreg_loss = self.model.sigreg(projected.reshape(-1, projected.size(-1)))
            lejepa_loss = inv_loss + self.model.lamb * sigreg_loss
            total_loss = lejepa_loss + self.ppc_lambda * ppc_regularizer

        return Step(
            total_loss,
            {
                "invariance_loss": float(inv_loss.detach()),
                "sigreg_loss": float(sigreg_loss.detach()),
                "ppc_regularizer": float(ppc_regularizer.detach()),
                "ppc_lambda": self.ppc_lambda,
                "total_loss": float(total_loss.detach()),
                "logical_views": 8.0,
                # The regularizer receives all_features.detach(), so there is
                # structurally no autograd path from PPC back into E_theta.
                "ppc_encoder_grad_path": float(all_features.detach().requires_grad),
            },
        )

    def checkpoint_metadata(self) -> dict[str, Any]:
        reference_sha = _module_sha256(self.reference_projector)
        trainable_sha = _module_sha256(self.projector)
        frozen = all(not p.requires_grad for p in self.reference_projector.parameters())
        reference_bn_eval = all(
            not module.training
            for module in self.reference_projector.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        )
        return {
            "method": "ppc_lejepa",
            "ppc_lambda": self.ppc_lambda,
            "ppc_epsilon": self.ppc_epsilon,
            "ppc_reference_initial_sha256": self._initial_reference_projector_sha256,
            "ppc_reference_projector_sha256": reference_sha,
            "ppc_reference_unchanged": reference_sha == self._initial_reference_projector_sha256,
            "ppc_reference_frozen": frozen,
            "ppc_reference_bn_eval": reference_bn_eval,
            "ppc_trainable_initial_sha256": self._initial_trainable_projector_sha256,
            "ppc_trainable_projector_sha256": trainable_sha,
            "ppc_trainable_changed_from_initial": trainable_sha != self._initial_trainable_projector_sha256,
            "ppc_encoder_stop_gradient": True,
        }

    def training_metadata(self) -> dict[str, Any]:
        return self.checkpoint_metadata()
