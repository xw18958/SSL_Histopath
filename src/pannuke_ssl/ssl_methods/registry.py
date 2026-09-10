from __future__ import annotations
import torch

def build_method(config: dict, device: torch.device):
    name=config["method"]["name"]
    if name=="ijepa":
        from .ijepa_standard import StandardIJEPA; return StandardIJEPA(config,device)
    if name=="lejepa":
        from .lejepa_standard import StandardLeJEPA; return StandardLeJEPA(config,device)
    raise KeyError(f"Unknown SSL method {name!r}; add one adapter and register it here")
