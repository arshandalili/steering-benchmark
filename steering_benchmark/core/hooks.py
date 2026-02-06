"""Model hook helpers for interventions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class HookHandle:
    handle: Any

    def remove(self) -> None:
        self.handle.remove()


class LayerHook:
    def __init__(self, module, intervention) -> None:
        self.module = module
        self.intervention = intervention
        self.handle = None

    def _hook(self, _module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
            hidden = self.intervention.apply(hidden)
            return (hidden,) + rest
        if isinstance(output, torch.Tensor):
            return self.intervention.apply(output)
        return output

    def attach(self) -> HookHandle:
        self.handle = self.module.register_forward_hook(self._hook)
        return HookHandle(self.handle)
