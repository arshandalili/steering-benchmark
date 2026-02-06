"""Model adapter interface."""
from __future__ import annotations

from typing import Iterable, List, Optional

import torch


class BaseModelAdapter:
    def encode_hidden(
        self,
        prompts: List[str],
        layer: int,
        token_position: str | int,
    ) -> torch.Tensor:
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        intervention=None,
        gen_cfg: Optional[dict] = None,
    ) -> str:
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        raise NotImplementedError
