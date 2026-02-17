"""Direction transform helpers."""
from __future__ import annotations

from typing import Iterable, Optional

import torch


def normalize(direction: torch.Tensor) -> torch.Tensor:
    return direction / (direction.norm() + 1e-6)


def topk(direction: torch.Tensor, k: int) -> torch.Tensor:
    k = max(1, int(k))
    indices = torch.topk(direction.abs(), k=k).indices
    mask = torch.zeros_like(direction, dtype=torch.bool)
    mask[indices] = True
    return direction * mask


def threshold(direction: torch.Tensor, value: float) -> torch.Tensor:
    mask = direction.abs() >= float(value)
    return direction * mask


def orthogonalize(direction: torch.Tensor, bases: Iterable[torch.Tensor]) -> torch.Tensor:
    out = direction.clone()
    for base in bases:
        base = base.to(out.device)
        denom = (base @ base) + 1e-6
        proj = (out @ base) / denom
        out = out - proj * base
    return out


def apply_transforms(
    direction: torch.Tensor,
    config: Optional[dict],
    *,
    orthogonalize_bases: Optional[Iterable[torch.Tensor]] = None,
) -> torch.Tensor:
    if not config:
        return direction
    out = direction
    effect_cfg = config.get("effect_filter")
    if effect_cfg is True:
        effect_cfg = {}
    if effect_cfg:
        scores = out.abs()
        if effect_cfg.get("topk") is not None:
            out = topk(out, int(effect_cfg["topk"]))
        elif effect_cfg.get("threshold") is not None:
            out = threshold(out, float(effect_cfg["threshold"]))
    if config.get("topk") is not None:
        out = topk(out, int(config["topk"]))
    if config.get("threshold") is not None:
        out = threshold(out, float(config["threshold"]))
    if config.get("orthogonalize"):
        bases = orthogonalize_bases or []
        if bases:
            out = orthogonalize(out, bases)
    if config.get("normalize", True):
        out = normalize(out)
    return out
