"""Metrics for steering benchmarks."""
from __future__ import annotations

import re
from typing import Callable, Dict


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(pred: str, target: str) -> float:
    return float(_normalize(pred) == _normalize(target))


def contains(pred: str, target: str) -> float:
    return float(_normalize(target) in _normalize(pred))


METRICS: Dict[str, Callable[[str, str], float]] = {
    "exact_match": exact_match,
    "contains": contains,
}
