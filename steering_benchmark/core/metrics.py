"""Metrics for steering benchmarks."""
from __future__ import annotations

import re
import string
from typing import Callable, Dict, Iterable


def _normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _iter_targets(target):
    if target is None:
        return []
    if isinstance(target, (list, tuple, set)):
        return list(target)
    return [target]


def exact_match(pred: str, target) -> float:
    pred_norm = _normalize(pred)
    for tgt in _iter_targets(target):
        if pred_norm == _normalize(str(tgt)):
            return 1.0
    return 0.0


def contains(pred: str, target) -> float:
    pred_norm = _normalize(pred)
    for tgt in _iter_targets(target):
        if _normalize(str(tgt)) in pred_norm:
            return 1.0
    return 0.0


METRICS: Dict[str, Callable[[str, str], float]] = {
    "exact_match": exact_match,
    "contains": contains,
}
