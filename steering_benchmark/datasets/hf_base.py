"""Shared helpers for HuggingFace datasets."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence


class HFDatasetMixin:
    def _build_splits(
        self,
        total: int,
        seed: int,
        split_ratios: Optional[Dict[str, float]],
        split_counts: Optional[Dict[str, int]],
        split_policy: str,
        split_names: Sequence[str],
        shuffle_splits: bool,
    ) -> Dict[str, List[int]]:
        indices = list(range(total))
        if shuffle_splits:
            rng = __import__("random")
            rng.seed(seed)
            rng.shuffle(indices)

        if split_policy in {"shared", "none"}:
            return {name: list(indices) for name in split_names}

        if split_counts:
            splits: Dict[str, List[int]] = {}
            offset = 0
            for name, count in split_counts.items():
                splits[name] = indices[offset : offset + int(count)]
                offset += int(count)
            if offset < total:
                splits["unused"] = indices[offset:]
            return splits

        if not split_ratios:
            split_ratios = {"train": 0.6, "eval": 0.2, "test": 0.2}

        splits = {}
        offset = 0
        for name, ratio in split_ratios.items():
            count = int(round(ratio * total))
            splits[name] = indices[offset : offset + count]
            offset += count
        if offset < total:
            splits["unused"] = indices[offset:]
        return splits
