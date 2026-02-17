"""BeaverTails dataset loader."""
from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence

from datasets import load_dataset

from steering_benchmark.datasets.base import BaseDataset, Example
from steering_benchmark.datasets.hf_base import HFDatasetMixin
from steering_benchmark.registry import register_dataset


@register_dataset("beavertails")
class BeaverTailsDataset(BaseDataset, HFDatasetMixin):
    def __init__(
        self,
        split: str = "train",
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        revision: Optional[str] = None,
        seed: int = 42,
        split_ratios: Optional[Dict[str, float]] = None,
        split_counts: Optional[Dict[str, int]] = None,
        split_policy: str = "ratio",
        shuffle_splits: bool = True,
        split_names: Optional[Sequence[str]] = None,
        max_rows: Optional[int] = None,
        prompt_field: str = "prompt",
        safe_label: str = "safe",
        unsafe_label: str = "unsafe",
    ) -> None:
        if hf_token is None:
            hf_token = __import__("os").environ.get("HF_TOKEN") or __import__("os").environ.get("HUGGINGFACE_HUB_TOKEN")
        dataset = load_dataset(
            "PKU-Alignment/BeaverTails",
            split=split,
            cache_dir=cache_dir,
            token=hf_token,
            revision=revision,
        )
        if max_rows is not None:
            dataset = dataset.select(range(min(max_rows, len(dataset))))
        self._dataset = dataset
        self.prompt_field = prompt_field
        self.safe_label = safe_label
        self.unsafe_label = unsafe_label
        self.split_names = list(split_names) if split_names is not None else ["train", "eval", "test"]
        self._indices_by_split = self._build_splits(
            total=len(self._dataset),
            seed=seed,
            split_ratios=split_ratios,
            split_counts=split_counts,
            split_policy=split_policy,
            split_names=self.split_names,
            shuffle_splits=shuffle_splits,
        )

    def _get_indices(self, split: Optional[str]) -> List[int]:
        if split is None:
            if "train" in self._indices_by_split:
                return self._indices_by_split["train"]
            return list(range(len(self._dataset)))
        if split not in self._indices_by_split:
            raise KeyError(f"Unknown split: {split}")
        return self._indices_by_split[split]

    def _is_safe(self, row: Dict) -> bool:
        if "is_safe" in row:
            return bool(row["is_safe"])
        if "label" in row:
            return str(row["label"]).lower() == self.safe_label
        if "safety" in row:
            return str(row["safety"]).lower() == self.safe_label
        return False

    def iter_group(
        self,
        group: str,
        limit: Optional[int] = None,
        split: Optional[str] = None,
    ) -> Iterator[Example]:
        indices = self._get_indices(split)
        count = 0
        for idx in indices:
            row = self._dataset[int(idx)]
            is_safe = self._is_safe(row)
            if group == "safe" and not is_safe:
                continue
            if group == "unsafe" and is_safe:
                continue
            prompt = row.get(self.prompt_field, "")
            target = "comply" if is_safe else "refuse"
            meta = {"row_id": str(idx)}
            yield Example(prompt=prompt, target=target, group=group, meta=meta)
            count += 1
            if limit is not None and count >= limit:
                break

    def groups(self) -> List[str]:
        return ["safe", "unsafe"]
