"""GYAFC dataset loader."""
from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence

from datasets import load_dataset

from steering_benchmark.datasets.base import BaseDataset, Example
from steering_benchmark.datasets.hf_base import HFDatasetMixin
from steering_benchmark.registry import register_dataset


@register_dataset("gyafc")
class GYAFCDataset(BaseDataset, HFDatasetMixin):
    def __init__(
        self,
        split: str = "train",
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        revision: Optional[str] = None,
        dataset_name: str = "osyvokon/pavlick-formality-scores",
        config_name: Optional[str] = None,
        trust_remote_code: bool = False,
        seed: int = 42,
        split_ratios: Optional[Dict[str, float]] = None,
        split_counts: Optional[Dict[str, int]] = None,
        split_policy: str = "ratio",
        shuffle_splits: bool = True,
        split_names: Optional[Sequence[str]] = None,
        max_rows: Optional[int] = None,
        informal_field: str = "informal",
        formal_field: str = "formal",
        prompt_template: str = "rewrite to {style}: {text}",
        classification_template: str = "sentence: {text}\nformality:",
        formality_threshold: float = 0.0,
        formal_label: str = "formal",
        informal_label: str = "informal",
    ) -> None:
        if hf_token is None:
            hf_token = __import__("os").environ.get("HF_TOKEN") or __import__("os").environ.get("HUGGINGFACE_HUB_TOKEN")
        if config_name:
            dataset = load_dataset(
                dataset_name,
                config_name,
                split=split,
                cache_dir=cache_dir,
                token=hf_token,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )
        else:
            dataset = load_dataset(
                dataset_name,
                split=split,
                cache_dir=cache_dir,
                token=hf_token,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )
        if max_rows is not None:
            dataset = dataset.select(range(min(max_rows, len(dataset))))
        self._dataset = dataset
        self.informal_field = informal_field
        self.formal_field = formal_field
        self.prompt_template = prompt_template
        self.classification_template = classification_template
        self.formality_threshold = float(formality_threshold)
        self.formal_label = formal_label
        self.informal_label = informal_label
        features = set(self._dataset.features.keys())
        if informal_field in features and formal_field in features:
            self._mode = "rewrite"
        elif "sentence" in features and "avg_score" in features:
            self._mode = "score"
        else:
            self._mode = "rewrite"
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
            if self._mode == "score":
                sentence = row.get("sentence", "")
                score = float(row.get("avg_score", 0.0))
                is_formal = score >= self.formality_threshold
                if group == "formal" and not is_formal:
                    continue
                if group == "informal" and is_formal:
                    continue
                prompt = self.classification_template.format(text=sentence)
                target = self.formal_label if is_formal else self.informal_label
            else:
                informal = row.get(self.informal_field, "")
                formal = row.get(self.formal_field, "")
                if group == "formal":
                    prompt = self.prompt_template.format(style="formal", text=informal)
                    target = formal
                elif group == "informal":
                    prompt = self.prompt_template.format(style="informal", text=formal)
                    target = informal
                else:
                    raise KeyError(f"Unknown group: {group}")
            meta = {"row_id": str(idx)}
            yield Example(prompt=prompt, target=target, group=group, meta=meta)
            count += 1
            if limit is not None and count >= limit:
                break

    def groups(self) -> List[str]:
        return ["formal", "informal"]
