"""Flores200 dataset loader."""
from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence

from datasets import load_dataset

from steering_benchmark.datasets.base import BaseDataset, Example
from steering_benchmark.datasets.hf_base import HFDatasetMixin
from steering_benchmark.registry import register_dataset


@register_dataset("flores200")
class Flores200Dataset(BaseDataset, HFDatasetMixin):
    def __init__(
        self,
        split: str = "dev",
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        revision: Optional[str] = None,
        dataset_name: str = "Muennighoff/flores200",
        config_name: Optional[str] = None,
        trust_remote_code: bool = True,
        seed: int = 42,
        split_ratios: Optional[Dict[str, float]] = None,
        split_counts: Optional[Dict[str, int]] = None,
        split_policy: str = "ratio",
        shuffle_splits: bool = True,
        split_names: Optional[Sequence[str]] = None,
        max_rows: Optional[int] = None,
        source_lang: str = "eng_Latn",
        target_lang: str = "spa_Latn",
        prompt_template: str = "translate to {target_lang}: {text}",
        copy_template: str = "repeat: {text}",
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
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.prompt_template = prompt_template
        self.copy_template = copy_template
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
            src = row.get(self.source_lang, "")
            tgt = row.get(self.target_lang, "")
            if group == "translate":
                prompt = self.prompt_template.format(target_lang=self.target_lang, text=src)
                target = tgt
                meta = {"row_id": str(idx), "target_lang": self.target_lang}
            elif group == "copy":
                prompt = self.copy_template.format(text=src)
                target = src
                meta = {"row_id": str(idx), "target_lang": self.source_lang}
            else:
                raise KeyError(f"Unknown group: {group}")
            yield Example(prompt=prompt, target=target, group=group, meta=meta)
            count += 1
            if limit is not None and count >= limit:
                break

    def groups(self) -> List[str]:
        return ["translate", "copy"]
