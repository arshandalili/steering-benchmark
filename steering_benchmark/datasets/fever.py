"""FEVER dataset loader."""
from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence

from datasets import load_dataset

from steering_benchmark.datasets.base import BaseDataset, Example
from steering_benchmark.datasets.hf_base import HFDatasetMixin
from steering_benchmark.registry import register_dataset


@register_dataset("fever")
class FeverDataset(BaseDataset, HFDatasetMixin):
    def __init__(
        self,
        split: str = "train",
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        revision: Optional[str] = None,
        trust_remote_code: bool = True,
        seed: int = 42,
        split_ratios: Optional[Dict[str, float]] = None,
        split_counts: Optional[Dict[str, int]] = None,
        split_policy: str = "ratio",
        shuffle_splits: bool = True,
        split_names: Optional[Sequence[str]] = None,
        max_rows: Optional[int] = None,
        prompt_template: str = "evidence: {evidence}\nclaim: {claim}\nlabel:",
    ) -> None:
        if hf_token is None:
            hf_token = __import__("os").environ.get("HF_TOKEN") or __import__("os").environ.get("HUGGINGFACE_HUB_TOKEN")
        dataset = load_dataset(
            "fever",
            split=split,
            cache_dir=cache_dir,
            token=hf_token,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        if max_rows is not None:
            dataset = dataset.select(range(min(max_rows, len(dataset))))
        self._dataset = dataset
        self.prompt_template = prompt_template
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

    def _render_evidence(self, evidence) -> str:
        if not evidence:
            return ""
        # evidence is list of lists; flatten to strings
        parts = []
        for group in evidence:
            for item in group:
                if len(item) >= 4:
                    parts.append(str(item[3]))
        return " ".join(parts)

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
            label = row.get("label", "")
            if group == "supports" and label != "SUPPORTS":
                continue
            if group == "refutes" and label != "REFUTES":
                continue
            claim = row.get("claim", "")
            evidence = self._render_evidence(row.get("evidence"))
            prompt = self.prompt_template.format(evidence=evidence, claim=claim)
            target = "supports" if label == "SUPPORTS" else "refutes"
            meta = {"row_id": str(idx), "context": evidence}
            yield Example(prompt=prompt, target=target, group=group, meta=meta)
            count += 1
            if limit is not None and count >= limit:
                break

    def groups(self) -> List[str]:
        return ["supports", "refutes"]
