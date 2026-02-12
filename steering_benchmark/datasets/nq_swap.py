"""HuggingFace NQ-Swap dataset loader."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

from datasets import load_dataset

from steering_benchmark.datasets.base import BaseDataset, Example
from steering_benchmark.registry import register_dataset


def _normalize_answer(value, policy: str) -> Union[str, Sequence[str]]:
    if value is None:
        return ""
    if isinstance(value, list):
        if not value:
            return ""
        if policy in {"list", "all", "raw"}:
            return [str(v) for v in value]
        if policy == "first":
            return str(value[0])
        if policy == "join":
            return " / ".join(str(v) for v in value)
    return str(value)


@register_dataset("nq_swap")
class NQSwapDataset(BaseDataset):
    def __init__(
        self,
        split: str = "dev",
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        seed: int = 42,
        split_ratios: Optional[Dict[str, float]] = None,
        split_counts: Optional[Dict[str, int]] = None,
        split_policy: str = "ratio",
        shuffle_splits: bool = True,
        split_names: Optional[Sequence[str]] = None,
        max_rows: Optional[int] = None,
        k_shot: int = 0,
        demo_pool_size: int = 256,
        demonstrations_org_context: bool = False,
        demonstrations_org_answer: bool = True,
        test_example_org_context: bool = False,
        context_template: str = "context: {context}\nquestion: {question}\nanswer:",
        param_template: str = "question: {question}\nanswer:",
        context_field: str = "sub_context",
        org_context_field: str = "org_context",
        question_field: str = "question",
        context_answer_field: str = "sub_answer",
        param_answer_field: str = "org_answer",
        answer_policy: str = "first",
    ) -> None:
        self.context_template = context_template
        self.param_template = param_template
        self.context_field = context_field
        self.org_context_field = org_context_field
        self.question_field = question_field
        self.context_answer_field = context_answer_field
        self.param_answer_field = param_answer_field
        self.answer_policy = answer_policy
        self.split_policy = split_policy
        self.shuffle_splits = shuffle_splits
        self.split_names = list(split_names) if split_names is not None else ["train", "eval", "test"]
        self.k_shot = int(k_shot)
        self.demo_pool_size = int(demo_pool_size)
        self.demonstrations_org_context = demonstrations_org_context
        self.demonstrations_org_answer = demonstrations_org_answer
        self.test_example_org_context = test_example_org_context

        if hf_token is None:
            hf_token = __import__("os").environ.get("HF_TOKEN") or __import__("os").environ.get("HUGGINGFACE_HUB_TOKEN")
        dataset = load_dataset(
            "pminervini/NQ-Swap",
            split=split,
            cache_dir=cache_dir,
            token=hf_token,
        )
        if max_rows is not None:
            dataset = dataset.select(range(min(max_rows, len(dataset))))

        self._dataset = dataset
        self._indices_by_split = self._build_splits(
            seed=seed,
            split_ratios=split_ratios,
            split_counts=split_counts,
        )
        self._demo_prefix_with_ctx, self._demo_prefix_without_ctx = self._build_demos(seed)

    def _build_splits(
        self,
        seed: int,
        split_ratios: Optional[Dict[str, float]],
        split_counts: Optional[Dict[str, int]],
    ) -> Dict[str, List[int]]:
        total = len(self._dataset)
        indices = list(range(total))
        if self.shuffle_splits:
            rng = __import__("random")
            rng.seed(seed)
            rng.shuffle(indices)

        if self.split_policy in {"shared", "sae", "none"}:
            return {name: list(indices) for name in self.split_names}

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

    def _build_demos(self, seed: int) -> Tuple[str, str]:
        if self.k_shot <= 0:
            return "", ""

        total = len(self._dataset)
        start = max(0, total - self.demo_pool_size)
        pool = [self._dataset[i] for i in range(start, total)]
        try:
            import numpy as np

            rng = np.random.RandomState(seed)
            rng.shuffle(pool)
        except Exception:
            rng = __import__("random")
            rng.seed(seed)
            rng.shuffle(pool)
        demos = pool[: self.k_shot]

        demo_ctx_field = self.org_context_field if self.demonstrations_org_context else self.context_field
        demo_ans_field = self.param_answer_field if self.demonstrations_org_answer else self.context_answer_field

        with_ctx_parts = []
        without_ctx_parts = []
        for demo in demos:
            demo_ctx_answer = _normalize_answer(demo.get(demo_ans_field), "first")
            with_ctx_parts.append(
                self.context_template.format(
                    context=demo.get(demo_ctx_field, ""),
                    question=demo.get(self.question_field, ""),
                )
                + " "
                + str(demo_ctx_answer)
                + "\n\n"
            )
            without_ctx_parts.append(
                self.param_template.format(question=demo.get(self.question_field, ""))
                + " "
                + str(_normalize_answer(demo.get(self.param_answer_field), "first"))
                + "\n\n"
            )
        return "".join(with_ctx_parts), "".join(without_ctx_parts)

    def _get_indices(self, split: Optional[str]) -> List[int]:
        if split is None:
            if "train" in self._indices_by_split:
                return self._indices_by_split["train"]
            return list(range(len(self._dataset)))
        if split not in self._indices_by_split:
            raise KeyError(f"Unknown split: {split}")
        return self._indices_by_split[split]

    def _build_prompt(self, row: Dict[str, str], template: str, context_field: Optional[str] = None) -> str:
        data = {
            "question": row.get(self.question_field, ""),
            "context": row.get(context_field or self.context_field, ""),
        }
        return template.format(**data)

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
            question = row.get(self.question_field, "")
            if group == "context":
                test_ctx_field = self.org_context_field if self.test_example_org_context else self.context_field
                prompt = self._build_prompt(row, self.context_template, test_ctx_field)
                if self._demo_prefix_with_ctx:
                    prompt = self._demo_prefix_with_ctx + prompt
                target = _normalize_answer(row.get(self.context_answer_field), self.answer_policy)
                alt_target = _normalize_answer(row.get(self.param_answer_field), self.answer_policy)
                targets = {"param": alt_target}
            elif group == "param":
                prompt = self.param_template.format(question=question)
                if self._demo_prefix_without_ctx:
                    prompt = self._demo_prefix_without_ctx + prompt
                target = _normalize_answer(row.get(self.param_answer_field), self.answer_policy)
                alt_target = _normalize_answer(row.get(self.context_answer_field), self.answer_policy)
                targets = {"context": alt_target}
            else:
                raise KeyError(f"Unknown group: {group}")

            meta = {"row_id": str(idx)}
            yield Example(prompt=prompt, target=target, group=group, meta=meta, targets=targets)
            count += 1
            if limit is not None and count >= limit:
                break

    def iter_pairs(
        self,
        group_pos: str,
        group_neg: str,
        limit: Optional[int] = None,
        split: Optional[str] = None,
    ) -> Iterator[Tuple[Example, Example]]:
        indices = self._get_indices(split)
        count = 0
        for idx in indices:
            row = self._dataset[int(idx)]
            question = row.get(self.question_field, "")

            test_ctx_field = self.org_context_field if self.test_example_org_context else self.context_field
            ctx_prompt = self._build_prompt(row, self.context_template, test_ctx_field)
            if self._demo_prefix_with_ctx:
                ctx_prompt = self._demo_prefix_with_ctx + ctx_prompt
            param_prompt = self.param_template.format(question=question)
            if self._demo_prefix_without_ctx:
                param_prompt = self._demo_prefix_without_ctx + param_prompt

            ctx_target = _normalize_answer(row.get(self.context_answer_field), self.answer_policy)
            param_target = _normalize_answer(row.get(self.param_answer_field), self.answer_policy)

            ctx_example = Example(
                prompt=ctx_prompt,
                target=ctx_target,
                group="context",
                meta={"row_id": str(idx)},
                targets={"param": param_target},
            )
            param_example = Example(
                prompt=param_prompt,
                target=param_target,
                group="param",
                meta={"row_id": str(idx)},
                targets={"context": ctx_target},
            )

            if group_pos == "context" and group_neg == "param":
                yield ctx_example, param_example
            elif group_pos == "param" and group_neg == "context":
                yield param_example, ctx_example
            else:
                raise KeyError(f"Unknown pair groups: {group_pos}/{group_neg}")

            count += 1
            if limit is not None and count >= limit:
                break

    def groups(self) -> List[str]:
        return ["context", "param"]
