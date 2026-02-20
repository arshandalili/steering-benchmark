"""HuggingFace MACNoise dataset loader."""
from __future__ import annotations

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


@register_dataset("macnoise")
class MACNoiseDataset(BaseDataset):
    def __init__(
        self,
        dataset_name: str = "GWHed/dataset_macnoise",
        split: str = "train_chatgpt",
        demonstrations_split: str = "train",
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        revision: Optional[str] = None,
        seed: int = 42,
        split_ratios: Optional[Dict[str, float]] = None,
        split_counts: Optional[Dict[str, int]] = None,
        split_policy: str = "ratio",
        shuffle_splits: bool = False,
        split_names: Optional[Sequence[str]] = None,
        max_rows: Optional[int] = 5120,
        k_shot: int = 4,
        demo_pool_size: Optional[int] = None,
        demonstrations_org_context: bool = True,
        demonstrations_org_answer: bool = True,
        test_example_org_context: bool = False,
        context_template: str = "context: {context}\nquestion: {question}\nanswer:",
        param_template: str = "question: {question}\nanswer:",
        context_template_with_answer: str = "context: {context}\nquestion: {question}\nanswer: {answer}",
        param_template_with_answer: str = "question: {question}\nanswer: {answer}",
        context_field: str = "sub_context",
        org_context_field: str = "org_context",
        param_context_field: Optional[str] = None,
        question_field: str = "question",
        context_answer_field: str = "sub_answer",
        param_answer_field: str = "org_answer",
        answer_policy: str = "list",
        train_use_answer: bool = False,
        train_include_param_context: bool = False,
    ) -> None:
        self.context_template = context_template
        self.param_template = param_template
        self.context_template_with_answer = context_template_with_answer
        self.param_template_with_answer = param_template_with_answer
        self.context_field = context_field
        self.org_context_field = org_context_field
        self.param_context_field = param_context_field or org_context_field
        self.question_field = question_field
        self.context_answer_field = context_answer_field
        self.param_answer_field = param_answer_field
        self.answer_policy = answer_policy
        self.train_use_answer = bool(train_use_answer)
        self.train_include_param_context = bool(train_include_param_context)
        self.split_policy = split_policy
        self.shuffle_splits = shuffle_splits
        self.split_names = list(split_names) if split_names is not None else ["train", "eval", "test"]
        self.k_shot = int(k_shot)
        self.demo_pool_size = int(demo_pool_size) if demo_pool_size is not None else None
        self.demonstrations_org_context = demonstrations_org_context
        self.demonstrations_org_answer = demonstrations_org_answer
        self.test_example_org_context = test_example_org_context

        if hf_token is None:
            hf_token = __import__("os").environ.get("HF_TOKEN") or __import__("os").environ.get("HUGGINGFACE_HUB_TOKEN")
        dataset = load_dataset(
            dataset_name,
            cache_dir=cache_dir,
            token=hf_token,
            revision=revision,
        )
        if split not in dataset:
            raise KeyError(f"Unknown split: {split}. Available splits: {list(dataset.keys())}")
        if demonstrations_split not in dataset:
            raise KeyError(
                f"Unknown demonstrations_split: {demonstrations_split}. Available splits: {list(dataset.keys())}"
            )

        eval_dataset = dataset[split]
        if max_rows is not None and int(max_rows) >= 0:
            eval_dataset = eval_dataset.select(range(min(int(max_rows), len(eval_dataset))))

        self._dataset = eval_dataset
        self._demonstration_pool = [dataset[demonstrations_split][i] for i in range(len(dataset[demonstrations_split]))]
        self._indices_by_split = self._build_splits(
            seed=seed,
            split_ratios=split_ratios,
            split_counts=split_counts,
        )
        self._selected_demos = self._select_demos(seed)

        demo_ctx_field = self.org_context_field if self.demonstrations_org_context else self.context_field
        demo_ans_field = self.param_answer_field if self.demonstrations_org_answer else self.context_answer_field
        self._demo_prefix_with_ctx = self._render_context_demos(
            demos=self._selected_demos,
            demo_ctx_field=demo_ctx_field,
            demo_ans_field=demo_ans_field,
        )
        self._demo_prefix_without_ctx = self._render_param_demos(self._selected_demos)

        # SpARE-style prompt-control variants for prompt baseline steering.
        self._demo_prefix_context_control = self._render_context_demos(
            demos=self._selected_demos,
            demo_ctx_field=self.context_field,
            demo_ans_field=self.context_answer_field,
        )
        self._demo_prefix_param_control = self._render_context_demos(
            demos=self._selected_demos,
            demo_ctx_field=self.org_context_field,
            demo_ans_field=self.param_answer_field,
        )

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

    def _select_demos(self, seed: int) -> List[Dict]:
        if self.k_shot <= 0:
            return []
        pool = list(self._demonstration_pool)
        try:
            import numpy as np

            rng = np.random.RandomState(seed)
            rng.shuffle(pool)
        except Exception:
            rng = __import__("random")
            rng.seed(seed)
            rng.shuffle(pool)
        if self.demo_pool_size is not None and self.demo_pool_size > 0:
            pool = pool[: self.demo_pool_size]
        return pool[: self.k_shot]

    def _render_context_demos(
        self,
        demos: Sequence[Dict],
        demo_ctx_field: str,
        demo_ans_field: str,
    ) -> str:
        with_ctx_parts = []
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
        return "".join(with_ctx_parts)

    def _render_param_demos(self, demos: Sequence[Dict]) -> str:
        without_ctx_parts = []
        for demo in demos:
            without_ctx_parts.append(
                self.param_template.format(question=demo.get(self.question_field, ""))
                + " "
                + str(_normalize_answer(demo.get(self.param_answer_field), "first"))
                + "\n\n"
            )
        return "".join(without_ctx_parts)

    def _get_indices(self, split: Optional[str]) -> List[int]:
        if split is None:
            if "train" in self._indices_by_split:
                return self._indices_by_split["train"]
            return list(range(len(self._dataset)))
        if split not in self._indices_by_split:
            raise KeyError(f"Unknown split: {split}")
        return self._indices_by_split[split]

    def _build_prompt(
        self,
        row: Dict[str, str],
        template: str,
        context_field: Optional[str] = None,
        answer: Optional[str] = None,
    ) -> str:
        data = {
            "question": row.get(self.question_field, ""),
            "context": row.get(context_field or self.context_field, ""),
        }
        if answer is not None:
            data["answer"] = answer
        return template.format(**data)

    def _build_param_prompt_without_answer(
        self,
        row: Dict[str, str],
        question: str,
        context_field: Optional[str],
    ) -> str:
        # Support context-aware param templates when {context} is provided.
        if "{context" in self.param_template:
            return self._build_prompt(row, self.param_template, context_field or self.context_field)
        return self.param_template.format(question=question)

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
            use_answer = self.train_use_answer and split == "train"
            if group == "context":
                if use_answer:
                    ctx_field = self.context_field
                else:
                    ctx_field = self.org_context_field if self.test_example_org_context else self.context_field
                context_value = row.get(ctx_field, "")
                ctx_answer = _normalize_answer(row.get(self.context_answer_field), self.answer_policy)
                ctx_answer_prompt = ctx_answer[0] if isinstance(ctx_answer, list) and ctx_answer else ctx_answer
                if use_answer:
                    prompt_body = self._build_prompt(
                        row,
                        self.context_template_with_answer,
                        ctx_field,
                        answer=str(ctx_answer_prompt),
                    )
                else:
                    prompt_body = self._build_prompt(row, self.context_template, ctx_field)
                prompt = prompt_body
                if self._demo_prefix_with_ctx:
                    prompt = self._demo_prefix_with_ctx + prompt
                target = ctx_answer
                alt_target = _normalize_answer(row.get(self.param_answer_field), self.answer_policy)
                targets = {"param": alt_target}
            elif group == "param":
                if use_answer:
                    param_ctx_field = self.param_context_field
                else:
                    param_ctx_field = self.param_context_field if self.train_include_param_context else None
                context_value = row.get(param_ctx_field or self.context_field, "")
                param_answer = _normalize_answer(row.get(self.param_answer_field), self.answer_policy)
                param_answer_prompt = param_answer[0] if isinstance(param_answer, list) and param_answer else param_answer
                if use_answer:
                    prompt_body = self._build_prompt(
                        row,
                        self.param_template_with_answer,
                        param_ctx_field or self.context_field,
                        answer=str(param_answer_prompt),
                    )
                else:
                    prompt_body = self._build_param_prompt_without_answer(row, question, param_ctx_field)
                prompt = prompt_body
                if self._demo_prefix_without_ctx:
                    prompt = self._demo_prefix_without_ctx + prompt
                target = param_answer
                alt_target = _normalize_answer(row.get(self.context_answer_field), self.answer_policy)
                targets = {"context": alt_target}
            else:
                raise KeyError(f"Unknown group: {group}")

            meta = {
                "row_id": str(idx),
                "context": str(context_value),
                "prompt_body": str(prompt_body),
                "demo_prefix_default": self._demo_prefix_with_ctx if group == "context" else self._demo_prefix_without_ctx,
                "demo_prefix_context_control": self._demo_prefix_context_control,
                "demo_prefix_param_control": self._demo_prefix_param_control,
            }
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
            use_answer = self.train_use_answer and split == "train"

            if use_answer:
                test_ctx_field = self.context_field
            else:
                test_ctx_field = self.org_context_field if self.test_example_org_context else self.context_field
            ctx_answer = _normalize_answer(row.get(self.context_answer_field), self.answer_policy)
            ctx_answer_prompt = ctx_answer[0] if isinstance(ctx_answer, list) and ctx_answer else ctx_answer
            if use_answer:
                ctx_prompt = self._build_prompt(
                    row,
                    self.context_template_with_answer,
                    test_ctx_field,
                    answer=str(ctx_answer_prompt),
                )
            else:
                ctx_prompt = self._build_prompt(row, self.context_template, test_ctx_field)
            if self._demo_prefix_with_ctx:
                ctx_prompt = self._demo_prefix_with_ctx + ctx_prompt
            if use_answer:
                param_ctx_field = self.param_context_field
            else:
                param_ctx_field = self.param_context_field if self.train_include_param_context else None
            param_answer = _normalize_answer(row.get(self.param_answer_field), self.answer_policy)
            param_answer_prompt = param_answer[0] if isinstance(param_answer, list) and param_answer else param_answer
            if use_answer:
                param_prompt = self._build_prompt(
                    row,
                    self.param_template_with_answer,
                    param_ctx_field or self.context_field,
                    answer=str(param_answer_prompt),
                )
            else:
                param_prompt = self._build_param_prompt_without_answer(row, question, param_ctx_field)
            if self._demo_prefix_without_ctx:
                param_prompt = self._demo_prefix_without_ctx + param_prompt

            ctx_target = ctx_answer
            param_target = param_answer

            ctx_example = Example(
                prompt=ctx_prompt,
                target=ctx_target,
                group="context",
                meta={"row_id": str(idx), "context": str(row.get(test_ctx_field, ""))},
                targets={"param": param_target},
            )
            param_example = Example(
                prompt=param_prompt,
                target=param_target,
                group="param",
                meta={"row_id": str(idx), "context": str(row.get(param_ctx_field or self.context_field, ""))},
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

    def __len__(self) -> int:
        return len(self._dataset)
