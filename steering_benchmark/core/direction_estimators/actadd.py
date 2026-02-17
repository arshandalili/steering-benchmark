"""ActAdd direction estimator."""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from steering_benchmark.core.direction_estimators.base import DirectionEstimator


class ActAddEstimator(DirectionEstimator):
    def _batched(self, pairs: List[Tuple[str, str]], batch_size: int):
        pos_batch: List[str] = []
        neg_batch: List[str] = []
        for pos_prompt, neg_prompt in pairs:
            pos_batch.append(pos_prompt)
            neg_batch.append(neg_prompt)
            if len(pos_batch) >= batch_size:
                yield pos_batch, neg_batch
                pos_batch, neg_batch = [], []
        if pos_batch:
            yield pos_batch, neg_batch

    def _collect_pairs(
        self,
        dataset,
        group_pos: str,
        group_neg: str,
        limit: int,
        split: Optional[str],
    ) -> List[Tuple[str, str]]:
        if hasattr(dataset, "iter_pairs"):
            pairs = []
            for pos_ex, neg_ex in dataset.iter_pairs(group_pos, group_neg, limit=limit, split=split):
                pairs.append((pos_ex.prompt, neg_ex.prompt))
            if pairs:
                return pairs
        pos_prompts = [ex.prompt for ex in dataset.iter_group(group_pos, limit=limit, split=split)]
        neg_prompts = [ex.prompt for ex in dataset.iter_group(group_neg, limit=limit, split=split)]
        n = min(len(pos_prompts), len(neg_prompts))
        return list(zip(pos_prompts[:n], neg_prompts[:n]))

    def fit_direction(
        self,
        model,
        dataset,
        run_cfg: dict,
        layer: int,
        token_position: str | int,
        split: Optional[str],
        group_pos: str,
        group_neg: str,
    ) -> torch.Tensor:
        train_cfg = run_cfg.get("train", {})
        limit = int(train_cfg.get("max_examples", 512))
        batch_size = int(train_cfg.get("batch_size", 8))
        pairs = self._collect_pairs(dataset, group_pos, group_neg, limit=limit, split=split)
        if not pairs:
            raise RuntimeError(f"No paired prompts found for {group_pos}/{group_neg}")
        running_sum = None
        count = 0
        for pos_batch, neg_batch in self._batched(pairs, batch_size):
            pos_hidden = model.encode_hidden(pos_batch, layer=layer, token_position=token_position)
            neg_hidden = model.encode_hidden(neg_batch, layer=layer, token_position=token_position)
            if not isinstance(pos_hidden, torch.Tensor):
                pos_hidden = torch.tensor(pos_hidden)
            if not isinstance(neg_hidden, torch.Tensor):
                neg_hidden = torch.tensor(neg_hidden)
            diff = pos_hidden - neg_hidden
            if running_sum is None:
                running_sum = diff.sum(dim=0)
            else:
                running_sum = running_sum + diff.sum(dim=0)
            count += diff.shape[0]
        if running_sum is None or count == 0:
            raise RuntimeError("No pairs were processed for ActAdd estimator")
        return running_sum / count
