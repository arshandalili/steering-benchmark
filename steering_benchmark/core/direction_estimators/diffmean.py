"""DiffMean direction estimator."""
from __future__ import annotations

from typing import Optional

import torch

from steering_benchmark.core.direction_estimators.base import DirectionEstimator


class DiffMeanEstimator(DirectionEstimator):
    def _batched_prompts(self, examples, batch_size: int):
        batch = []
        for ex in examples:
            batch.append(ex.prompt)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _mean_hidden(
        self,
        model,
        dataset,
        group: str,
        layer: int,
        token_position: str | int,
        limit: int,
        batch_size: int,
        split: Optional[str],
    ) -> torch.Tensor:
        running_sum = None
        count = 0
        examples = dataset.iter_group(group, limit=limit, split=split)
        for batch in self._batched_prompts(examples, batch_size):
            hidden = model.encode_hidden(batch, layer=layer, token_position=token_position).float()
            if running_sum is None:
                running_sum = hidden.sum(dim=0)
            else:
                running_sum = running_sum + hidden.sum(dim=0)
            count += hidden.shape[0]
        if running_sum is None or count == 0:
            raise RuntimeError(f"No examples found for group '{group}'")
        return running_sum / count

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
        mean_a = self._mean_hidden(model, dataset, group_pos, layer, token_position, limit, batch_size, split)
        mean_b = self._mean_hidden(model, dataset, group_neg, layer, token_position, limit, batch_size, split)
        return mean_a - mean_b
