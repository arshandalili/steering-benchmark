"""Contrastive Activation Addition (CAA) estimator."""
from __future__ import annotations

from typing import Optional

import torch

from steering_benchmark.core.direction_estimators.actadd import ActAddEstimator


class CAAEstimator(ActAddEstimator):
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
        # Explicitly uses paired differences; identical to ActAdd with explicit naming.
        return super().fit_direction(model, dataset, run_cfg, layer, token_position, split, group_pos, group_neg)
