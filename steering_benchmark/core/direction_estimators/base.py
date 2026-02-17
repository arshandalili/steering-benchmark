"""Direction estimator interfaces."""
from __future__ import annotations

from typing import Optional

import torch


class DirectionEstimator:
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
        raise NotImplementedError
