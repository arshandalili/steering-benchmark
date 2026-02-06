"""Steering method interface."""
from __future__ import annotations

from typing import Dict

from steering_benchmark.core.intervention import InterventionSpec


class SteeringMethod:
    def __init__(self, config: Dict) -> None:
        self.config = config

    def fit(self, model, dataset, run_cfg: Dict) -> InterventionSpec:
        raise NotImplementedError
