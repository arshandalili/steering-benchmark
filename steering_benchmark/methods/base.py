"""Steering method interface."""
from __future__ import annotations

from typing import Dict, Optional

from steering_benchmark.core.intervention import InterventionSpec


class SteeringMethod:
    def __init__(self, config: Dict) -> None:
        self.config = config

    def fit(self, model, dataset, run_cfg: Dict) -> InterventionSpec:
        raise NotImplementedError

    def adapt_model(self, model, dataset, run_cfg: Dict):
        """Optionally fine-tune or adapt the model before fitting."""
        return model

    def generate(self, model, prompt: str, intervention=None, gen_cfg: Optional[dict] = None) -> str:
        """Optional decoding-time steering override."""
        return model.generate(prompt, intervention=intervention, gen_cfg=gen_cfg)

    def get_cost(self) -> Dict[str, float]:
        """Optional cost reporting for training-based methods."""
        return {}
