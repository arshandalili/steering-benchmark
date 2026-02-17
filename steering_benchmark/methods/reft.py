"""ReFT steering baseline."""
from __future__ import annotations

from typing import Dict

from steering_benchmark.core.intervention import InterventionSpec, Rank1Intervention
from steering_benchmark.methods.base import SteeringMethod
from steering_benchmark.registry import register_method
from steering_benchmark.training.reft import train_reft


@register_method("reft")
class ReFTSteering(SteeringMethod):
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self._cost = {}
        self._u = None
        self._v = None
        self._layer = None
        self._token_position = None

    def adapt_model(self, model, dataset, run_cfg: Dict):
        reft_cfg = self.config.get("reft", {})
        result = train_reft(model, dataset, run_cfg, reft_cfg)
        self._cost = {k: v for k, v in result.items() if k not in {"u", "v", "layer", "token_position"}}
        self._u = result.get("u")
        self._v = result.get("v")
        self._layer = result.get("layer", reft_cfg.get("layer", 0))
        self._token_position = result.get("token_position", reft_cfg.get("token_position", "last"))
        return model

    def fit(self, model, dataset, run_cfg: Dict):
        if self._u is None or self._v is None:
            raise RuntimeError("ReFT parameters not trained; ensure adapt_model runs before fit.")
        scale = float(self.config.get("scale", 1.0))
        intervention = Rank1Intervention(
            u=self._u,
            v=self._v,
            scale=scale,
            token_position=self._token_position,
        )
        return InterventionSpec(layer=int(self._layer), intervention=intervention)

    def get_cost(self) -> Dict[str, float]:
        return self._cost
