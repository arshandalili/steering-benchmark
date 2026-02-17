"""RePS steering baseline."""
from __future__ import annotations

from typing import Dict

from steering_benchmark.core.intervention import InterventionSpec, VectorAddIntervention
from steering_benchmark.core.direction_transforms import apply_transforms
from steering_benchmark.methods.base import SteeringMethod
from steering_benchmark.registry import register_method
from steering_benchmark.training.reps import train_reps


@register_method("reps")
class RePSSteering(SteeringMethod):
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self._cost = {}

    def fit(self, model, dataset, run_cfg: Dict):
        reps_cfg = self.config.get("reps", {})
        result = train_reps(model, dataset, run_cfg, reps_cfg)
        self._cost = {k: v for k, v in result.items() if k != "direction"}
        direction = result["direction"]
        transform_cfg = {"normalize": bool(self.config.get("normalize", True))}
        direction = apply_transforms(direction, transform_cfg)
        scale = float(self.config.get("scale", 1.0))
        token_position = self.config.get("token_position", "last")
        layer = int(self.config.get("layer", reps_cfg.get("layer", 0)))
        intervention = VectorAddIntervention(direction=direction, scale=scale, token_position=token_position)
        return InterventionSpec(layer=layer, intervention=intervention)

    def get_cost(self) -> Dict[str, float]:
        return self._cost
