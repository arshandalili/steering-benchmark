"""PCA steering method."""
from __future__ import annotations

from typing import Dict

from steering_benchmark.core.direction_estimators import get_estimator
from steering_benchmark.core.direction_transforms import apply_transforms
from steering_benchmark.core.intervention import InterventionPlan, InterventionSpec, VectorAddIntervention
from steering_benchmark.methods.base import SteeringMethod
from steering_benchmark.registry import register_method


@register_method("pca")
class PCASteering(SteeringMethod):
    def fit(self, model, dataset, run_cfg: Dict):
        layers = self.config.get("layers")
        if layers is None:
            layers = [int(self.config.get("layer", 0))]
        layers = [int(layer) for layer in layers]
        token_position = self.config.get("token_position", "last")
        scale = float(self.config.get("scale", 1.0))
        transform_cfg = {"normalize": bool(self.config.get("normalize", True))}

        estimator = get_estimator("pca")
        train_cfg = run_cfg.get("train", {})
        group_pos = train_cfg.get("group_pos", train_cfg.get("group_a", "context"))
        group_neg = train_cfg.get("group_neg", train_cfg.get("group_b", "param"))
        split = train_cfg.get("split", run_cfg.get("splits", {}).get("train"))

        specs = []
        for layer in layers:
            direction = estimator.fit_direction(
                model,
                dataset,
                run_cfg,
                layer=layer,
                token_position=token_position,
                split=split,
                group_pos=group_pos,
                group_neg=group_neg,
            )
            direction = apply_transforms(direction, transform_cfg)
            intervention = VectorAddIntervention(direction=direction, scale=scale, token_position=token_position)
            specs.append(InterventionSpec(layer=layer, intervention=intervention))
        return specs[0] if len(specs) == 1 else InterventionPlan(specs=specs)
