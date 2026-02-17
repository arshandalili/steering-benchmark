"""Composable steering method built from estimator/transform/op/schedule pieces."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch

from steering_benchmark.core.direction_estimators import get_estimator
from steering_benchmark.core.direction_transforms import apply_transforms
from steering_benchmark.core.intervention_ops import build_intervention
from steering_benchmark.core.schedules import build_schedule
from steering_benchmark.methods.base import SteeringMethod
from steering_benchmark.registry import register_method


@register_method("composite")
class CompositeSteering(SteeringMethod):
    def _load_orthogonal_bases(self, transform_cfg: dict) -> List[torch.Tensor]:
        bases = []
        for path in transform_cfg.get("orthogonalize_paths", []) or []:
            if Path(path).exists():
                bases.append(torch.load(path, map_location="cpu"))
        return bases

    def fit(self, model, dataset, run_cfg: Dict):
        estimator_cfg = dict(self.config.get("estimator", {}))
        transform_cfg = dict(self.config.get("transform", {}))
        op_cfg = dict(self.config.get("op", {}))
        schedule_cfg = dict(self.config.get("schedule", {}))

        estimator_type = estimator_cfg.get("type", "diffmean")
        estimator = get_estimator(estimator_type)

        layers = schedule_cfg.get("layers") or self.config.get("layers")
        if layers is None:
            layers = [int(self.config.get("layer", 0))]
        layers = [int(layer) for layer in layers]

        token_position = schedule_cfg.get("token_position", self.config.get("token_position", "last"))
        scale = float(op_cfg.get("scale", self.config.get("scale", 1.0)))
        op_type = op_cfg.get("type", "add")

        train_cfg = run_cfg.get("train", {})
        group_pos = estimator_cfg.get("group_pos", train_cfg.get("group_pos", train_cfg.get("group_a", "context")))
        group_neg = estimator_cfg.get("group_neg", train_cfg.get("group_neg", train_cfg.get("group_b", "param")))
        split = estimator_cfg.get("split", train_cfg.get("split", run_cfg.get("splits", {}).get("train")))

        direction_path = estimator_cfg.get("direction_path")
        orthogonal_bases = self._load_orthogonal_bases(transform_cfg)

        interventions = []
        for layer in layers:
            direction = None
            if direction_path:
                path = direction_path.format(layer=layer) if "{layer}" in direction_path else direction_path
                if Path(path).exists():
                    direction = torch.load(path, map_location="cpu")
            if direction is None:
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
                direction = apply_transforms(direction, transform_cfg, orthogonalize_bases=orthogonal_bases)
                if direction_path:
                    path = direction_path.format(layer=layer) if "{layer}" in direction_path else direction_path
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(direction, path)
            interventions.append(build_intervention(op_type, direction, scale, token_position))

        return build_schedule(layers, interventions)
