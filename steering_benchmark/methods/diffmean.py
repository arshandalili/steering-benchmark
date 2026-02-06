"""DiffMean steering method."""
from __future__ import annotations

from typing import Dict, Iterable, List

import torch

from steering_benchmark.core.intervention import InterventionPlan, InterventionSpec, VectorIntervention
from steering_benchmark.methods.base import SteeringMethod
from steering_benchmark.registry import register_method


@register_method("diffmean")
class DiffMeanSteering(SteeringMethod):
    def _batched_prompts(self, examples: Iterable, batch_size: int) -> Iterable[List[str]]:
        batch = []
        for ex in examples:
            batch.append(ex.prompt)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _mean_hidden(self, model, dataset, group: str, layer: int, token_position: str | int, limit: int, batch_size: int) -> torch.Tensor:
        if token_position == "all":
            raise ValueError("DiffMean requires a specific token position, not 'all'.")

        running_sum = None
        count = 0
        examples = dataset.iter_group(group, limit=limit)
        for batch in self._batched_prompts(examples, batch_size):
            hidden = model.encode_hidden(batch, layer=layer, token_position=token_position)
            hidden = torch.tensor(hidden)
            if running_sum is None:
                running_sum = hidden.sum(dim=0)
            else:
                running_sum = running_sum + hidden.sum(dim=0)
            count += hidden.shape[0]

        if running_sum is None or count == 0:
            raise RuntimeError(f"No examples found for group '{group}'")
        return running_sum / count

    def fit(self, model, dataset, run_cfg: Dict) -> InterventionSpec:
        layers = self.config.get("layers")
        if layers is None:
            layers = [int(self.config.get("layer", 0))]
        layers = [int(layer) for layer in layers]
        token_position = self.config.get("token_position", "last")
        scale = float(self.config.get("scale", 1.0))
        normalize = bool(self.config.get("normalize", True))

        train_cfg = run_cfg.get("train", {})
        group_a = train_cfg.get("group_a", "context")
        group_b = train_cfg.get("group_b", "param")
        limit = int(train_cfg.get("max_examples", 512))
        batch_size = int(train_cfg.get("batch_size", 8))

        specs = []
        for layer in layers:
            mean_a = self._mean_hidden(model, dataset, group_a, layer, token_position, limit, batch_size)
            mean_b = self._mean_hidden(model, dataset, group_b, layer, token_position, limit, batch_size)
            direction = mean_a - mean_b

            if normalize:
                direction = direction / (direction.norm() + 1e-6)

            intervention = VectorIntervention(direction=direction, scale=scale, token_position=token_position)
            specs.append(InterventionSpec(layer=layer, intervention=intervention))

        if len(specs) == 1:
            return specs[0]
        return InterventionPlan(specs=specs)
