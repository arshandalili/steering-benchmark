"""Intervention schedule builders."""
from __future__ import annotations

from typing import Iterable, List

from steering_benchmark.core.intervention import InterventionPlan, InterventionSpec


def build_schedule(layers: Iterable[int], interventions) -> InterventionPlan | InterventionSpec:
    layer_list: List[int] = [int(layer) for layer in layers]
    if not layer_list:
        raise ValueError("Schedule requires at least one layer")
    if len(layer_list) == 1:
        return InterventionSpec(layer=layer_list[0], intervention=interventions[0])
    specs = [InterventionSpec(layer=layer, intervention=intervention) for layer, intervention in zip(layer_list, interventions)]
    return InterventionPlan(specs=specs)
