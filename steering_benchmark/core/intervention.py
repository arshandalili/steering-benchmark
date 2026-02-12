"""Intervention primitives."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VectorIntervention:
    direction: torch.Tensor
    scale: float
    token_position: str | int

    def apply(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states is None:
            return hidden_states

        direction = self.direction.to(hidden_states.device)
        if direction.dim() == 1:
            direction = direction.view(1, 1, -1)

        if self.token_position == "all":
            hidden_states = hidden_states + self.scale * direction
            return hidden_states

        if self.token_position == "last":
            idx = -1
        else:
            idx = int(self.token_position)

        hidden_states = hidden_states.clone()
        hidden_states[:, idx, :] = hidden_states[:, idx, :] + self.scale * direction.squeeze(0)
        return hidden_states


@dataclass
class InterventionSpec:
    layer: int
    intervention: VectorIntervention


@dataclass
class InterventionPlan:
    specs: list[InterventionSpec]


def scale_intervention(intervention, factor: float):
    if intervention is None:
        return None
    specs = getattr(intervention, "specs", None)
    if specs is None:
        return InterventionSpec(
            layer=intervention.layer,
            intervention=VectorIntervention(
                direction=intervention.intervention.direction,
                scale=intervention.intervention.scale * factor,
                token_position=intervention.intervention.token_position,
            ),
        )
    scaled_specs = []
    for spec in specs:
        scaled_specs.append(
            InterventionSpec(
                layer=spec.layer,
                intervention=VectorIntervention(
                    direction=spec.intervention.direction,
                    scale=spec.intervention.scale * factor,
                    token_position=spec.intervention.token_position,
                ),
            )
        )
    return InterventionPlan(specs=scaled_specs)
