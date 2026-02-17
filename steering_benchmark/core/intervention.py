"""Intervention primitives."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import torch


class Intervention(Protocol):
    def apply(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...

    def scale_by(self, factor: float):
        ...


def _expand_direction(direction: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    direction = direction.to(hidden_states.device)
    if direction.dim() == 1:
        direction = direction.view(1, 1, -1)
    return direction


@dataclass
class VectorAddIntervention:
    direction: torch.Tensor
    scale: float
    token_position: str | int

    def apply(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states is None:
            return hidden_states
        direction = _expand_direction(self.direction, hidden_states)
        if self.token_position == "all":
            return hidden_states + self.scale * direction
        if self.token_position == "last":
            idx = -1
        else:
            idx = int(self.token_position)
        hidden_states = hidden_states.clone()
        hidden_states[:, idx, :] = hidden_states[:, idx, :] + self.scale * direction.squeeze(0)
        return hidden_states

    def scale_by(self, factor: float):
        return replace(self, scale=self.scale * factor)


@dataclass
class VectorReplaceIntervention:
    direction: torch.Tensor
    scale: float
    token_position: str | int

    def apply(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states is None:
            return hidden_states
        direction = _expand_direction(self.direction, hidden_states) * self.scale
        if self.token_position == "all":
            return direction.expand_as(hidden_states)
        if self.token_position == "last":
            idx = -1
        else:
            idx = int(self.token_position)
        hidden_states = hidden_states.clone()
        hidden_states[:, idx, :] = direction.squeeze(0)
        return hidden_states

    def scale_by(self, factor: float):
        return replace(self, scale=self.scale * factor)


@dataclass
class VectorProjectOutIntervention:
    direction: torch.Tensor
    scale: float
    token_position: str | int

    def apply(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states is None:
            return hidden_states
        direction = _expand_direction(self.direction, hidden_states)
        norm = (direction.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
        if self.token_position == "all":
            proj = (hidden_states * direction).sum(dim=-1, keepdim=True) / norm
            return hidden_states - self.scale * proj * direction
        if self.token_position == "last":
            idx = -1
        else:
            idx = int(self.token_position)
        hidden_states = hidden_states.clone()
        token_states = hidden_states[:, idx, :]
        proj = (token_states * direction.squeeze(0)).sum(dim=-1, keepdim=True) / norm.squeeze(0)
        hidden_states[:, idx, :] = token_states - self.scale * proj * direction.squeeze(0)
        return hidden_states

    def scale_by(self, factor: float):
        return replace(self, scale=self.scale * factor)


@dataclass
class VectorClampIntervention:
    direction: torch.Tensor
    scale: float
    token_position: str | int

    def apply(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states is None:
            return hidden_states
        direction = _expand_direction(self.direction, hidden_states)
        if self.token_position == "all":
            return hidden_states + self.scale * torch.tanh(direction)
        if self.token_position == "last":
            idx = -1
        else:
            idx = int(self.token_position)
        hidden_states = hidden_states.clone()
        hidden_states[:, idx, :] = hidden_states[:, idx, :] + self.scale * torch.tanh(direction.squeeze(0))
        return hidden_states

    def scale_by(self, factor: float):
        return replace(self, scale=self.scale * factor)


@dataclass
class SAEClampIntervention:
    encoder_weight: torch.Tensor
    decoder_weight: torch.Tensor
    encoder_bias: torch.Tensor | None
    decoder_bias: torch.Tensor | None
    activation: str
    target_features: torch.Tensor
    feature_mask: torch.Tensor
    scale: float
    token_position: str | int

    def _encode(self, hidden: torch.Tensor) -> torch.Tensor:
        acts = hidden @ self.encoder_weight.T
        if self.encoder_bias is not None:
            acts = acts + self.encoder_bias
        if self.activation == "relu":
            acts = torch.relu(acts)
        elif self.activation == "gelu":
            acts = torch.nn.functional.gelu(acts)
        return acts

    def _decode(self, features: torch.Tensor) -> torch.Tensor:
        hidden = features @ self.decoder_weight.T
        if self.decoder_bias is not None:
            hidden = hidden + self.decoder_bias
        return hidden

    def apply(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states is None:
            return hidden_states
        if self.token_position == "all":
            hidden = hidden_states
            features = self._encode(hidden)
            target = self.target_features.to(hidden.device)
            mask = self.feature_mask.to(hidden.device)
            delta_features = (target - features) * mask
            delta_hidden = self._decode(delta_features)
            return hidden_states + self.scale * delta_hidden

        if self.token_position == "last":
            idx = -1
        else:
            idx = int(self.token_position)
        hidden_states = hidden_states.clone()
        hidden = hidden_states[:, idx, :]
        features = self._encode(hidden)
        target = self.target_features.to(hidden.device)
        mask = self.feature_mask.to(hidden.device)
        delta_features = (target - features) * mask
        delta_hidden = self._decode(delta_features)
        hidden_states[:, idx, :] = hidden + self.scale * delta_hidden
        return hidden_states

    def scale_by(self, factor: float):
        return replace(self, scale=self.scale * factor)


@dataclass
class Rank1Intervention:
    u: torch.Tensor
    v: torch.Tensor
    scale: float
    token_position: str | int

    def apply(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states is None:
            return hidden_states
        u = self.u.to(hidden_states.device)
        v = self.v.to(hidden_states.device)
        if self.token_position == "all":
            proj = hidden_states @ u
            return hidden_states + self.scale * proj.unsqueeze(-1) * v
        if self.token_position == "last":
            idx = -1
        else:
            idx = int(self.token_position)
        hidden_states = hidden_states.clone()
        proj = hidden_states[:, idx, :] @ u
        hidden_states[:, idx, :] = hidden_states[:, idx, :] + self.scale * proj.unsqueeze(-1) * v
        return hidden_states

    def scale_by(self, factor: float):
        return replace(self, scale=self.scale * factor)


@dataclass
class InterventionSpec:
    layer: int
    intervention: Intervention


@dataclass
class InterventionPlan:
    specs: list[InterventionSpec]


def scale_intervention(intervention, factor: float):
    if intervention is None:
        return None
    specs = getattr(intervention, "specs", None)
    if specs is None:
        return InterventionSpec(layer=intervention.layer, intervention=intervention.intervention.scale_by(factor))
    scaled_specs = []
    for spec in specs:
        scaled_specs.append(
            InterventionSpec(
                layer=spec.layer,
                intervention=spec.intervention.scale_by(factor),
            )
        )
    return InterventionPlan(specs=scaled_specs)
