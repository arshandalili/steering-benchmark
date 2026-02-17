"""Intervention operation builders."""
from __future__ import annotations

from steering_benchmark.core.intervention import (
    VectorAddIntervention,
    VectorReplaceIntervention,
    VectorClampIntervention,
    VectorProjectOutIntervention,
)


def build_intervention(op_type: str, direction, scale: float, token_position: str | int):
    op_type = op_type or "add"
    if op_type == "add":
        return VectorAddIntervention(direction=direction, scale=scale, token_position=token_position)
    if op_type == "replace":
        return VectorReplaceIntervention(direction=direction, scale=scale, token_position=token_position)
    if op_type == "clamp":
        return VectorClampIntervention(direction=direction, scale=scale, token_position=token_position)
    if op_type in {"project_out", "project-out"}:
        return VectorProjectOutIntervention(direction=direction, scale=scale, token_position=token_position)
    raise KeyError(f"Unknown intervention op: {op_type}")
