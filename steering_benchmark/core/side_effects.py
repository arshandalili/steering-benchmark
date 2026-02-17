"""Side-effect evaluation helpers."""
from __future__ import annotations

from typing import Dict


def build_side_effects_config(run_cfg: Dict) -> Dict:
    """Normalize side-effect configuration."""
    cfg = dict(run_cfg.get("side_effects", {}) or {})
    cfg.setdefault("datasets", [])
    cfg.setdefault("metrics", [])
    return cfg
