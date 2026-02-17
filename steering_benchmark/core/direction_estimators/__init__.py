"""Direction estimator registry."""
from __future__ import annotations

from typing import Dict, Type

from steering_benchmark.core.direction_estimators.base import DirectionEstimator
from steering_benchmark.core.direction_estimators.diffmean import DiffMeanEstimator
from steering_benchmark.core.direction_estimators.actadd import ActAddEstimator
from steering_benchmark.core.direction_estimators.linear_probe import LinearProbeEstimator
from steering_benchmark.core.direction_estimators.pca import PCAEstimator
from steering_benchmark.core.direction_estimators.lda import LDAEstimator
from steering_benchmark.core.direction_estimators.caa import CAAEstimator

ESTIMATOR_REGISTRY: Dict[str, Type[DirectionEstimator]] = {
    "diffmean": DiffMeanEstimator,
    "actadd": ActAddEstimator,
    "linear_probe": LinearProbeEstimator,
    "pca": PCAEstimator,
    "lda": LDAEstimator,
    "caa": CAAEstimator,
}


def get_estimator(name: str) -> DirectionEstimator:
    if name not in ESTIMATOR_REGISTRY:
        raise KeyError(f"Unknown direction estimator: {name}")
    return ESTIMATOR_REGISTRY[name]()
