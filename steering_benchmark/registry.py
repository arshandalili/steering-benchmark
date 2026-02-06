"""Simple plugin registry for datasets, methods, and models."""
from __future__ import annotations

from typing import Callable, Dict, Type, TypeVar

T = TypeVar("T")

DATASET_REGISTRY: Dict[str, Type] = {}
METHOD_REGISTRY: Dict[str, Type] = {}
MODEL_REGISTRY: Dict[str, Type] = {}


def _register(registry: Dict[str, Type], name: str) -> Callable[[Type[T]], Type[T]]:
    def decorator(cls: Type[T]) -> Type[T]:
        if name in registry:
            raise ValueError(f"{name} already registered")
        registry[name] = cls
        return cls

    return decorator


def register_dataset(name: str) -> Callable[[Type[T]], Type[T]]:
    return _register(DATASET_REGISTRY, name)


def register_method(name: str) -> Callable[[Type[T]], Type[T]]:
    return _register(METHOD_REGISTRY, name)


def register_model(name: str) -> Callable[[Type[T]], Type[T]]:
    return _register(MODEL_REGISTRY, name)
