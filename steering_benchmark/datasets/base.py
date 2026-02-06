"""Dataset primitives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Optional


@dataclass
class Example:
    prompt: str
    target: str
    group: str
    meta: Dict[str, str]


class BaseDataset:
    def iter_group(self, group: str, limit: Optional[int] = None) -> Iterator[Example]:
        raise NotImplementedError

    def groups(self) -> Iterable[str]:
        raise NotImplementedError
