"""Dataset primitives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Optional, Sequence, Union


TargetType = Union[str, Sequence[str]]


@dataclass
class Example:
    prompt: str
    target: TargetType
    group: str
    meta: Dict[str, str]
    targets: Optional[Dict[str, TargetType]] = None


class BaseDataset:
    def iter_group(
        self,
        group: str,
        limit: Optional[int] = None,
        split: Optional[str] = None,
    ) -> Iterator[Example]:
        raise NotImplementedError

    def groups(self) -> Iterable[str]:
        raise NotImplementedError
