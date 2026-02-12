"""JSONL dataset with templated prompt construction."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional

import json

from steering_benchmark.datasets.base import BaseDataset, Example
from steering_benchmark.registry import register_dataset


@register_dataset("jsonl_template")
class JsonlTemplateDataset(BaseDataset):
    def __init__(
        self,
        path: str,
        templates: Dict[str, str],
        fields: Dict[str, str],
        targets: Dict[str, str],
    ) -> None:
        self.path = Path(path)
        self.templates = templates
        self.fields = fields
        self.targets = targets
        self._rows = self._load_rows()

    def _load_rows(self) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def _build_prompt(self, row: Dict[str, str], template: str) -> str:
        data = {}
        for key, field_name in self.fields.items():
            data[key] = row.get(field_name, "")
        return template.format(**data)

    def iter_group(
        self,
        group: str,
        limit: Optional[int] = None,
        split: Optional[str] = None,
    ) -> Iterator[Example]:
        if group not in self.templates:
            raise KeyError(f"Unknown group template: {group}")
        template = self.templates[group]
        target_field = self.targets[group]

        count = 0
        for row in self._rows:
            prompt = self._build_prompt(row, template)
            target = str(row.get(target_field, ""))
            meta = {"row_id": str(count)}
            yield Example(prompt=prompt, target=target, group=group, meta=meta)
            count += 1
            if limit is not None and count >= limit:
                break

    def groups(self) -> List[str]:
        return list(self.templates.keys())
