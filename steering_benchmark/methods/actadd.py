"""Activation Addition (ActAdd) steering method."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from steering_benchmark.core.intervention import InterventionPlan, InterventionSpec, VectorAddIntervention
from steering_benchmark.methods.base import SteeringMethod
from steering_benchmark.registry import register_method


@register_method("actadd")
class ActAddSteering(SteeringMethod):
    def _batched(self, pairs: Iterable[Tuple[str, str]], batch_size: int) -> Iterable[Tuple[List[str], List[str]]]:
        pos_batch: List[str] = []
        neg_batch: List[str] = []
        for pos_prompt, neg_prompt in pairs:
            pos_batch.append(pos_prompt)
            neg_batch.append(neg_prompt)
            if len(pos_batch) >= batch_size:
                yield pos_batch, neg_batch
                pos_batch, neg_batch = [], []
        if pos_batch:
            yield pos_batch, neg_batch

    def _collect_pairs(
        self,
        dataset,
        group_pos: str,
        group_neg: str,
        limit: int,
        split: Optional[str],
    ) -> List[Tuple[str, str]]:
        if hasattr(dataset, "iter_pairs"):
            pairs = []
            for pos_ex, neg_ex in dataset.iter_pairs(group_pos, group_neg, limit=limit, split=split):
                pairs.append((pos_ex.prompt, neg_ex.prompt))
            if pairs:
                return pairs
        # fallback: independent groups
        pos_prompts = [ex.prompt for ex in dataset.iter_group(group_pos, limit=limit, split=split)]
        neg_prompts = [ex.prompt for ex in dataset.iter_group(group_neg, limit=limit, split=split)]
        n = min(len(pos_prompts), len(neg_prompts))
        return list(zip(pos_prompts[:n], neg_prompts[:n]))

    def _mean_diff(
        self,
        model,
        dataset,
        group_pos: str,
        group_neg: str,
        layer: int,
        token_position: str | int,
        limit: int,
        batch_size: int,
        split: Optional[str],
        progress: bool = False,
    ) -> torch.Tensor:
        pairs = self._collect_pairs(dataset, group_pos, group_neg, limit, split)
        if not pairs:
            raise RuntimeError(f"No paired prompts found for {group_pos}/{group_neg}")

        running_sum = None
        count = 0
        iterator = self._batched(pairs, batch_size)
        if progress:
            try:
                from tqdm import tqdm
                import math

                total = math.ceil(len(pairs) / batch_size)
                iterator = tqdm(iterator, total=total, desc=f"actadd layer {layer}")
            except Exception:
                pass
        for pos_batch, neg_batch in iterator:
            pos_hidden = model.encode_hidden(pos_batch, layer=layer, token_position=token_position)
            neg_hidden = model.encode_hidden(neg_batch, layer=layer, token_position=token_position)
            if not isinstance(pos_hidden, torch.Tensor):
                pos_hidden = torch.tensor(pos_hidden)
            if not isinstance(neg_hidden, torch.Tensor):
                neg_hidden = torch.tensor(neg_hidden)
            diff = pos_hidden - neg_hidden
            if running_sum is None:
                running_sum = diff.sum(dim=0)
            else:
                running_sum = running_sum + diff.sum(dim=0)
            count += diff.shape[0]

        if running_sum is None or count == 0:
            raise RuntimeError("No pairs were processed for ActAdd")
        return running_sum / count

    def fit(self, model, dataset, run_cfg: Dict) -> InterventionSpec | InterventionPlan:
        layers = self.config.get("layers")
        if layers is None:
            layers = [int(self.config.get("layer", 0))]
        layers = [int(layer) for layer in layers]

        token_position = self.config.get("token_position", "last")
        scale = float(self.config.get("scale", 1.0))
        normalize = bool(self.config.get("normalize", True))
        direction_path = self.config.get("direction_path")

        train_cfg = run_cfg.get("train", {})
        group_pos = train_cfg.get("group_pos", train_cfg.get("group_a", "context"))
        group_neg = train_cfg.get("group_neg", train_cfg.get("group_b", "param"))
        limit = int(train_cfg.get("max_examples", 512))
        batch_size = int(train_cfg.get("batch_size", 8))
        split = train_cfg.get("split", run_cfg.get("splits", {}).get("train"))
        progress = bool(run_cfg.get("progress", True))

        if direction_path:
            if "{layer}" in direction_path:
                if all(Path(direction_path.format(layer=layer)).exists() for layer in layers):
                    specs = []
                    for layer in layers:
                        direction = torch.load(direction_path.format(layer=layer), map_location="cpu")
                        intervention = VectorAddIntervention(direction=direction, scale=scale, token_position=token_position)
                        specs.append(InterventionSpec(layer=layer, intervention=intervention))
                    return specs[0] if len(specs) == 1 else InterventionPlan(specs=specs)
            elif Path(direction_path).exists():
                direction = torch.load(direction_path, map_location="cpu")
                intervention = VectorAddIntervention(direction=direction, scale=scale, token_position=token_position)
                return InterventionSpec(layer=layers[0], intervention=intervention)

        specs = []
        for layer in layers:
            direction = self._mean_diff(
                model,
                dataset,
                group_pos,
                group_neg,
                layer,
                token_position,
                limit,
                batch_size,
                split,
                progress=progress,
            )
            if normalize:
                direction = direction / (direction.norm() + 1e-6)
            intervention = VectorAddIntervention(direction=direction, scale=scale, token_position=token_position)
            specs.append(InterventionSpec(layer=layer, intervention=intervention))

        if direction_path and specs:
            if "{layer}" in direction_path:
                for spec in specs:
                    out_path = Path(direction_path.format(layer=spec.layer))
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(spec.intervention.direction, out_path)
            else:
                Path(direction_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(specs[0].intervention.direction, direction_path)

        return specs[0] if len(specs) == 1 else InterventionPlan(specs=specs)
