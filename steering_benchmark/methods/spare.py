"""Sparse autoencoder-based steering (SpARE-style)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch

from steering_benchmark.core.intervention import InterventionPlan, InterventionSpec, VectorIntervention
from steering_benchmark.methods.base import SteeringMethod
from steering_benchmark.registry import register_method


@dataclass
class SAEWeights:
    encoder_weight: torch.Tensor
    decoder_weight: torch.Tensor
    encoder_bias: torch.Tensor | None = None
    decoder_bias: torch.Tensor | None = None
    activation: str = "relu"

    @classmethod
    def load(cls, path: str, activation: str = "relu") -> "SAEWeights":
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"SAE weights not found: {path}")

        if path_obj.suffix == ".npz":
            data = np.load(path_obj)
            encoder_weight = torch.tensor(data["encoder_weight"])
            decoder_weight = torch.tensor(data["decoder_weight"])
            encoder_bias = torch.tensor(data["encoder_bias"]) if "encoder_bias" in data else None
            decoder_bias = torch.tensor(data["decoder_bias"]) if "decoder_bias" in data else None
        else:
            data = torch.load(path_obj, map_location="cpu")
            encoder_weight = data["encoder_weight"]
            decoder_weight = data["decoder_weight"]
            encoder_bias = data.get("encoder_bias")
            decoder_bias = data.get("decoder_bias")

        return cls(
            encoder_weight=encoder_weight,
            decoder_weight=decoder_weight,
            encoder_bias=encoder_bias,
            decoder_bias=decoder_bias,
            activation=activation,
        )


class SparseAutoencoder:
    def __init__(self, weights: SAEWeights) -> None:
        self.weights = weights

    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        acts = hidden @ self.weights.encoder_weight.T
        if self.weights.encoder_bias is not None:
            acts = acts + self.weights.encoder_bias
        if self.weights.activation == "relu":
            acts = torch.relu(acts)
        elif self.weights.activation == "gelu":
            acts = torch.nn.functional.gelu(acts)
        return acts

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        hidden = features @ self.weights.decoder_weight.T
        if self.weights.decoder_bias is not None:
            hidden = hidden + self.weights.decoder_bias
        return hidden


@register_method("spare")
class SpARESteering(SteeringMethod):
    def _batched_prompts(self, examples: Iterable, batch_size: int) -> Iterable[List[str]]:
        batch = []
        for ex in examples:
            batch.append(ex.prompt)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _mean_features(
        self,
        model,
        dataset,
        group: str,
        layer: int,
        token_position: str | int,
        limit: int,
        batch_size: int,
        sae: SparseAutoencoder,
        split: str | None,
    ) -> torch.Tensor:
        running_sum = None
        count = 0
        examples = dataset.iter_group(group, limit=limit, split=split)
        for batch in self._batched_prompts(examples, batch_size):
            hidden = model.encode_hidden(batch, layer=layer, token_position=token_position).float()
            features = sae.encode(hidden)
            if running_sum is None:
                running_sum = features.sum(dim=0)
            else:
                running_sum = running_sum + features.sum(dim=0)
            count += features.shape[0]

        if running_sum is None or count == 0:
            raise RuntimeError(f"No examples found for group '{group}'")
        return running_sum / count

    def fit(self, model, dataset, run_cfg: Dict) -> InterventionSpec:
        layers = self.config.get("layers")
        if layers is None:
            layers = [int(self.config.get("layer", 0))]
        layers = [int(layer) for layer in layers]
        token_position = self.config.get("token_position", "last")
        scale = float(self.config.get("scale", 1.0))
        normalize = bool(self.config.get("normalize", True))

        direction_path = self.config.get("direction_path")
        if direction_path:
            if "{layer}" in direction_path:
                if all(Path(direction_path.format(layer=layer)).exists() for layer in layers):
                    specs = []
                    for layer in layers:
                        direction = torch.load(direction_path.format(layer=layer), map_location="cpu")
                        intervention = VectorIntervention(direction=direction, scale=scale, token_position=token_position)
                        specs.append(InterventionSpec(layer=layer, intervention=intervention))
                    if len(specs) == 1:
                        return specs[0]
                    return InterventionPlan(specs=specs)
            elif Path(direction_path).exists():
                direction = torch.load(direction_path, map_location="cpu")
                intervention = VectorIntervention(direction=direction, scale=scale, token_position=token_position)
                return InterventionSpec(layer=layers[0], intervention=intervention)

        sae_cfg = self.config.get("sae", {})
        weights_path = sae_cfg.get("weights_path")
        activation = sae_cfg.get("activation", "relu")
        topk = sae_cfg.get("topk")
        topk_prop = sae_cfg.get("topk_proportion")
        threshold = sae_cfg.get("threshold")

        if not weights_path:
            raise ValueError("SpARE requires sae.weights_path in method config.")

        sae = SparseAutoencoder(SAEWeights.load(weights_path, activation=activation))

        train_cfg = run_cfg.get("train", {})
        group_a = train_cfg.get("group_a", "context")
        group_b = train_cfg.get("group_b", "param")
        limit = int(train_cfg.get("max_examples", 512))
        batch_size = int(train_cfg.get("batch_size", 8))
        split = train_cfg.get("split", run_cfg.get("splits", {}).get("train"))

        specs = []
        for layer in layers:
            mean_a = self._mean_features(
                model, dataset, group_a, layer, token_position, limit, batch_size, sae, split
            )
            mean_b = self._mean_features(
                model, dataset, group_b, layer, token_position, limit, batch_size, sae, split
            )
            diff = mean_a - mean_b

            if threshold is not None:
                mask = diff.abs() >= float(threshold)
            elif topk is not None or topk_prop is not None:
                if topk is None:
                    topk = int(float(topk_prop) * diff.numel())
                topk = max(1, int(topk))
                indices = torch.topk(diff.abs(), k=topk).indices
                mask = torch.zeros_like(diff, dtype=torch.bool)
                mask[indices] = True
            else:
                mask = torch.ones_like(diff, dtype=torch.bool)

            diff = diff * mask
            direction = sae.decode(diff)

            if normalize:
                direction = direction / (direction.norm() + 1e-6)

            intervention = VectorIntervention(direction=direction, scale=scale, token_position=token_position)
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

        if len(specs) == 1:
            return specs[0]
        return InterventionPlan(specs=specs)
