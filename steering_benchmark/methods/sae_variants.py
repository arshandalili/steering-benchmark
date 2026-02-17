"""SAE feature steering variants."""
from __future__ import annotations

from typing import Dict, Iterable, List

import torch

from steering_benchmark.core.intervention import (
    InterventionPlan,
    InterventionSpec,
    VectorAddIntervention,
    SAEClampIntervention,
)
from steering_benchmark.methods.base import SteeringMethod
from steering_benchmark.methods.spare import SparseAutoencoder, load_sae_from_config
from steering_benchmark.registry import register_method


def _feature_effect_scores(model, decoder_weight: torch.Tensor, effect_type: str) -> torch.Tensor:
    if effect_type == "unembedding_l2":
        output_weight = None
        if hasattr(model, "model"):
            try:
                output_weight = model.model.get_output_embeddings().weight
            except Exception:
                output_weight = getattr(model.model, "lm_head", None)
                if output_weight is not None:
                    output_weight = output_weight.weight
        if output_weight is not None:
            scores = torch.norm(output_weight @ decoder_weight, dim=0)
            return scores
    return torch.norm(decoder_weight, dim=0)


def _sae_weights_for_variant(sae: SparseAutoencoder):
    backend = sae.sae_lens
    if hasattr(backend, "W_enc") and hasattr(backend, "W_dec"):
        encoder_weight = backend.W_enc.T
        decoder_weight = backend.W_dec.T
        encoder_bias = getattr(backend, "b_enc", None)
        decoder_bias = getattr(backend, "b_dec", None)
        activation = getattr(getattr(backend, "cfg", None), "activation_fn_str", "relu")
        return encoder_weight, decoder_weight, encoder_bias, decoder_bias, activation

    if hasattr(backend, "encoder") and hasattr(backend.encoder, "weight") and hasattr(backend, "W_dec"):
        encoder_weight = backend.encoder.weight
        decoder_weight = backend.W_dec.T
        encoder_bias = getattr(backend.encoder, "bias", None)
        decoder_bias = getattr(backend, "b_dec", None)
        return encoder_weight, decoder_weight, encoder_bias, decoder_bias, "relu"

    raise ValueError("Unsupported SAE backend for sae_variant.")


@register_method("sae_variant")
class SAEVariantSteering(SteeringMethod):
    def _batched_prompts(self, examples: Iterable, batch_size: int):
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

    def _select_features(self, diff: torch.Tensor, sae: SparseAutoencoder, model, selection_cfg: dict) -> torch.Tensor:
        _, decoder_weight, _, _, _ = _sae_weights_for_variant(sae)
        mask = torch.ones_like(diff, dtype=torch.bool)
        if selection_cfg.get("effect_filter"):
            effect_cfg = selection_cfg.get("effect_filter", {})
            if not isinstance(effect_cfg, dict):
                effect_cfg = {}
            effect_type = effect_cfg.get("type", "unembedding_l2")
            scores = _feature_effect_scores(model, decoder_weight, effect_type)
            if effect_cfg.get("topk") is not None:
                k = max(1, int(effect_cfg["topk"]))
                indices = torch.topk(scores, k=k).indices
                mask = torch.zeros_like(diff, dtype=torch.bool)
                mask[indices] = True
            elif effect_cfg.get("threshold") is not None:
                mask = scores >= float(effect_cfg["threshold"])
        if selection_cfg.get("topk") is not None:
            k = max(1, int(selection_cfg["topk"]))
            indices = torch.topk(diff.abs(), k=k).indices
            topk_mask = torch.zeros_like(diff, dtype=torch.bool)
            topk_mask[indices] = True
            mask = mask & topk_mask
        if selection_cfg.get("threshold") is not None:
            mask = mask & (diff.abs() >= float(selection_cfg["threshold"]))
        return mask

    def fit(self, model, dataset, run_cfg: Dict):
        layers = self.config.get("layers")
        if layers is None:
            layers = [int(self.config.get("layer", 0))]
        layers = [int(layer) for layer in layers]
        token_position = self.config.get("token_position", "last")
        scale = float(self.config.get("scale", 1.0))
        normalize = bool(self.config.get("normalize", True))

        sae_cfg = self.config.get("sae", {})

        selection_cfg = self.config.get("feature_selection", {})
        feature_op = self.config.get("feature_op", "add")
        decode_mode = self.config.get("decode_mode", "direction_add")
        if decode_mode not in {"direction_add"}:
            raise ValueError(f"Unsupported decode_mode: {decode_mode}")

        train_cfg = run_cfg.get("train", {})
        group_a = train_cfg.get("group_a", "context")
        group_b = train_cfg.get("group_b", "param")
        limit = int(train_cfg.get("max_examples", 512))
        batch_size = int(train_cfg.get("batch_size", 8))
        split = train_cfg.get("split", run_cfg.get("splits", {}).get("train"))

        specs: List[InterventionSpec] = []
        for layer in layers:
            try:
                sae = load_sae_from_config(sae_cfg, layer)
            except Exception as exc:
                raise ValueError("SAEVariant requires TransformerLens SAE config.") from exc
            mean_a = self._mean_features(model, dataset, group_a, layer, token_position, limit, batch_size, sae, split)
            mean_b = self._mean_features(model, dataset, group_b, layer, token_position, limit, batch_size, sae, split)
            diff = mean_a - mean_b
            mask = self._select_features(diff, sae, model, selection_cfg)
            encoder_weight, decoder_weight, encoder_bias, decoder_bias, activation = _sae_weights_for_variant(sae)
            if feature_op == "clamp":
                intervention = SAEClampIntervention(
                    encoder_weight=encoder_weight,
                    decoder_weight=decoder_weight,
                    encoder_bias=encoder_bias,
                    decoder_bias=decoder_bias,
                    activation=activation,
                    target_features=mean_a,
                    feature_mask=mask.float(),
                    scale=scale,
                    token_position=token_position,
                )
            else:
                diff = diff * mask
                direction = sae.decode(diff)
                if normalize:
                    direction = direction / (direction.norm() + 1e-6)
                intervention = VectorAddIntervention(direction=direction, scale=scale, token_position=token_position)

            specs.append(InterventionSpec(layer=layer, intervention=intervention))

        return specs[0] if len(specs) == 1 else InterventionPlan(specs=specs)
