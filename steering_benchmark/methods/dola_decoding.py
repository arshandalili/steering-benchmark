"""DoLa decoding baseline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from steering_benchmark.decoding.dola import dola_decode
from steering_benchmark.methods.base import SteeringMethod
from steering_benchmark.registry import register_method


@dataclass
class _DecodingControl:
    scale: float = 1.0

    def scale_by(self, factor: float) -> "_DecodingControl":
        return _DecodingControl(scale=self.scale * float(factor))


@register_method("dola_decoding")
class DoLADecodingSteering(SteeringMethod):
    def fit(self, model, dataset, run_cfg: Dict):
        # DoLA is decoding-time only; this control object lets factor sweeps scale alpha.
        return _DecodingControl(scale=1.0)

    def generate(self, model, prompt: str, intervention=None, gen_cfg: Optional[dict] = None) -> str:
        gen_cfg = dict(gen_cfg or {})
        if intervention is None:
            return model.generate(prompt, intervention=None, gen_cfg=gen_cfg)
        control_scale = float(getattr(intervention, "scale", 1.0))
        if abs(control_scale) < 1e-12:
            return model.generate(prompt, intervention=None, gen_cfg=gen_cfg)
        if not hasattr(model, "model") or not hasattr(model, "tokenizer"):
            return model.generate(prompt, intervention=intervention, gen_cfg=gen_cfg)
        max_new_tokens = int(gen_cfg.get("max_new_tokens", 32))
        alpha = float(self.config.get("alpha", 1.0)) * control_scale
        early_layer = int(self.config.get("early_layer", 4))
        early_layers = self.config.get("early_layers")
        if early_layers is not None:
            early_layers = [int(layer) for layer in early_layers]
        relative_top = self.config.get("relative_top")
        relative_top_min_tokens = int(self.config.get("relative_top_min_tokens", 1))
        temperature = float(self.config.get("temperature", 1.0))
        top_k = self.config.get("top_k")
        top_p = self.config.get("top_p")
        repetition_penalty = self.config.get("repetition_penalty")
        if repetition_penalty is not None:
            repetition_penalty = float(repetition_penalty)
        post_softmax = bool(self.config.get("post_softmax", True))
        eos_token_id = gen_cfg.get("eos_token_id", None)
        if eos_token_id is None and "eos_token_id_text" in gen_cfg:
            eos_text = gen_cfg.get("eos_token_id_text")
            eos_ids = model.tokenizer.encode(eos_text, add_special_tokens=False)
            if eos_ids:
                eos_token_id = eos_ids[-1]
        max_prompt_tokens = gen_cfg.get("max_prompt_tokens")
        if max_prompt_tokens is None:
            max_prompt_tokens = getattr(model.model.config, "max_position_embeddings", None)
            if max_prompt_tokens is None:
                max_prompt_tokens = getattr(model.model.config, "n_positions", None)
        strategy = self.config.get("strategy", "fixed")
        return dola_decode(
            model.model,
            model.tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            early_layer=early_layer,
            early_layers=early_layers,
            alpha=alpha,
            relative_top=relative_top,
            relative_top_min_tokens=relative_top_min_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            post_softmax=post_softmax,
            eos_token_id=eos_token_id,
            strategy=strategy,
            max_prompt_tokens=max_prompt_tokens,
        )
