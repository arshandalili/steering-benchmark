"""Contrastive decoding baseline."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional

from steering_benchmark.decoding.cad import contrastive_decode
from steering_benchmark.methods.base import SteeringMethod
from steering_benchmark.registry import register_method


@dataclass
class _DecodingControl:
    scale: float = 1.0

    def scale_by(self, factor: float) -> "_DecodingControl":
        return _DecodingControl(scale=self.scale * float(factor))


@register_method("cad_decoding")
class CADDecodingSteering(SteeringMethod):
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self._amateur_model = None

    @staticmethod
    def _strip_context(prompt: str) -> str:
        # For context QA prompts, CAD's amateur branch should use query-only input.
        chunks = re.split(r"\n\s*\n", prompt.strip())
        stripped_chunks = []
        changed = False
        for chunk in chunks:
            match = re.search(r"question\s*:", chunk, flags=re.IGNORECASE)
            if match:
                stripped_chunks.append(chunk[match.start() :].lstrip())
                changed = True
            else:
                stripped_chunks.append(chunk)
        if changed:
            return "\n\n".join(stripped_chunks)

        match = re.search(r"(question\s*:.*)$", prompt, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).lstrip()
        return prompt

    def _load_amateur(self, model):
        hf_id = self.config.get("amateur_hf_id")
        if not hf_id:
            return None
        if self._amateur_model is not None:
            return self._amateur_model
        try:
            from transformers import AutoModelForCausalLM
        except Exception as exc:
            raise ImportError("transformers is required for CAD decoding") from exc
        import torch

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config.get("amateur_dtype", "bfloat16"), torch.bfloat16)
        device_map = self.config.get("amateur_device_map", "auto")
        trust_remote_code = bool(self.config.get("amateur_trust_remote_code", False))
        self._amateur_model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self._amateur_model.eval()
        return self._amateur_model

    def fit(self, model, dataset, run_cfg: Dict):
        # CAD is decoding-time only; this control object lets factor sweeps scale alpha.
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
        amateur_prompt = self.config.get("amateur_prompt")
        if amateur_prompt is None:
            amateur_template = self.config.get("amateur_template")
            if amateur_template:
                amateur_prompt = amateur_template.format(prompt=prompt)
            else:
                amateur_prompt = self._strip_context(prompt)
        max_new_tokens = int(gen_cfg.get("max_new_tokens", 32))
        alpha = float(self.config.get("alpha", self.config.get("weight", 1.0))) * control_scale
        plausibility_alpha = self.config.get("plausibility_alpha")
        temperature = float(self.config.get("temperature", 1.0))
        amateur_temperature = self.config.get("amateur_temperature")
        if amateur_temperature is not None:
            amateur_temperature = float(amateur_temperature)
        top_k = self.config.get("top_k")
        top_p = self.config.get("top_p")
        eos_token_id = gen_cfg.get("eos_token_id", None)
        if eos_token_id is None and "eos_token_id_text" in gen_cfg:
            eos_text = gen_cfg.get("eos_token_id_text")
            eos_ids = model.tokenizer.encode(eos_text, add_special_tokens=False)
            if eos_ids:
                eos_token_id = eos_ids[-1]

        amateur_model = self._load_amateur(model)
        max_prompt_tokens = gen_cfg.get("max_prompt_tokens")
        if max_prompt_tokens is None:
            max_prompt_tokens = getattr(model.model.config, "max_position_embeddings", None)
            if max_prompt_tokens is None:
                max_prompt_tokens = getattr(model.model.config, "n_positions", None)
        return contrastive_decode(
            expert_model=model.model,
            tokenizer=model.tokenizer,
            prompt=prompt,
            amateur_prompt=amateur_prompt,
            amateur_model=amateur_model,
            max_new_tokens=max_new_tokens,
            alpha=alpha,
            plausibility_alpha=plausibility_alpha,
            temperature=temperature,
            amateur_temperature=amateur_temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
            max_prompt_tokens=max_prompt_tokens,
        )
