"""HuggingFace model adapter with hook support."""
from __future__ import annotations

from typing import Optional
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering_benchmark.core.hooks import LayerHook
from steering_benchmark.models.base import BaseModelAdapter
from steering_benchmark.registry import register_model


@register_model("hf")
class HFModelAdapter(BaseModelAdapter):
    def __init__(
        self,
        hf_id: str,
        dtype: str = "bfloat16",
        device_map: str | dict = "auto",
        trust_remote_code: bool = False,
        hook_module: str = "model.layers.{layer}",
        use_fast: bool = True,
    ) -> None:
        self.hf_id = hf_id
        self.hook_module = hook_module

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            token=hf_token,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_id, use_fast=use_fast, token=hf_token
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _select_hidden(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        token_position: str | int,
    ) -> torch.Tensor:
        if token_position == "all":
            return hidden_states

        if token_position == "last":
            lengths = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_idx, lengths, :]

        idx = int(token_position)
        return hidden_states[:, idx, :]

    def encode_hidden(self, prompts, layer: int, token_position: str | int) -> torch.Tensor:
        max_length = getattr(self.tokenizer, "model_max_length", None)
        if max_length is None or max_length > 100000:
            max_length = 2048
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[layer + 1]
        selected = self._select_hidden(hidden, inputs["attention_mask"], token_position)
        return selected.detach().cpu()

    def generate(self, prompt: str, intervention=None, gen_cfg: Optional[dict] = None) -> str:
        gen_cfg = dict(gen_cfg or {})
        if "eos_token_id_text" in gen_cfg:
            eos_text = gen_cfg.pop("eos_token_id_text")
            eos_ids = self.tokenizer.encode(eos_text, add_special_tokens=False)
            if eos_ids:
                gen_cfg["eos_token_id"] = eos_ids[-1]
        max_prompt_tokens = gen_cfg.pop("max_prompt_tokens", None)
        max_length = max_prompt_tokens or getattr(self.tokenizer, "model_max_length", None)
        if max_length is None or max_length > 100000:
            max_length = 2048
        if self.tokenizer.eos_token_id is not None:
            gen_cfg.setdefault("pad_token_id", self.tokenizer.eos_token_id)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(self.device)
        handles = []
        if intervention is not None:
            specs = getattr(intervention, "specs", None)
            if specs is None:
                specs = [intervention]
            for spec in specs:
                module_path = self.hook_module.format(layer=spec.layer)
                module = self.model.get_submodule(module_path)
                hook = LayerHook(module, spec.intervention)
                handles.append(hook.attach())

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_cfg)
        for handle in handles:
            handle.remove()

        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        prompt_len = inputs["input_ids"].shape[1]
        generated = sequences[0][prompt_len:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)
