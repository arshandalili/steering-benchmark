"""HuggingFace model adapter with hook support."""
from __future__ import annotations

from typing import Optional

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

        self.model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, use_fast=use_fast)
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
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[layer + 1]
        selected = self._select_hidden(hidden, inputs["attention_mask"], token_position)
        return selected.detach().cpu()

    def generate(self, prompt: str, intervention=None, gen_cfg: Optional[dict] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        gen_cfg = gen_cfg or {}
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

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
