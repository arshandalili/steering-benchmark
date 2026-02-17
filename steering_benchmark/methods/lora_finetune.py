"""LoRA finetuning baseline."""
from __future__ import annotations

from typing import Dict

from steering_benchmark.methods.base import SteeringMethod
from steering_benchmark.registry import register_method
from steering_benchmark.training.lora import train_lora


@register_method("lora_finetune")
class LoRASteering(SteeringMethod):
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self._cost = {}

    def adapt_model(self, model, dataset, run_cfg: Dict):
        lora_cfg = self.config.get("lora", {})
        self._cost = train_lora(model, dataset, run_cfg, lora_cfg)
        return model

    def fit(self, model, dataset, run_cfg: Dict):
        return None

    def get_cost(self) -> Dict[str, float]:
        return self._cost
