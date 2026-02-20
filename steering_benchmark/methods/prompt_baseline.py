"""Prompt-only baseline steering."""
from __future__ import annotations

from typing import Dict

from steering_benchmark.methods.base import SteeringMethod
from steering_benchmark.registry import register_method


@register_method("prompt_baseline")
class PromptBaselineSteering(SteeringMethod):
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.mode = config.get("mode", "context")
        self.prompt_control = config.get("prompt_control", "auto")
        self.separator = config.get("separator", "\n\n")
        self.prefix_context = config.get(
            "prefix_context",
            "Use the provided context to answer the question. If the context is insufficient, say 'unknown'.",
        )
        self.prefix_param = config.get(
            "prefix_param",
            "Answer using only your general knowledge. Ignore any provided context.",
        )

    def fit(self, model, dataset, run_cfg: Dict):
        return None

    def transform_prompt(self, example) -> str:
        mode = self.mode
        if mode == "auto":
            mode = example.group

        # Prefer demonstration-level prompt control when dataset supplies prompt parts.
        if self.prompt_control in {"auto", "demo_rewrite", "demos"}:
            meta = getattr(example, "meta", {}) or {}
            prompt_body = meta.get("prompt_body")
            if prompt_body is not None:
                demo_prefix = None
                if mode == "param":
                    demo_prefix = meta.get("demo_prefix_param_control")
                else:
                    demo_prefix = meta.get("demo_prefix_context_control")
                if demo_prefix is not None:
                    return f"{demo_prefix}{prompt_body}"

        if mode == "param":
            prefix = self.prefix_param
        else:
            prefix = self.prefix_context
        return f"{prefix}{self.separator}{example.prompt}"
