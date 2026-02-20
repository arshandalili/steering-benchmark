"""Contrastive decoding utilities."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k is None or k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < min_values, float("-inf"))


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p is None or p <= 0 or p >= 1:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)
    # Remove tokens with cumulative probability above the threshold
    mask = cumulative > p
    # Shift mask right to keep at least one token
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(1, sorted_indices, sorted_logits)
    return filtered


def contrastive_decode(
    expert_model,
    tokenizer,
    prompt: str,
    amateur_prompt: str | None = None,
    amateur_model=None,
    max_new_tokens: int = 32,
    alpha: float = 1.0,
    plausibility_alpha: float | None = None,
    temperature: float = 1.0,
    amateur_temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
    max_prompt_tokens: int | None = None,
) -> str:
    device = next(expert_model.parameters()).device
    if max_prompt_tokens is None:
        max_prompt_tokens = getattr(tokenizer, "model_max_length", None)
        if max_prompt_tokens is None or max_prompt_tokens > 100000:
            max_prompt_tokens = 2048
        model_max = getattr(getattr(expert_model, "config", None), "max_position_embeddings", None)
        if model_max is None:
            model_max = getattr(getattr(expert_model, "config", None), "n_positions", None)
        if model_max is not None:
            max_prompt_tokens = min(max_prompt_tokens, int(model_max))
    expert_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
    ).to(device)["input_ids"]
    if amateur_prompt is None:
        amateur_prompt = prompt
    if amateur_model is None:
        amateur_model = expert_model
    amateur_ids = tokenizer(
        amateur_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
    ).to(device)["input_ids"]
    if eos_token_id is None and getattr(tokenizer, "eos_token_id", None) is not None:
        eos_token_id = tokenizer.eos_token_id
    if amateur_temperature is None:
        amateur_temperature = temperature
    generated = []
    for _ in range(max_new_tokens):
        with torch.no_grad():
            expert_out = expert_model(expert_ids, return_dict=True)
            amateur_out = amateur_model(amateur_ids, return_dict=True)
        expert_logits = expert_out.logits[:, -1, :] / max(temperature, 1e-6)
        amateur_logits = amateur_out.logits[:, -1, :] / max(amateur_temperature, 1e-6)

        expert_log_probs = F.log_softmax(expert_logits, dim=-1)
        amateur_log_probs = F.log_softmax(amateur_logits, dim=-1)
        # CAD objective: log p_c(y) + alpha * log(p_c(y) / p_a(y))
        logits = (1.0 + alpha) * expert_log_probs - alpha * amateur_log_probs

        if plausibility_alpha is not None:
            expert_probs = F.softmax(expert_logits, dim=-1)
            max_prob = expert_probs.max(dim=-1, keepdim=True).values
            mask = expert_probs >= (plausibility_alpha * max_prob)
            logits = logits.masked_fill(~mask, float("-inf"))

        if top_k:
            logits = _top_k_filter(logits, int(top_k))
        if top_p:
            logits = _top_p_filter(logits, float(top_p))

        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        expert_ids = torch.cat([expert_ids, next_token], dim=-1)
        amateur_ids = torch.cat([amateur_ids, next_token], dim=-1)
        if max_prompt_tokens and expert_ids.size(1) > max_prompt_tokens:
            expert_ids = expert_ids[:, -max_prompt_tokens:]
            amateur_ids = amateur_ids[:, -max_prompt_tokens:]
        generated.append(next_token)
        if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
            break
    if not generated:
        return ""
    out_ids = torch.cat(generated, dim=-1)[0]
    return tokenizer.decode(out_ids, skip_special_tokens=True)
