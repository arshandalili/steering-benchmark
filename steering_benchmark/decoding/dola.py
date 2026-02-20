"""DoLa-style decoding utilities."""
from __future__ import annotations

import math
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
    mask = cumulative > p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(1, sorted_indices, sorted_logits)
    return filtered


def _js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    kl_pm = (p * (p.clamp_min(1e-9).log() - m.clamp_min(1e-9).log())).sum(dim=-1)
    kl_qm = (q * (q.clamp_min(1e-9).log() - m.clamp_min(1e-9).log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def _apply_repetition_penalty(
    scores: torch.Tensor, input_ids: torch.Tensor, penalty: float
) -> torch.Tensor:
    if penalty is None or penalty == 1.0 or input_ids is None:
        return scores
    scores = scores.clone()
    for batch_idx in range(scores.size(0)):
        for token_id in set(input_ids[batch_idx].tolist()):
            token_score = scores[batch_idx, token_id]
            if token_score < 0:
                scores[batch_idx, token_id] = token_score * penalty
            else:
                scores[batch_idx, token_id] = token_score / penalty
    return scores


def _relative_top_mask(
    logits: torch.Tensor,
    relative_top: float,
    min_tokens_to_keep: int = 1,
) -> torch.Tensor | None:
    if relative_top is None or relative_top <= 0 or relative_top >= 1:
        return None
    scores_normalized = F.log_softmax(logits, dim=-1)
    sorted_logits, _ = torch.sort(scores_normalized, descending=True)
    min_thresh = sorted_logits[:, max(min_tokens_to_keep - 1, 0)]
    probs_max = scores_normalized.max(dim=-1).values
    probs_thresh = torch.min(min_thresh, probs_max + math.log(relative_top))
    return scores_normalized < probs_thresh.unsqueeze(-1)


def dola_decode(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 32,
    early_layer: int = 4,
    early_layers: list[int] | None = None,
    alpha: float = 1.0,
    relative_top: float | None = None,
    relative_top_min_tokens: int = 1,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    post_softmax: bool = True,
    eos_token_id: int | None = None,
    strategy: str = "fixed",
    max_prompt_tokens: int | None = None,
) -> str:
    device = next(model.parameters()).device
    if max_prompt_tokens is None:
        max_prompt_tokens = getattr(tokenizer, "model_max_length", None)
        if max_prompt_tokens is None or max_prompt_tokens > 100000:
            max_prompt_tokens = 2048
        model_max = getattr(getattr(model, "config", None), "max_position_embeddings", None)
        if model_max is None:
            model_max = getattr(getattr(model, "config", None), "n_positions", None)
        if model_max is not None:
            max_prompt_tokens = min(max_prompt_tokens, int(model_max))
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
    ).to(device)["input_ids"]
    if eos_token_id is None and getattr(tokenizer, "eos_token_id", None) is not None:
        eos_token_id = tokenizer.eos_token_id
    generated = []
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        lm_head = model.get_output_embeddings()
        last_hidden = hidden_states[-1][:, -1, :]
        last_logits = lm_head(last_hidden) / max(temperature, 1e-6)

        if strategy == "max_jsd" and early_layers:
            probs_last = F.softmax(last_logits, dim=-1)
            best_layer = None
            best_score = None
            for layer_idx in early_layers:
                layer_idx = min(max(0, int(layer_idx)), len(hidden_states) - 2)
                early_hidden = hidden_states[layer_idx][:, -1, :]
                early_logits = lm_head(early_hidden) / max(temperature, 1e-6)
                probs_early = F.softmax(early_logits, dim=-1)
                jsd = _js_divergence(probs_last, probs_early)
                score = jsd.mean().item()
                if best_score is None or score > best_score:
                    best_score = score
                    best_layer = layer_idx
            layer_idx = best_layer if best_layer is not None else early_layer
        else:
            layer_idx = min(max(0, int(early_layer)), len(hidden_states) - 2)

        early_hidden = hidden_states[layer_idx][:, -1, :]
        early_logits = lm_head(early_hidden) / max(temperature, 1e-6)
        final_log_probs = F.log_softmax(last_logits, dim=-1)
        early_log_probs = F.log_softmax(early_logits, dim=-1)
        logits = final_log_probs - alpha * early_log_probs
        if post_softmax:
            logits = F.log_softmax(logits, dim=-1)

        if relative_top is not None:
            mask = _relative_top_mask(last_logits, relative_top, min_tokens_to_keep=relative_top_min_tokens)
            if mask is not None:
                logits = logits.masked_fill(mask, float("-inf"))

        if repetition_penalty is not None and repetition_penalty != 1.0:
            logits = _apply_repetition_penalty(logits, input_ids, repetition_penalty)

        if top_k:
            logits = _top_k_filter(logits, int(top_k))
        if top_p:
            logits = _top_p_filter(logits, float(top_p))

        next_token = torch.argmax(logits, dim=-1, keepdim=True).to(input_ids.device)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if max_prompt_tokens and input_ids.size(1) > max_prompt_tokens:
            input_ids = input_ids[:, -max_prompt_tokens:]
        generated.append(next_token)
        if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
            break
    if not generated:
        return ""
    out_ids = torch.cat(generated, dim=-1)[0]
    return tokenizer.decode(out_ids, skip_special_tokens=True)
