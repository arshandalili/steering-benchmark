"""Rank-1 representation finetuning baseline."""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import torch

from steering_benchmark.training.utils import build_prompt_target_pairs


def _compute_nll(model, tokenizer, prompts: List[str], targets: List[str]) -> torch.Tensor:
    device = next(model.parameters()).device
    losses = []
    for prompt, target in zip(prompts, targets):
        text = prompt + " " + target
        inputs = tokenizer(text, return_tensors="pt").to(device)
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        prompt_len = prompt_ids.shape[1]
        outputs = model(**inputs, return_dict=True)
        logits = outputs.logits[:, :-1, :]
        labels = inputs["input_ids"][:, 1:]
        mask = torch.zeros_like(labels, dtype=torch.float32)
        mask[:, prompt_len - 1 :] = 1.0
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        nll = -(token_log_probs * mask).sum() / mask.sum().clamp_min(1.0)
        losses.append(nll)
    return torch.stack(losses).mean()


def train_reft(model_adapter, dataset, run_cfg: Dict, reft_cfg: Dict) -> Dict:
    hf_model = model_adapter.model
    tokenizer = model_adapter.tokenizer
    device = model_adapter.device

    for param in hf_model.parameters():
        param.requires_grad = False
    hf_model.train()

    train_cfg = run_cfg.get("train", {})
    group_pos = train_cfg.get("group_pos", train_cfg.get("group_a", "context"))
    group_neg = train_cfg.get("group_neg", train_cfg.get("group_b", None))
    split = train_cfg.get("split", run_cfg.get("splits", {}).get("train"))
    limit = int(train_cfg.get("max_examples", 256))
    lr = float(train_cfg.get("lr", 5e-4))
    max_steps = int(train_cfg.get("max_steps", 50))

    layer = int(reft_cfg.get("layer", 0))
    token_position = reft_cfg.get("token_position", "last")

    hidden_dim = hf_model.config.hidden_size
    u = torch.nn.Parameter(torch.randn(hidden_dim, device=device) * 0.02)
    v = torch.nn.Parameter(torch.randn(hidden_dim, device=device) * 0.02)

    def hook(module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if token_position == "all":
            proj = hidden @ u
            hidden = hidden + proj.unsqueeze(-1) * v
        else:
            idx = -1 if token_position == "last" else int(token_position)
            proj = hidden[:, idx, :] @ u
            hidden = hidden.clone()
            hidden[:, idx, :] = hidden[:, idx, :] + proj.unsqueeze(-1) * v
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    module_path = model_adapter.hook_module.format(layer=layer)
    module = hf_model.get_submodule(module_path)
    handle = module.register_forward_hook(hook)

    optimizer = torch.optim.AdamW([u, v], lr=lr)
    start = time.time()
    step = 0
    total_tokens = 0

    pos_pairs = build_prompt_target_pairs(dataset, group_pos, limit=limit, split=split)
    neg_pairs = build_prompt_target_pairs(dataset, group_neg, limit=limit, split=split) if group_neg else []

    for step in range(max_steps):
        idx = step % len(pos_pairs)
        prompt, target = pos_pairs[idx]
        pos_loss = _compute_nll(hf_model, tokenizer, [prompt], [target])
        loss = pos_loss
        total_tokens += len(tokenizer(prompt + " " + target)["input_ids"])
        if neg_pairs:
            nidx = step % len(neg_pairs)
            n_prompt, n_target = neg_pairs[nidx]
            neg_loss = _compute_nll(hf_model, tokenizer, [n_prompt], [n_target])
            loss = pos_loss - neg_loss
            total_tokens += len(tokenizer(n_prompt + " " + n_target)["input_ids"])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    elapsed = time.time() - start
    handle.remove()
    return {
        "train_steps": step + 1,
        "train_tokens": total_tokens,
        "train_seconds": elapsed,
        "trainable_params": u.numel() + v.numel(),
        "u": u.detach().cpu(),
        "v": v.detach().cpu(),
        "layer": layer,
        "token_position": token_position,
    }
