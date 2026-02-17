"""HyperSteer-style direction generator."""
from __future__ import annotations

from typing import Dict, List

import torch


def _embed_texts(model_adapter, texts: List[str]) -> torch.Tensor:
    tokenizer = model_adapter.tokenizer
    embeddings = model_adapter.model.get_input_embeddings()
    device = model_adapter.device
    vecs = []
    for text in texts:
        ids = tokenizer(text, return_tensors="pt").to(device)["input_ids"]
        with torch.no_grad():
            emb = embeddings(ids).mean(dim=1)
        vecs.append(emb.squeeze(0))
    return torch.stack(vecs, dim=0)


def train_hypersteer(model_adapter, hyper_cfg: Dict) -> Dict:
    behaviors = hyper_cfg.get("behaviors", [])
    directions = hyper_cfg.get("directions", [])
    target_behavior = hyper_cfg.get("target_behavior")
    if not behaviors or not directions or target_behavior is None:
        raise ValueError("HyperSteer requires behaviors, directions, and target_behavior in config")
    if len(behaviors) != len(directions):
        raise ValueError("HyperSteer behaviors and directions must be same length")

    device = model_adapter.device
    x = _embed_texts(model_adapter, behaviors)  # (B, D)
    y = torch.stack([torch.tensor(d, device=device) for d in directions], dim=0)  # (B, H)

    # Ridge regression to fit linear map from embedding to direction.
    reg = float(hyper_cfg.get("ridge", 1e-3))
    xtx = x.T @ x + reg * torch.eye(x.shape[1], device=device)
    xty = x.T @ y
    w = torch.linalg.solve(xtx, xty)  # (D, H)

    target_vec = _embed_texts(model_adapter, [target_behavior])[0]
    direction = target_vec @ w
    return {
        "direction": direction.detach().cpu(),
        "train_steps": 1,
        "train_tokens": int(x.shape[0]),
        "train_seconds": 0.0,
        "trainable_params": w.numel(),
    }
