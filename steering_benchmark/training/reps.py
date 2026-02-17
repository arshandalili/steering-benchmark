"""Reference-free preference steering helper."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def train_reps(model_adapter, dataset, run_cfg: Dict, reps_cfg: Dict) -> Dict:
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:
        raise ImportError("scikit-learn is required for RePS training") from exc

    train_cfg = run_cfg.get("train", {})
    group_pos = train_cfg.get("group_pos", train_cfg.get("group_a", "context"))
    group_neg = train_cfg.get("group_neg", train_cfg.get("group_b", "param"))
    split = train_cfg.get("split", run_cfg.get("splits", {}).get("train"))
    limit = int(train_cfg.get("max_examples", 512))
    batch_size = int(train_cfg.get("batch_size", 8))
    layer = int(reps_cfg.get("layer", 0))
    token_position = reps_cfg.get("token_position", "last")

    def _collect(group: str) -> np.ndarray:
        examples = dataset.iter_group(group, limit=limit, split=split)
        batch = []
        vecs = []
        for ex in examples:
            batch.append(ex.prompt)
            if len(batch) >= batch_size:
                hidden = model_adapter.encode_hidden(batch, layer=layer, token_position=token_position).float()
                vecs.append(hidden.cpu().numpy())
                batch = []
        if batch:
            hidden = model_adapter.encode_hidden(batch, layer=layer, token_position=token_position).float()
            vecs.append(hidden.cpu().numpy())
        return np.concatenate(vecs, axis=0) if vecs else np.zeros((0, 0))

    pos_hidden = _collect(group_pos)
    neg_hidden = _collect(group_neg)
    if pos_hidden.size == 0 or neg_hidden.size == 0:
        raise RuntimeError("No examples found for RePS training")
    x = np.concatenate([pos_hidden, neg_hidden], axis=0)
    y = np.concatenate([np.ones(pos_hidden.shape[0]), np.zeros(neg_hidden.shape[0])], axis=0)
    clf = LogisticRegression(C=float(reps_cfg.get("C", 1.0)), max_iter=int(reps_cfg.get("max_iter", 1000)))
    clf.fit(x, y)
    direction = torch.tensor(clf.coef_[0], dtype=torch.float32)
    return {
        "direction": direction,
        "train_steps": 1,
        "train_tokens": int(x.shape[0]),
        "train_seconds": 0.0,
        "trainable_params": direction.numel(),
    }
