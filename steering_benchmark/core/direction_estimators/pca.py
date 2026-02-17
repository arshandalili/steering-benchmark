"""PCA direction estimator."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from steering_benchmark.core.direction_estimators.base import DirectionEstimator


class PCAEstimator(DirectionEstimator):
    def _collect_hidden(
        self,
        model,
        dataset,
        group: str,
        layer: int,
        token_position: str | int,
        limit: int,
        batch_size: int,
        split: Optional[str],
    ) -> np.ndarray:
        examples = dataset.iter_group(group, limit=limit, split=split)
        batch = []
        vectors = []
        for ex in examples:
            batch.append(ex.prompt)
            if len(batch) >= batch_size:
                hidden = model.encode_hidden(batch, layer=layer, token_position=token_position).float()
                vectors.append(hidden.cpu().numpy())
                batch = []
        if batch:
            hidden = model.encode_hidden(batch, layer=layer, token_position=token_position).float()
            vectors.append(hidden.cpu().numpy())
        if not vectors:
            return np.zeros((0, 0))
        return np.concatenate(vectors, axis=0)

    def fit_direction(
        self,
        model,
        dataset,
        run_cfg: dict,
        layer: int,
        token_position: str | int,
        split: Optional[str],
        group_pos: str,
        group_neg: str,
    ) -> torch.Tensor:
        try:
            from sklearn.decomposition import PCA
        except Exception as exc:
            raise ImportError("scikit-learn is required for PCA estimator") from exc

        train_cfg = run_cfg.get("train", {})
        limit = int(train_cfg.get("max_examples", 512))
        batch_size = int(train_cfg.get("batch_size", 8))
        n_components = int(train_cfg.get("pca_components", 2))
        pos_only = bool(train_cfg.get("pca_pos_only", True))

        pos_hidden = self._collect_hidden(model, dataset, group_pos, layer, token_position, limit, batch_size, split)
        if pos_hidden.size == 0:
            raise RuntimeError("No positive examples found for PCA estimator")
        if pos_only:
            x = pos_hidden
        else:
            neg_hidden = self._collect_hidden(model, dataset, group_neg, layer, token_position, limit, batch_size, split)
            if neg_hidden.size == 0:
                raise RuntimeError("No negative examples found for PCA estimator")
            x = np.concatenate([pos_hidden, neg_hidden], axis=0)
        pca = PCA(n_components=min(n_components, x.shape[1]))
        pca.fit(x)
        components = pca.components_
        if pos_only:
            direction = torch.tensor(components[0], dtype=torch.float32)
            return direction
        mean_diff = pos_hidden.mean(axis=0) - neg_hidden.mean(axis=0)
        scores = np.abs(components @ mean_diff)
        idx = int(np.argmax(scores))
        direction = torch.tensor(components[idx], dtype=torch.float32)
        return direction
