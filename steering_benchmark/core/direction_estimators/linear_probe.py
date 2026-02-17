"""Linear probe direction estimator."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from steering_benchmark.core.direction_estimators.base import DirectionEstimator


class LinearProbeEstimator(DirectionEstimator):
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
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except Exception as exc:
            raise ImportError("scikit-learn is required for linear_probe estimator") from exc

        train_cfg = run_cfg.get("train", {})
        limit = int(train_cfg.get("max_examples", 512))
        batch_size = int(train_cfg.get("batch_size", 8))
        c_value = float(train_cfg.get("probe_c", 1.0))
        max_iter = int(train_cfg.get("probe_max_iter", 1000))
        fit_intercept = bool(train_cfg.get("probe_fit_intercept", True))
        class_weight = train_cfg.get("probe_class_weight")
        standardize = bool(train_cfg.get("probe_standardize", False))

        pos_hidden = self._collect_hidden(model, dataset, group_pos, layer, token_position, limit, batch_size, split)
        neg_hidden = self._collect_hidden(model, dataset, group_neg, layer, token_position, limit, batch_size, split)
        if pos_hidden.size == 0 or neg_hidden.size == 0:
            raise RuntimeError("No examples found for linear probe training")

        x = np.concatenate([pos_hidden, neg_hidden], axis=0)
        y = np.concatenate([np.ones(pos_hidden.shape[0]), np.zeros(neg_hidden.shape[0])], axis=0)

        scaler = None
        if standardize:
            scaler = StandardScaler()
            x = scaler.fit_transform(x)

        clf = LogisticRegression(
            C=c_value,
            max_iter=max_iter,
            solver="liblinear",
            fit_intercept=fit_intercept,
            class_weight=class_weight,
        )
        clf.fit(x, y)
        coef = clf.coef_[0]
        if scaler is not None:
            coef = coef / scaler.scale_
        direction = torch.tensor(coef, dtype=torch.float32)
        return direction
