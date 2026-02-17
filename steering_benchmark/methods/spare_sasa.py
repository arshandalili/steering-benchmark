"""SpARE-style steering for manifold (grouped) SASA autoencoders."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from steering_benchmark.core.intervention import InterventionPlan, InterventionSpec
from steering_benchmark.methods.spare import SpAREIntervention, SpARESteering, SparseAutoencoder
from steering_benchmark.registry import register_method


def _resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _batch_topk_mask(group_norms: torch.Tensor, k_groups: int) -> torch.Tensor:
    batch_size = group_norms.shape[0]
    flat_norms = group_norms.float().reshape(-1)
    k_total = int(k_groups) * int(batch_size)
    if k_total <= 0:
        return torch.zeros_like(group_norms)
    if k_total >= flat_norms.numel():
        return torch.ones_like(group_norms)
    _, topk_idx = flat_norms.topk(k_total, largest=True, sorted=False)
    mask_f32 = torch.zeros_like(flat_norms, dtype=torch.float32)
    mask_f32[topk_idx] = 1.0
    return mask_f32.view_as(group_norms).to(dtype=group_norms.dtype)


def _per_sample_topk_mask(group_norms: torch.Tensor, k_groups: int) -> torch.Tensor:
    batch_size, n_groups = group_norms.shape
    k = min(max(int(k_groups), 0), int(n_groups))
    if k <= 0:
        return torch.zeros_like(group_norms)
    topk_idx = group_norms.float().topk(k, dim=-1, largest=True, sorted=False).indices
    mask_f32 = torch.zeros((batch_size, n_groups), device=group_norms.device, dtype=torch.float32)
    mask_f32.scatter_(1, topk_idx, 1.0)
    return mask_f32.to(dtype=group_norms.dtype)


class _LocalBatchTopKManifoldSAE(torch.nn.Module):
    def __init__(
        self,
        w_enc: torch.Tensor,
        w_dec: torch.Tensor,
        b_enc: torch.Tensor,
        b_dec: torch.Tensor,
        n_groups: int,
        group_rank: int,
        k_groups: int,
        normalize_activations: str = "none",
        topk_mode: str = "per_sample",
    ) -> None:
        super().__init__()
        self.W_enc = torch.nn.Parameter(w_enc, requires_grad=False)
        self.W_dec = torch.nn.Parameter(w_dec, requires_grad=False)
        self.b_enc = torch.nn.Parameter(b_enc, requires_grad=False)
        self.b_dec = torch.nn.Parameter(b_dec, requires_grad=False)
        self.n_groups = int(n_groups)
        self.group_rank = int(group_rank)
        self.k_groups = int(k_groups)
        self.normalize_activations = str(normalize_activations or "none")
        self.topk_mode = str(topk_mode or "per_sample").strip().lower()
        if self.topk_mode not in {"batch", "per_sample"}:
            raise ValueError(f"Unsupported topk_mode: {self.topk_mode}. Use 'batch' or 'per_sample'.")
        self._ln_mu: Optional[torch.Tensor] = None
        self._ln_std: Optional[torch.Tensor] = None

    @property
    def dtype(self) -> torch.dtype:
        return self.W_enc.dtype

    @property
    def device(self) -> torch.device:
        return self.W_enc.device

    def _normalize_in(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_activations != "layer_norm":
            self._ln_mu = None
            self._ln_std = None
            return x
        mu = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        self._ln_mu = mu
        self._ln_std = std
        return (x - mu) / std

    def _unnormalize_out(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_activations != "layer_norm":
            return x
        if self._ln_mu is None or self._ln_std is None:
            return x
        if self._ln_mu.shape[0] != x.shape[0]:
            return x
        return x * self._ln_std + self._ln_mu

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        needs_flat = x.dim() == 3
        if needs_flat:
            x = x.reshape(-1, x.shape[-1])
        x = x.to(self.dtype)
        sae_in = self._normalize_in(x)
        pre_acts = sae_in @ self.W_enc + self.b_enc
        groups = pre_acts.view(-1, self.n_groups, self.group_rank)
        group_norms = groups.norm(dim=-1)
        if self.topk_mode == "batch":
            mask = _batch_topk_mask(group_norms, self.k_groups)
        else:
            mask = _per_sample_topk_mask(group_norms, self.k_groups)
        active_groups = groups * mask.unsqueeze(-1)
        feature_acts = active_groups.view(-1, self.n_groups * self.group_rank)
        if needs_flat:
            feature_acts = feature_acts.view(original_shape[0], original_shape[1], -1)
        return feature_acts

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        original_shape = feature_acts.shape
        needs_flat = feature_acts.dim() == 3
        if needs_flat:
            feature_acts = feature_acts.reshape(-1, feature_acts.shape[-1])
        feature_acts = feature_acts.to(self.dtype)
        sae_out = feature_acts @ self.W_dec + self.b_dec
        sae_out = self._unnormalize_out(sae_out)
        if needs_flat:
            sae_out = sae_out.view(original_shape[0], original_shape[1], -1)
        return sae_out


def _load_local_batchtopk_manifold_sae(
    local_dir: str,
    device: str,
    dtype: str,
    k_groups: Optional[int] = None,
    normalize_activations: Optional[str] = None,
    topk_mode: Optional[str] = None,
) -> _LocalBatchTopKManifoldSAE:
    root = Path(local_dir)
    if not root.exists():
        raise FileNotFoundError(f"SASA SAE directory not found: {root}")
    cfg_path = root / "cfg.json"
    weights_path = root / "sae_weights.safetensors"
    if not cfg_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"SASA SAE requires cfg.json and sae_weights.safetensors in: {root}")

    cfg = json.loads(cfg_path.read_text())
    if cfg.get("architecture") not in {"batchtopk_manifold_sae", "topk_sasa"}:
        raise ValueError(f"Unsupported SASA architecture: {cfg.get('architecture')}")

    try:
        from safetensors.torch import load_file as load_safetensors
    except Exception as exc:
        raise ImportError("safetensors is required to load SASA SAE weights.") from exc
    state = load_safetensors(str(weights_path))

    for key in ("W_enc", "W_dec", "b_enc", "b_dec"):
        if key not in state:
            raise KeyError(f"SASA SAE weights missing tensor: {key}")

    n_groups = int(cfg["n_groups"])
    group_rank = int(cfg["group_rank"])
    d_sae = int(cfg["d_sae"])
    if d_sae != n_groups * group_rank:
        raise ValueError(
            f"Invalid SASA shape: d_sae={d_sae}, n_groups={n_groups}, group_rank={group_rank}"
        )

    sae = _LocalBatchTopKManifoldSAE(
        w_enc=state["W_enc"].float(),
        w_dec=state["W_dec"].float(),
        b_enc=state["b_enc"].float(),
        b_dec=state["b_dec"].float(),
        n_groups=n_groups,
        group_rank=group_rank,
        k_groups=int(k_groups if k_groups is not None else cfg.get("k_groups", 10)),
        normalize_activations=str(
            normalize_activations if normalize_activations is not None else cfg.get("normalize_activations", "none")
        ),
        topk_mode=str(topk_mode if topk_mode is not None else "per_sample"),
    )
    sae = sae.to(device=torch.device(device), dtype=_resolve_dtype(dtype))
    return sae


def load_sasa_from_config(sae_cfg: Dict, layer: int) -> SparseAutoencoder:
    local_template = sae_cfg.get("local_path_template")
    local_path = sae_cfg.get("local_path")
    if local_template:
        local_path = str(local_template).format(layer=layer)
    if not local_path:
        raise ValueError("SpARE_SASA requires `sae.local_path` or `sae.local_path_template`.")
    device = sae_cfg.get("device", "cpu")
    dtype = sae_cfg.get("dtype", "float32")
    k_groups = sae_cfg.get("k_groups")
    normalize_activations = sae_cfg.get("normalize_activations")
    topk_mode = sae_cfg.get("topk_mode", "per_sample")
    sae = _load_local_batchtopk_manifold_sae(
        local_dir=str(local_path),
        device=device,
        dtype=dtype,
        k_groups=int(k_groups) if k_groups is not None else None,
        normalize_activations=str(normalize_activations) if normalize_activations is not None else None,
        topk_mode=str(topk_mode) if topk_mode is not None else None,
    )
    return SparseAutoencoder(sae_lens=sae)


class SpARESASAIntervention(SpAREIntervention):
    def __init__(
        self,
        sae: SparseAutoencoder,
        z_context: torch.Tensor,
        z_param: torch.Tensor,
        scale: float,
        token_position: str | int,
        target_behavior: str,
        n_groups: int,
        group_rank: int,
        input_dependent: bool = True,
    ) -> None:
        super().__init__(
            sae=sae,
            z_context=z_context,
            z_param=z_param,
            scale=scale,
            token_position=token_position,
            target_behavior=target_behavior,
            input_dependent=input_dependent,
        )
        self.n_groups = int(n_groups)
        self.group_rank = int(group_rank)
        self.d_sae = self.n_groups * self.group_rank

    def _reshape_groups(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[-1] != self.d_sae:
            raise ValueError(f"SASA intervention expected d_sae={self.d_sae}, got {z.shape[-1]}")
        return z.view(z.shape[0], self.n_groups, self.group_rank)

    def _delta(self, hidden: torch.Tensor) -> torch.Tensor:
        z = self._encode(hidden)
        if not self.input_dependent:
            desired = self._z_param if self.target_behavior == "param" else self._z_context
            opposite = self._z_context if self.target_behavior == "param" else self._z_param
            desired_b = desired.unsqueeze(0).expand(z.shape[0], -1)
            opposite_b = opposite.unsqueeze(0).expand(z.shape[0], -1)
            return -self._decode(opposite_b) + self._decode(desired_b)

        z_groups = self._reshape_groups(z)
        z_context_groups = self._z_context.view(1, self.n_groups, self.group_rank)
        z_param_groups = self._z_param.view(1, self.n_groups, self.group_rank)

        if self.target_behavior == "param":
            desired_groups = z_param_groups
            opposite_groups = z_context_groups
        else:
            desired_groups = z_context_groups
            opposite_groups = z_param_groups

        eps = 1e-6
        desired_norm = desired_groups.norm(dim=-1, keepdim=True)
        opposite_norm = opposite_groups.norm(dim=-1, keepdim=True)
        desired_unit = desired_groups / desired_norm.clamp_min(eps)
        opposite_unit = opposite_groups / opposite_norm.clamp_min(eps)

        # Use directional projections inside each 8D group so context/param
        # prototypes are treated as distinct directions within the same subspace.
        proj_desired = (z_groups * desired_unit).sum(dim=-1, keepdim=True)
        proj_opposite = (z_groups * opposite_unit).sum(dim=-1, keepdim=True)

        minus_amt = torch.clamp(proj_opposite, min=0.0)
        minus_amt = torch.minimum(minus_amt, opposite_norm)

        plus_amt = torch.clamp(desired_norm - proj_desired, min=0.0)
        plus_amt = torch.minimum(plus_amt, desired_norm)

        # Decode remove/add parts separately so decoder bias / normalization offsets
        # cancel as in the original SpARE formulation.
        z_minus_groups = minus_amt * opposite_unit
        z_plus_groups = plus_amt * desired_unit
        z_minus = z_minus_groups.reshape(z.shape)
        z_plus = z_plus_groups.reshape(z.shape)
        return -self._decode(z_minus) + self._decode(z_plus)

    def scale_by(self, factor: float):
        return SpARESASAIntervention(
            sae=self.sae,
            z_context=self.z_context,
            z_param=self.z_param,
            scale=self.scale * factor,
            token_position=self.token_position,
            target_behavior=self.target_behavior,
            n_groups=self.n_groups,
            group_rank=self.group_rank,
            input_dependent=self.input_dependent,
        )


@register_method("spare_sasa")
class SpARESASASteering(SpARESteering):
    def _group_shape(self, sae: SparseAutoencoder) -> Tuple[int, int]:
        backend = sae.sae_lens
        n_groups = int(getattr(backend, "n_groups", 0))
        group_rank = int(getattr(backend, "group_rank", 0))
        if n_groups <= 0 or group_rank <= 0:
            raise ValueError("SASA SAE must expose positive n_groups and group_rank.")
        return n_groups, group_rank

    def _directional_fisher_scores(
        self,
        feats_ctx_groups: torch.Tensor,
        feats_param_groups: torch.Tensor,
        z_context_groups: torch.Tensor,
        z_param_groups: torch.Tensor,
    ) -> np.ndarray:
        # Direction per group in 8D latent space.
        dir_groups = z_context_groups - z_param_groups
        dir_norm = dir_groups.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        dir_unit = dir_groups / dir_norm

        proj_ctx = (feats_ctx_groups * dir_unit.unsqueeze(0)).sum(dim=-1)
        proj_param = (feats_param_groups * dir_unit.unsqueeze(0)).sum(dim=-1)

        mean_ctx = proj_ctx.mean(dim=0)
        mean_param = proj_param.mean(dim=0)
        var_ctx = proj_ctx.var(dim=0, unbiased=False)
        var_param = proj_param.var(dim=0, unbiased=False)

        # Down-weight groups with near-zero context/param prototype separation.
        sep = dir_norm.squeeze(-1)
        score = ((mean_ctx - mean_param).pow(2) / (var_ctx + var_param + 1e-6)) * sep
        return score.detach().cpu().numpy().astype(np.float32)

    def _directional_mi_scores(
        self,
        feats_ctx_groups: torch.Tensor,
        feats_param_groups: torch.Tensor,
        z_context_groups: torch.Tensor,
        z_param_groups: torch.Tensor,
        mi_chunk: int,
    ) -> np.ndarray:
        dir_groups = z_context_groups - z_param_groups
        dir_norm = dir_groups.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        dir_unit = dir_groups / dir_norm

        proj_ctx = (feats_ctx_groups * dir_unit.unsqueeze(0)).sum(dim=-1)
        proj_param = (feats_param_groups * dir_unit.unsqueeze(0)).sum(dim=-1)
        return self._mutual_information(proj_ctx, proj_param, chunk_size=mi_chunk)

    def _functional_group_activations(
        self,
        feats_context: torch.Tensor,
        feats_param: torch.Tensor,
        weights_context: Optional[torch.Tensor],
        weights_param: Optional[torch.Tensor],
        topk: Optional[int],
        topk_prop: Optional[float],
        mi_prop: Optional[float],
        mi_chunk: int,
        n_groups: int,
        group_rank: int,
        selection_mode: str = "norm_mi",
        target_behavior: str = "context",
        respect_direction: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if feats_context.shape[-1] != n_groups * group_rank or feats_param.shape[-1] != n_groups * group_rank:
            raise ValueError("SASA features have unexpected dimensionality.")

        feats_ctx_groups = feats_context.view(-1, n_groups, group_rank)
        feats_param_groups = feats_param.view(-1, n_groups, group_rank)

        if weights_context is not None:
            z_context_groups = (weights_context.view(-1, 1, 1) * feats_ctx_groups).sum(dim=0)
        else:
            z_context_groups = feats_ctx_groups.mean(dim=0)
        if weights_param is not None:
            z_param_groups = (weights_param.view(-1, 1, 1) * feats_param_groups).sum(dim=0)
        else:
            z_param_groups = feats_param_groups.mean(dim=0)

        signed_effect: np.ndarray
        if selection_mode == "directional_fisher":
            scores = self._directional_fisher_scores(
                feats_ctx_groups=feats_ctx_groups,
                feats_param_groups=feats_param_groups,
                z_context_groups=z_context_groups,
                z_param_groups=z_param_groups,
            )
            dir_groups = z_context_groups - z_param_groups
            dir_unit = dir_groups / dir_groups.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            proj_ctx = (feats_ctx_groups * dir_unit.unsqueeze(0)).sum(dim=-1)
            proj_param = (feats_param_groups * dir_unit.unsqueeze(0)).sum(dim=-1)
            signed_effect = (proj_ctx.mean(dim=0) - proj_param.mean(dim=0)).detach().cpu().numpy().astype(np.float32)
        elif selection_mode == "directional_mi":
            scores = self._directional_mi_scores(
                feats_ctx_groups=feats_ctx_groups,
                feats_param_groups=feats_param_groups,
                z_context_groups=z_context_groups,
                z_param_groups=z_param_groups,
                mi_chunk=mi_chunk,
            )
            dir_groups = z_context_groups - z_param_groups
            dir_unit = dir_groups / dir_groups.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            proj_ctx = (feats_ctx_groups * dir_unit.unsqueeze(0)).sum(dim=-1)
            proj_param = (feats_param_groups * dir_unit.unsqueeze(0)).sum(dim=-1)
            signed_effect = (proj_ctx.mean(dim=0) - proj_param.mean(dim=0)).detach().cpu().numpy().astype(np.float32)
        else:
            feats_ctx_intensity = feats_ctx_groups.norm(dim=-1)
            feats_param_intensity = feats_param_groups.norm(dim=-1)
            scores = self._mutual_information(feats_ctx_intensity, feats_param_intensity, chunk_size=mi_chunk)
            signed_effect = (
                (feats_ctx_intensity.mean(dim=0) - feats_param_intensity.mean(dim=0))
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        scores_sorted = np.sort(scores)[::-1]
        if topk_prop is not None:
            total = float(scores_sorted.sum())
            if total > 0:
                target = float(topk_prop) * total
                cum = np.cumsum(scores_sorted)
                topk = int(np.searchsorted(cum, target, side="left")) + 1
            else:
                topk = 1 if topk is None else topk
        elif mi_prop is not None:
            total = float(scores_sorted.sum())
            if total > 0:
                target = float(mi_prop) * total
                cum = np.cumsum(scores_sorted)
                topk = int(np.searchsorted(cum, target, side="left")) + 1
            else:
                topk = 1 if topk is None else topk
        if topk is None:
            if mi_prop is None:
                raise ValueError("SpARE_SASA requires mi_proportion, topk, or topk_proportion.")
            topk = 1
        topk = max(1, int(topk))
        indices = np.argpartition(scores, -topk)[-topk:]
        if respect_direction:
            if target_behavior == "param":
                dir_mask = signed_effect[indices] < 0
            else:
                dir_mask = signed_effect[indices] > 0
            filtered = indices[dir_mask]
            if filtered.size > 0:
                indices = filtered
            else:
                # Fallback: keep the strongest signed-effect group among selected.
                idx_local = int(np.argmax(np.abs(signed_effect[indices])))
                indices = np.array([indices[idx_local]], dtype=np.int64)

        group_mask = torch.zeros(n_groups, dtype=torch.bool)
        group_mask[torch.tensor(indices, dtype=torch.long)] = True
        z_context_sel = torch.zeros_like(z_context_groups)
        z_param_sel = torch.zeros_like(z_param_groups)
        # Keep both context and param prototype vectors for each selected group.
        z_context_sel[group_mask] = z_context_groups[group_mask]
        z_param_sel[group_mask] = z_param_groups[group_mask]

        return z_context_sel.reshape(-1), z_param_sel.reshape(-1)

    def fit(self, model, dataset, run_cfg: Dict) -> InterventionSpec:
        layers = self.config.get("layers")
        if layers is None:
            layers = [int(self.config.get("layer", 0))]
        layers = [int(layer) for layer in layers]
        token_position = self.config.get("token_position", "last")
        scale = float(self.config.get("scale", self.config.get("edit_degree", 1.0)))

        target_behavior = self.config.get("target_behavior")
        if target_behavior not in {"context", "param"}:
            target_behavior = run_cfg.get("train", {}).get("group_a", "context")
        if target_behavior not in {"context", "param"}:
            target_behavior = "context"

        sae_cfg = self.config.get("sae", {})
        if not sae_cfg.get("local_path") and not sae_cfg.get("local_path_template"):
            raise ValueError("SpARE_SASA requires `sae.local_path` or `sae.local_path_template`.")

        topk = sae_cfg.get("select_topk", sae_cfg.get("topk"))
        topk_prop = sae_cfg.get("select_topk_proportion", sae_cfg.get("topk_proportion"))
        mi_prop = sae_cfg.get("mi_proportion", sae_cfg.get("select_k"))
        mi_chunk = int(sae_cfg.get("mi_chunk_size", 4096))
        use_confidence = bool(sae_cfg.get("use_confidence_weights", True))
        selection_mode = str(sae_cfg.get("selection_mode", "norm_mi"))
        respect_direction = bool(sae_cfg.get("respect_direction", True))

        grouping_cfg = run_cfg.get("grouping", {})
        batch_size = int(grouping_cfg.get("batch_size", run_cfg.get("train", {}).get("batch_size", 8)))
        prompts_context: List = []
        prompts_param: List = []

        specs = []
        for layer in layers:
            sae = load_sasa_from_config(sae_cfg, layer)
            n_groups, group_rank = self._group_shape(sae)

            if not prompts_context and not prompts_param:
                prompts_context, prompts_param = self._group_prompts(model, dataset, run_cfg)
                if not prompts_context or not prompts_param:
                    raise RuntimeError("SpARE_SASA requires both context and param groups after behavior grouping.")

            feats_context = self._collect_features(model, prompts_context, layer, token_position, batch_size, sae)
            feats_param = self._collect_features(model, prompts_param, layer, token_position, batch_size, sae)
            weights_context = None
            weights_param = None
            if use_confidence:
                weights_context = self._confidence_weights(model, prompts_context, numerator="context")
                weights_param = self._confidence_weights(model, prompts_param, numerator="param")

            z_context, z_param = self._functional_group_activations(
                feats_context=feats_context,
                feats_param=feats_param,
                weights_context=weights_context,
                weights_param=weights_param,
                topk=topk,
                topk_prop=topk_prop,
                mi_prop=mi_prop,
                mi_chunk=mi_chunk,
                n_groups=n_groups,
                group_rank=group_rank,
                selection_mode=selection_mode,
                target_behavior=target_behavior,
                respect_direction=respect_direction,
            )

            intervention = SpARESASAIntervention(
                sae=sae,
                z_context=z_context,
                z_param=z_param,
                scale=scale,
                token_position=token_position,
                target_behavior=target_behavior,
                n_groups=n_groups,
                group_rank=group_rank,
                input_dependent=True,
            )
            specs.append(InterventionSpec(layer=layer, intervention=intervention))

        if len(specs) == 1:
            return specs[0]
        return InterventionPlan(specs=specs)
