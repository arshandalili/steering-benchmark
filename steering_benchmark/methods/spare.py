"""Sparse autoencoder-based steering (SpARE-style)."""
from __future__ import annotations

import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

import numpy as np
import torch

from steering_benchmark.core.hooks import LayerHook
from steering_benchmark.core.intervention import InterventionPlan, InterventionSpec
from steering_benchmark.core.metrics import _iter_targets, _normalize
from steering_benchmark.methods.base import SteeringMethod
from steering_benchmark.datasets.base import TargetType
from steering_benchmark.registry import register_method


def _postprocess_prediction(pred: str, policy: Optional[str]) -> str:
    if pred is None:
        return ""
    if not policy or policy == "none":
        return pred
    text = pred.strip()
    lowered = text.lower()

    if policy in {"answer_only", "auto"}:
        for marker in ("answer:", "assistant:", "response:"):
            idx = lowered.rfind(marker)
            if idx != -1:
                text = text[idx + len(marker) :].strip()
                lowered = text.lower()
                break
        text = re.sub(r"^(the answer is|answer is|it is|it's)\s+", "", text, flags=re.IGNORECASE).strip()

    if policy in {"first_line", "line", "auto"}:
        lines = text.splitlines()
        text = lines[0].strip() if lines else ""

    if policy in {"first_sentence", "auto"} and text:
        parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
        text = parts[0].strip()

    return text


def _is_match(pred: str, target) -> bool:
    pred_norm = _normalize(pred or "")
    for tgt in _iter_targets(target):
        if pred_norm == _normalize(str(tgt)):
            return True
    return False


def _select_target(target: TargetType) -> str:
    return str(next(iter(_iter_targets(target)), "")) if target is not None else ""


@dataclass
class GroupedExample:
    prompt: str
    target_context: TargetType
    target_param: TargetType


class SparseAutoencoder:
    def __init__(self, sae_lens: Any | None = None) -> None:
        if sae_lens is None:
            raise ValueError("SparseAutoencoder requires a loaded SAE instance.")
        self.sae_lens = sae_lens
        self._mode = "transformer_lens_sae" if hasattr(sae_lens, "pre_acts") and hasattr(sae_lens, "W_dec") else "sae_lens"
        self._device: torch.device | None = None

    def _ensure_device(self, device: torch.device) -> None:
        if self._device == device:
            return
        self._device = device
        try:
            self.sae_lens.to(device)
        except Exception:
            pass

    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        self._ensure_device(hidden.device)
        sae_dtype = getattr(self.sae_lens, "dtype", None)
        hidden_in = hidden.to(sae_dtype) if sae_dtype is not None and hidden.dtype != sae_dtype else hidden
        if self._mode == "transformer_lens_sae":
            return self.sae_lens.pre_acts(hidden_in)
        return self.sae_lens.encode(hidden_in)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        self._ensure_device(features.device)
        sae_dtype = getattr(self.sae_lens, "dtype", None)
        feats_in = features.to(sae_dtype) if sae_dtype is not None and features.dtype != sae_dtype else features
        if self._mode == "transformer_lens_sae":
            out = feats_in @ self.sae_lens.W_dec
            b_dec = getattr(self.sae_lens, "b_dec", None)
            if b_dec is not None:
                out = out + b_dec
            return out
        return self.sae_lens.decode(feats_in)


def _import_transformerlens_sae_class():
    try:
        from spare.sae import Sae

        return Sae
    except Exception:
        pass
    spare_repo = Path("/data/arshan/hallucination/SAE-based-representation-engineering")
    if spare_repo.exists():
        sys.path.insert(0, str(spare_repo))
        try:
            from spare.sae import Sae

            return Sae
        except Exception:
            pass
    try:
        from spare.sae import Sae

        return Sae
    except Exception as exc:
        raise ImportError(
            "SpARE requires `spare.sae.Sae` (TransformerLens-style SAE from SAE-based-representation-engineering)."
        ) from exc


def _resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _load_transformerlens_sae(
    release: str, sae_id: str, device: str, dtype: str, force_download: bool
) -> Any:
    del force_download  # Sae.load_from_hub delegates to snapshot_download and does not expose this flag.
    Sae = _import_transformerlens_sae_class()
    sae = Sae.load_from_hub(name=release, hookpoint=sae_id, device=device, decoder=True)
    try:
        sae = sae.to(dtype=_resolve_dtype(dtype))
    except Exception:
        pass
    for p in sae.parameters():
        p.requires_grad = False
    return sae


class _LocalMistralSAE(torch.nn.Module):
    def __init__(
        self,
        w_enc: torch.Tensor,
        w_dec: torch.Tensor,
        b_enc: torch.Tensor,
        b_dec: torch.Tensor,
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        self.W_enc = torch.nn.Parameter(w_enc, requires_grad=False)
        self.W_dec = torch.nn.Parameter(w_dec, requires_grad=False)
        self.b_enc = torch.nn.Parameter(b_enc, requires_grad=False)
        self.b_dec = torch.nn.Parameter(b_dec, requires_grad=False)
        self.normalize_input = bool(normalize_input)

    @property
    def dtype(self) -> torch.dtype:
        return self.W_enc.dtype

    @property
    def device(self) -> torch.device:
        return self.W_enc.device

    def pre_acts(self, hidden: torch.Tensor) -> torch.Tensor:
        x = hidden.to(self.dtype)
        if self.normalize_input:
            coeff = (x.shape[-1] ** 0.5) / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            x = x * coeff
        return torch.relu(x @ self.W_enc + self.b_enc)


def _prepare_mistral_checkpoint_unpickler() -> None:
    saes_pkg = types.ModuleType("saes")
    cfg_mod = types.ModuleType("saes.config")

    class LanguageModelSAERunnerConfig:  # checkpoint compatibility placeholder
        pass

    cfg_mod.LanguageModelSAERunnerConfig = LanguageModelSAERunnerConfig
    sys.modules.setdefault("saes", saes_pkg)
    sys.modules["saes.config"] = cfg_mod
    try:
        from torch.serialization import add_safe_globals

        add_safe_globals([LanguageModelSAERunnerConfig])
    except Exception:
        pass


def _load_local_mistral_sae(
    path: str,
    device: str,
    dtype: str,
    normalize_input: bool = True,
) -> Any:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Local Mistral SAE checkpoint not found: {ckpt_path}")
    _prepare_mistral_checkpoint_unpickler()
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected local Mistral SAE payload format: {type(state)!r}")
    try:
        w_enc = state["W_enc"].float()
        w_dec = state["W_dec"].float()
        b_enc = state["b_enc"].float()
        b_dec = state["b_dec"].float()
    except KeyError as exc:
        raise KeyError(
            "Local Mistral SAE checkpoint must contain W_enc, W_dec, b_enc, b_dec in state_dict."
        ) from exc
    sae = _LocalMistralSAE(
        w_enc=w_enc,
        w_dec=w_dec,
        b_enc=b_enc,
        b_dec=b_dec,
        normalize_input=normalize_input,
    )
    sae = sae.to(device=torch.device(device), dtype=_resolve_dtype(dtype))
    return sae


def load_sae_from_config(sae_cfg: Dict, layer: int) -> SparseAutoencoder:
    local_format = sae_cfg.get("local_format")
    local_template = sae_cfg.get("local_path_template")
    if local_format == "mistral_sae_pt" and local_template:
        local_path = str(local_template).format(layer=layer)
        device = sae_cfg.get("device", "cpu")
        dtype = sae_cfg.get("dtype", "float32")
        normalize_input = bool(sae_cfg.get("normalize_input", True))
        sae = _load_local_mistral_sae(
            local_path,
            device=device,
            dtype=dtype,
            normalize_input=normalize_input,
        )
        return SparseAutoencoder(sae_lens=sae)

    release = sae_cfg.get("release")
    sae_id = sae_cfg.get("sae_id")
    if not release or not sae_id:
        raise ValueError("SpARE requires TransformerLens SAE config fields: sae.release and sae.sae_id.")
    sae_id = sae_id.format(layer=layer)
    device = sae_cfg.get("device", "cpu")
    dtype = sae_cfg.get("dtype", "float32")
    force_download = bool(sae_cfg.get("force_download", False))
    sae = _load_transformerlens_sae(release, sae_id, device=device, dtype=dtype, force_download=force_download)
    return SparseAutoencoder(sae_lens=sae)


class SpAREIntervention:
    def __init__(
        self,
        sae: SparseAutoencoder,
        z_context: torch.Tensor,
        z_param: torch.Tensor,
        scale: float,
        token_position: str | int,
        target_behavior: str,
        input_dependent: bool = True,
    ) -> None:
        if target_behavior not in {"context", "param"}:
            raise ValueError(f"Unsupported target_behavior: {target_behavior}")
        self.sae = sae
        self.z_context = z_context
        self.z_param = z_param
        self.scale = scale
        self.token_position = token_position
        self.target_behavior = target_behavior
        self.input_dependent = input_dependent
        self._device: torch.device | None = None
        self._z_context: torch.Tensor | None = None
        self._z_param: torch.Tensor | None = None

    def _ensure_device(self, device: torch.device) -> None:
        if self._device == device:
            return
        self._device = device
        self._z_context = self.z_context.to(device)
        self._z_param = self.z_param.to(device)

    def _encode(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.sae.encode(hidden)

    def _decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.sae.decode(features)

    def _delta(self, hidden: torch.Tensor) -> torch.Tensor:
        z = self._encode(hidden)
        if self.input_dependent:
            if self.target_behavior == "param":
                z_minus = torch.minimum(z, self._z_context)
                z_plus = torch.clamp(self._z_param - z, min=0.0)
            else:
                z_minus = torch.minimum(z, self._z_param)
                z_plus = torch.clamp(self._z_context - z, min=0.0)
            return -self._decode(z_minus) + self._decode(z_plus)
        # Input-independent ablation: use fixed zC/zM
        if self.target_behavior == "param":
            return -self._decode(self._z_context) + self._decode(self._z_param)
        return -self._decode(self._z_param) + self._decode(self._z_context)

    def apply(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states is None:
            return hidden_states
        self._ensure_device(hidden_states.device)
        if self.token_position == "all":
            if hidden_states.dim() == 2:
                delta = self._delta(hidden_states).to(hidden_states.dtype)
                return hidden_states + self.scale * delta
            batch, seq, dim = hidden_states.shape
            flat = hidden_states.reshape(-1, dim)
            delta = self._delta(flat).reshape(batch, seq, dim).to(hidden_states.dtype)
            return hidden_states + self.scale * delta
        idx = -1 if self.token_position == "last" else int(self.token_position)
        hidden_states = hidden_states.clone()
        token_states = hidden_states[:, idx, :]
        delta = self._delta(token_states).to(token_states.dtype)
        hidden_states[:, idx, :] = token_states + self.scale * delta
        return hidden_states

    def scale_by(self, factor: float):
        return SpAREIntervention(
            sae=self.sae,
            z_context=self.z_context,
            z_param=self.z_param,
            scale=self.scale * factor,
            token_position=self.token_position,
            target_behavior=self.target_behavior,
            input_dependent=self.input_dependent,
        )


@register_method("spare")
class SpARESteering(SteeringMethod):
    def _batched(self, items: List[str], batch_size: int) -> Iterable[List[str]]:
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    def generate(self, model, prompt: str, intervention=None, gen_cfg: Dict | None = None) -> str:
        # Upstream SpARE patches only the prompt forward (last prompt token),
        # then performs greedy decoding without hooks.
        if intervention is None:
            return super().generate(model, prompt, intervention=None, gen_cfg=gen_cfg)
        if not hasattr(model, "model") or not hasattr(model, "tokenizer") or not hasattr(model, "hook_module"):
            return super().generate(model, prompt, intervention=intervention, gen_cfg=gen_cfg)

        gen_cfg = dict(gen_cfg or {})
        if "eos_token_id_text" in gen_cfg:
            eos_text = gen_cfg.pop("eos_token_id_text")
            eos_ids = model.tokenizer.encode(eos_text, add_special_tokens=False)
            if eos_ids:
                gen_cfg["eos_token_id"] = eos_ids[-1]

        max_new_tokens = int(gen_cfg.pop("max_new_tokens", 16))
        do_sample = bool(gen_cfg.pop("do_sample", False))
        if do_sample:
            # Original SpARE path is greedy-only; keep sampling behavior on adapter path.
            fallback_cfg = dict(gen_cfg)
            fallback_cfg["max_new_tokens"] = max_new_tokens
            fallback_cfg["do_sample"] = True
            return super().generate(model, prompt, intervention=intervention, gen_cfg=fallback_cfg)

        use_cache_default = "gemma" not in str(getattr(model, "hf_id", "")).lower()
        use_cache = bool(gen_cfg.pop("use_cache", use_cache_default))

        max_prompt_tokens = gen_cfg.pop("max_prompt_tokens", None)
        max_length = max_prompt_tokens or getattr(model.tokenizer, "model_max_length", None)
        if max_length is None or max_length > 100000:
            max_length = 2048
        model_max = getattr(model.model.config, "max_position_embeddings", None)
        if model_max is None:
            model_max = getattr(model.model.config, "n_positions", None)
        if model_max is not None:
            max_length = min(max_length, int(model_max))
            max_length = min(max_length, max(1, int(model_max) - max_new_tokens))

        eos_token_id = gen_cfg.get("eos_token_id", model.tokenizer.eos_token_id)
        inputs = model.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(model.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        specs = getattr(intervention, "specs", None)
        if specs is None:
            specs = [intervention]

        handles = []
        try:
            for spec in specs:
                module_path = model.hook_module.format(layer=spec.layer)
                module = model.model.get_submodule(module_path)
                hook = LayerHook(module, spec.intervention)
                handles.append(hook.attach())
            with torch.no_grad():
                outputs = model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()

        generated_ids: List[int] = []
        if use_cache:
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
            for _ in range(max_new_tokens):
                tok = int(next_token[0, 0].item())
                generated_ids.append(tok)
                if eos_token_id is not None and tok == int(eos_token_id):
                    break
                with torch.no_grad():
                    outputs = model.model(
                        input_ids=next_token,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
        else:
            cur_ids = input_ids
            cur_mask = attention_mask
            for _ in range(max_new_tokens):
                with torch.no_grad():
                    outputs = model.model(
                        input_ids=cur_ids,
                        attention_mask=cur_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
                tok = int(next_token[0, 0].item())
                generated_ids.append(tok)
                if eos_token_id is not None and tok == int(eos_token_id):
                    break
                cur_ids = torch.cat([cur_ids, next_token], dim=1)
                if cur_mask is not None:
                    cur_mask = torch.cat(
                        [
                            cur_mask,
                            torch.ones(
                                (cur_mask.size(0), 1),
                                dtype=cur_mask.dtype,
                                device=cur_mask.device,
                            ),
                        ],
                        dim=1,
                    )

        return model.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _resolve_model_name(self, model) -> str:
        hf_id = getattr(model, "hf_id", None)
        if not hf_id:
            return ""
        return Path(str(hf_id).rstrip("/")).name

    def _select_indices_from_mi_expectation(
        self,
        mi_scores: torch.Tensor,
        expectation: torch.Tensor,
        select_topk_proportion: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sort_indices = torch.argsort(mi_scores, descending=True)
        sort_mi = mi_scores.index_select(0, sort_indices)

        target_mi = float(sort_mi.sum().item()) * float(select_topk_proportion)
        cur_cumulate = 0.0
        select_num = 0
        for idx in sort_indices.tolist():
            cur_cumulate += float(mi_scores[idx].item())
            select_num += 1
            if cur_cumulate > target_mi:
                break
        select_num = max(1, select_num)

        context_indices: List[int] = []
        param_indices: List[int] = []
        for idx in sort_indices.tolist():
            exp_val = float(expectation[idx].item())
            if exp_val > 0:
                context_indices.append(idx)
            elif exp_val < 0:
                param_indices.append(idx)
            if len(context_indices) + len(param_indices) >= select_num:
                break

        return (
            torch.tensor(context_indices, dtype=torch.long),
            torch.tensor(param_indices, dtype=torch.long),
        )

    def _load_cached_functional_vectors(
        self,
        sae_cfg: Dict,
        model_name: str,
        layer: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        cache_root = Path(
            sae_cfg.get(
                "cache_root",
                "/data/arshan/hallucination/SAE-based-representation-engineering/cache_data",
            )
        )
        hiddens_name = sae_cfg.get("hiddens_name", "grouped_activations")
        mi_name = sae_cfg.get("mutual_information_save_name", "mutual_information")
        select_topk_proportion = sae_cfg.get(
            "select_topk_proportion",
            sae_cfg.get("topk_proportion"),
        )
        if select_topk_proportion is None:
            return None

        func_dir = cache_root / model_name / "func_weights" / hiddens_name / f"layer{layer}"
        use_context_path = func_dir / "use_context_weight.pt"
        use_param_path = func_dir / "use_parameter_weight.pt"
        mi_path = cache_root / model_name / mi_name / f"layer-{layer} mi_expectation.pt"
        if not use_context_path.exists() or not use_param_path.exists() or not mi_path.exists():
            return None

        use_context_weight = torch.load(use_context_path, map_location="cpu").float()
        use_param_weight = torch.load(use_param_path, map_location="cpu").float()
        mi_expectation = torch.load(mi_path, map_location="cpu")
        mi_scores = mi_expectation["mi_scores"].float().cpu()
        expectation = mi_expectation["expectation"].float().cpu()

        context_idx, param_idx = self._select_indices_from_mi_expectation(
            mi_scores=mi_scores,
            expectation=expectation,
            select_topk_proportion=float(select_topk_proportion),
        )
        z_context = torch.zeros_like(use_context_weight)
        z_param = torch.zeros_like(use_param_weight)
        if context_idx.numel() > 0:
            z_context[context_idx] = use_context_weight[context_idx]
        if param_idx.numel() > 0:
            z_param[param_idx] = use_param_weight[param_idx]
        return z_context, z_param

    def _group_prompts(
        self,
        model,
        dataset,
        run_cfg: Dict,
    ) -> Tuple[List[GroupedExample], List[GroupedExample]]:
        grouping_cfg = run_cfg.get("grouping", {})
        eval_cfg = run_cfg.get("eval", {})
        splits_cfg = run_cfg.get("splits", {})

        group = grouping_cfg.get("group", eval_cfg.get("group", "context"))
        split = grouping_cfg.get("split", splits_cfg.get("train"))
        max_examples = grouping_cfg.get("max_examples", run_cfg.get("train", {}).get("max_examples", 512))
        gen_cfg = grouping_cfg.get("generation", eval_cfg.get("generation", {"max_new_tokens": 8, "do_sample": False}))
        answer_extraction = grouping_cfg.get("answer_extraction", eval_cfg.get("answer_extraction", "first_line"))

        prompts_context: List[GroupedExample] = []
        prompts_param: List[GroupedExample] = []
        for ex in dataset.iter_group(group, limit=max_examples, split=split):
            if ex.targets is None:
                raise ValueError("SpARE requires dataset examples with both context and param targets.")
            ctx_target = ex.target if ex.group == "context" else ex.targets.get("context")
            param_target = ex.target if ex.group == "param" else ex.targets.get("param")
            if ctx_target is None or param_target is None:
                continue
            pred_raw = model.generate(ex.prompt, intervention=None, gen_cfg=gen_cfg)
            pred = _postprocess_prediction(pred_raw, answer_extraction)
            match_ctx = _is_match(pred, ctx_target)
            match_param = _is_match(pred, param_target)
            if match_ctx and not match_param:
                prompts_context.append(
                    GroupedExample(prompt=ex.prompt, target_context=ctx_target, target_param=param_target)
                )
            elif match_param and not match_ctx:
                prompts_param.append(
                    GroupedExample(prompt=ex.prompt, target_context=ctx_target, target_param=param_target)
                )
        return prompts_context, prompts_param

    def _collect_features(
        self,
        model,
        prompts: List[GroupedExample],
        layer: int,
        token_position: str | int,
        batch_size: int,
        sae: SparseAutoencoder,
    ) -> torch.Tensor:
        feats: List[torch.Tensor] = []
        sae_device = getattr(sae.sae_lens, "device", None)
        if sae_device is None:
            try:
                sae_device = next(sae.sae_lens.parameters()).device
            except Exception:
                sae_device = torch.device("cpu")
        sae_device = torch.device(sae_device)
        for batch in self._batched([item.prompt for item in prompts], batch_size):
            hidden = model.encode_hidden(batch, layer=layer, token_position=token_position).float()
            hidden = hidden.to(sae_device)
            feats.append(sae.encode(hidden).cpu())
        if not feats:
            raise RuntimeError("No features collected for SpARE.")
        return torch.cat(feats, dim=0)

    def _confidence_weights(self, model, examples: List[GroupedExample], numerator: str) -> torch.Tensor:
        weights: List[float] = []
        for ex in examples:
            ctx_target = _select_target(ex.target_context)
            param_target = _select_target(ex.target_param)
            if not ctx_target or not param_target:
                weights.append(1.0)
                continue
            # model.loglikelihood returns NLL; upstream SpARE weights use the
            # opposite-answer loss in the numerator (ss/so cross weighting).
            nll_ctx = float(model.loglikelihood(ex.prompt, ctx_target))
            nll_param = float(model.loglikelihood(ex.prompt, param_target))
            denom = nll_ctx + nll_param
            if denom == 0:
                weights.append(0.5)
            elif numerator == "context":
                weights.append(nll_param / denom)
            else:
                weights.append(nll_ctx / denom)
        weights_t = torch.tensor(weights, dtype=torch.float32)
        total = float(weights_t.sum())
        if total <= 0:
            weights_t = torch.ones_like(weights_t) / max(len(weights), 1)
        else:
            weights_t = weights_t / total
        return weights_t

    def _mutual_information(
        self,
        feats_context: torch.Tensor,
        feats_param: torch.Tensor,
        chunk_size: int,
    ) -> np.ndarray:
        try:
            from sklearn.feature_selection import mutual_info_classif
        except Exception as exc:
            raise ImportError("scikit-learn is required for SpARE mutual information") from exc
        x = torch.cat([feats_context, feats_param], dim=0).cpu().numpy()
        y = np.concatenate(
            [np.ones(len(feats_context), dtype=np.int32), np.zeros(len(feats_param), dtype=np.int32)]
        )
        num_features = x.shape[1]
        mi = np.zeros(num_features, dtype=np.float32)
        for start in range(0, num_features, chunk_size):
            end = min(start + chunk_size, num_features)
            mi[start:end] = mutual_info_classif(x[:, start:end], y, discrete_features=False)
        return mi

    def _functional_activations(
        self,
        feats_context: torch.Tensor,
        feats_param: torch.Tensor,
        weights_context: Optional[torch.Tensor],
        weights_param: Optional[torch.Tensor],
        topk: Optional[int],
        topk_prop: Optional[float],
        mi_prop: Optional[float],
        mi_chunk: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if weights_context is not None:
            z_context = (weights_context.unsqueeze(1) * feats_context).sum(dim=0)
        else:
            z_context = feats_context.mean(dim=0)
        if weights_param is not None:
            z_param = (weights_param.unsqueeze(1) * feats_param).sum(dim=0)
        else:
            z_param = feats_param.mean(dim=0)
        diff = z_context - z_param

        mi = self._mutual_information(feats_context, feats_param, chunk_size=mi_chunk)
        mi_sorted = np.sort(mi)[::-1]
        if topk_prop is not None:
            total = float(mi_sorted.sum())
            if total > 0:
                target = float(topk_prop) * total
                cum = np.cumsum(mi_sorted)
                topk = int(np.searchsorted(cum, target, side="left")) + 1
            else:
                topk = 1 if topk is None else topk
        elif mi_prop is not None:
            total = float(mi_sorted.sum())
            if total > 0:
                target = float(mi_prop) * total
                cum = np.cumsum(mi_sorted)
                topk = int(np.searchsorted(cum, target, side="left")) + 1
            else:
                topk = 1 if topk is None else topk
        if topk is None:
            if mi_prop is None:
                raise ValueError("SpARE requires mi_proportion, topk, or topk_proportion to select MI features.")
            topk = 1
        topk = max(1, int(topk))
        indices = np.argpartition(mi, -topk)[-topk:]

        mask = torch.zeros_like(z_context, dtype=torch.bool)
        mask[torch.tensor(indices, dtype=torch.long)] = True

        zc = torch.where(mask & (diff > 0), z_context, torch.zeros_like(z_context))
        zm = torch.where(mask & (diff <= 0), z_param, torch.zeros_like(z_param))
        return zc, zm

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
        topk = sae_cfg.get("select_topk", sae_cfg.get("topk"))
        topk_prop = sae_cfg.get("select_topk_proportion", sae_cfg.get("topk_proportion"))
        mi_prop = sae_cfg.get("mi_proportion", sae_cfg.get("select_k"))
        mi_chunk = int(sae_cfg.get("mi_chunk_size", 4096))
        use_confidence = bool(sae_cfg.get("use_confidence_weights", True))
        use_upstream_cache = bool(sae_cfg.get("use_upstream_cache", True))
        if not sae_cfg.get("release") or not sae_cfg.get("sae_id"):
            if not (sae_cfg.get("local_format") == "mistral_sae_pt" and sae_cfg.get("local_path_template")):
                raise ValueError("SpARE requires either TransformerLens SAE config fields (sae.release + sae.sae_id) "
                                 "or local Mistral SAE fields (sae.local_format + sae.local_path_template).")

        grouping_cfg = run_cfg.get("grouping", {})
        batch_size = int(grouping_cfg.get("batch_size", run_cfg.get("train", {}).get("batch_size", 8)))
        model_name = self._resolve_model_name(model)
        prompts_context: List[GroupedExample] = []
        prompts_param: List[GroupedExample] = []

        specs = []
        for layer in layers:
            sae = load_sae_from_config(sae_cfg, layer)
            cached_vectors = None
            if use_upstream_cache and model_name:
                cached_vectors = self._load_cached_functional_vectors(
                    sae_cfg=sae_cfg,
                    model_name=model_name,
                    layer=layer,
                )

            if cached_vectors is not None:
                z_context, z_param = cached_vectors
            else:
                if not prompts_context and not prompts_param:
                    prompts_context, prompts_param = self._group_prompts(model, dataset, run_cfg)
                    if not prompts_context or not prompts_param:
                        raise RuntimeError("SpARE requires both context and param groups after behavior grouping.")
                feats_context = self._collect_features(model, prompts_context, layer, token_position, batch_size, sae)
                feats_param = self._collect_features(model, prompts_param, layer, token_position, batch_size, sae)
                weights_context = None
                weights_param = None
                if use_confidence:
                    weights_context = self._confidence_weights(model, prompts_context, numerator="context")
                    weights_param = self._confidence_weights(model, prompts_param, numerator="param")
                z_context, z_param = self._functional_activations(
                    feats_context,
                    feats_param,
                    weights_context,
                    weights_param,
                    topk=topk,
                    topk_prop=topk_prop,
                    mi_prop=mi_prop,
                    mi_chunk=mi_chunk,
                )
            intervention = SpAREIntervention(
                sae=sae,
                z_context=z_context,
                z_param=z_param,
                scale=scale,
                token_position=token_position,
                target_behavior=target_behavior,
                input_dependent=True,
            )
            specs.append(InterventionSpec(layer=layer, intervention=intervention))

        if len(specs) == 1:
            return specs[0]
        return InterventionPlan(specs=specs)
