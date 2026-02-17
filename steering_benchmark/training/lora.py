"""LoRA training baseline."""
from __future__ import annotations

import inspect
import time
from typing import Dict

import torch
from torch.utils.data import DataLoader

from steering_benchmark.training.utils import PromptTargetDataset, build_prompt_target_pairs, collate_batch


def train_lora(model_adapter, dataset, run_cfg: Dict, lora_cfg: Dict) -> Dict:
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as exc:
        raise ImportError("peft is required for LoRA training") from exc

    hf_model = model_adapter.model
    tokenizer = model_adapter.tokenizer

    train_cfg = run_cfg.get("train", {})
    group_pos = train_cfg.get("group_pos", train_cfg.get("group_a", "context"))
    split = train_cfg.get("split", run_cfg.get("splits", {}).get("train"))
    limit = int(train_cfg.get("max_examples", 512))
    batch_size = int(train_cfg.get("batch_size", 4))
    max_steps = int(train_cfg.get("max_steps", 100))
    lr = float(train_cfg.get("lr", 1e-4))
    max_length = int(train_cfg.get("max_length", 512))

    pairs = build_prompt_target_pairs(dataset, group_pos, limit=limit, split=split)
    train_dataset = PromptTargetDataset(tokenizer, pairs, max_length=max_length)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, pad_token_id=pad_token_id),
    )

    target_modules = lora_cfg.get("target_modules") or lora_cfg.get("lora_components")
    layers_to_transform = (
        lora_cfg.get("layers_to_transform")
        or lora_cfg.get("layers")
        or lora_cfg.get("lora_layers")
    )
    layers_pattern = lora_cfg.get("layers_pattern")
    bias = lora_cfg.get("bias", "none")
    modules_to_save = lora_cfg.get("modules_to_save")
    use_rslora = lora_cfg.get("use_rslora")

    lora_kwargs = dict(
        r=int(lora_cfg.get("r", 8)),
        lora_alpha=int(lora_cfg.get("alpha", 16)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias=bias,
        modules_to_save=modules_to_save,
    )
    if layers_to_transform is not None:
        lora_kwargs["layers_to_transform"] = layers_to_transform
    if layers_pattern is not None:
        lora_kwargs["layers_pattern"] = layers_pattern
    if use_rslora is not None:
        sig = inspect.signature(LoraConfig.__init__)
        if "use_rslora" in sig.parameters:
            lora_kwargs["use_rslora"] = bool(use_rslora)

    lora_config = LoraConfig(**lora_kwargs)
    hf_model = get_peft_model(hf_model, lora_config)
    hf_model.train()
    optimizer = torch.optim.AdamW(hf_model.parameters(), lr=lr)

    start = time.time()
    step = 0
    total_tokens = 0
    for batch in dataloader:
        batch = {k: v.to(model_adapter.device) for k, v in batch.items()}
        outputs = hf_model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_tokens += batch["input_ids"].numel()
        step += 1
        if step >= max_steps:
            break

    elapsed = time.time() - start
    trainable_params = sum(p.numel() for p in hf_model.parameters() if p.requires_grad)
    model_adapter.model = hf_model

    return {
        "train_steps": step,
        "train_tokens": total_tokens,
        "train_seconds": elapsed,
        "trainable_params": trainable_params,
    }
