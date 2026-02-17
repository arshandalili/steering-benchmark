"""Training helpers."""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import Dataset


def build_prompt_target_pairs(dataset, group: str, limit: int, split: str | None) -> List[Tuple[str, str]]:
    pairs = []
    for ex in dataset.iter_group(group, limit=limit, split=split):
        target = ex.target
        if isinstance(target, (list, tuple, set)):
            target = list(target)[0] if target else ""
        pairs.append((ex.prompt, str(target)))
    return pairs


class PromptTargetDataset(Dataset):
    def __init__(self, tokenizer, pairs: List[Tuple[str, str]], max_length: int = 512):
        self.tokenizer = tokenizer
        self.pairs = pairs
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx):
        prompt, target = self.pairs[idx]
        full_text = prompt + " " + target
        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        labels = input_ids.clone()
        prompt_ids = self.tokenizer(prompt, truncation=True, max_length=self.max_length, return_tensors="pt")[
            "input_ids"
        ].squeeze(0)
        labels[: prompt_ids.shape[0]] = -100
        return {"input_ids": input_ids, "labels": labels}


def collate_batch(batch, pad_token_id: int = 0):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = input_ids.ne(pad_token_id).long()
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
