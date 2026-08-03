import codecs
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer, GPT2TokenizerFast

import utils
from jamo import Tokenizer


class IterablDataset(Dataset):
    """Pretraining dataset with device-aware, backward-compatible cache reads."""

    def __init__(self, corpus: Path, tokenizer: Union[Tokenizer, AutoTokenizer], block_size: int,
                 cache_dir="", device="auto"):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.device = utils.resolve_device(device)
        self.from_cache = cache_dir != ""
        self.texts = []
        self.tokenizer_is_custom = isinstance(self.tokenizer, Tokenizer)
        self.pad_token_id = (
            getattr(tokenizer, "pad_token_id", None)
            if tokenizer is not None else None
        )
        if self.pad_token_id is None and tokenizer is not None:
            self.pad_token_id = getattr(tokenizer, "pad_id", None)
        if self.pad_token_id is None:
            self.pad_token_id = 1

        if not self.from_cache:
            start = time.time()
            corpus = Path(corpus)
            total_files = [corpus] if corpus.is_file() else sorted(corpus.glob("*"))
            print(f"Total Chunk: {len(total_files)}")

            if total_files:
                worker_count = max((os.cpu_count() or 1) - 1, 1)
                with Pool(worker_count) as pool:
                    for chunk in tqdm.tqdm(pool.imap_unordered(self.process_chunk, total_files),
                                           total=len(total_files)):
                        self.texts.extend(chunk)

            print(f"Loading Done in {time.time() - start:.4f}s")
            self.num_subsets = len(self.texts)
        else:
            with h5py.File(cache_dir, "r") as h5f:
                self.tokens = h5f["tokens"][:]
                self.attention_masks = (
                    h5f["attention_mask"][:] if "attention_mask" in h5f else None
                )
            if self.tokens.dtype == np.int8:
                raise ValueError(
                    "The cached token dataset uses int8, which overflows the vocabulary. "
                    "Rebuild it with save_cache()."
                )
            self.num_subsets = self.tokens.shape[0]

    def process_chunk(self, path):
        with codecs.open(path, "r", encoding="utf-8", errors="ignore") as file:
            return [line.strip() for line in file if line.strip()]

    @utils.profile
    def save_cache(self, save_dir):
        token_rows = []
        mask_rows = []
        for text in self.texts:
            token, mask = self._collate_fn(f"<s> {text} </s>" if not self.tokenizer_is_custom else text)
            token_rows.append(token)
            mask_rows.append(mask)

        self.tokens = np.asarray(token_rows, dtype=np.int16)
        self.attention_masks = np.asarray(mask_rows, dtype=np.uint8)
        del self.texts

        save_dir = Path(save_dir)
        save_dir.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(str(save_dir), "w") as h5f:
            h5f.create_dataset("tokens", data=self.tokens, dtype=np.int16)
            h5f.create_dataset("attention_mask", data=self.attention_masks, dtype=np.uint8)
        self.from_cache = True

    def _collate_fn(self, text):
        if self.tokenizer_is_custom:
            token = self.tokenizer.encode(
                text, bos=True, eos=True, max_length=self.block_size + 1, pad=True
            )
            mask = [int(token_id != self.pad_token_id) for token_id in token]
            return token, mask

        token_data = self.tokenizer(
            text,
            max_length=self.block_size + 1,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True,
        )
        token = token_data["input_ids"]
        mask = token_data["attention_mask"]
        if token.ndim > 1:
            token = token[0]
        if mask.ndim > 1:
            mask = mask[0]
        return token.tolist(), mask.tolist()

    def __getitem__(self, idx):
        if not self.from_cache:
            text = self.texts[idx]
            token, mask = self._collate_fn(
                f"<s> {text} </s>" if not self.tokenizer_is_custom else text
            )
        else:
            token = self.tokens[idx]
            if self.attention_masks is None:
                mask = (token != self.pad_token_id).astype(np.uint8)
            else:
                mask = self.attention_masks[idx]

        token = torch.as_tensor(token, dtype=torch.long, device=self.device)
        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        if token.ndim > 1:
            token = token[0]
        if mask.ndim > 1:
            mask = mask[0]

        x = token[:-1].clone()
        y = token[1:].clone()
        label_mask = mask[1:]
        y = y.masked_fill(~label_mask, -1)
        return x, y, label_mask

    def __len__(self):
        return self.num_subsets

    def __repr__(self) -> str:
        return f"Total {self.num_subsets} subsets."


PROMPT_DICT = {
    "prompt_input": (
        "명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        "### 명령어:\n{instruction}\n\n### 입력:\n{input}\n\n### 응답:\n"
    ),
    "prompt_no_input": (
        "명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        "### 명령어:\n{instruction}\n\n### 응답:\n"
    ),
}


def _preprocess_hg(strings, tokenizer: GPT2TokenizerFast, block_size):
    return [
        tokenizer(text, padding="longest", truncation=True)["input_ids"]
        for text in strings
    ]


def _preprocess_spm(strings, tokenizer: Tokenizer, block_size):
    return [
        tokenizer.encode(text, bos=False, eos=False, max_length=block_size + 1, pad=True)
        for text in strings
    ]


def _preprocess_with_completion_masks(sources, targets, tokenizer, block_size, is_custom):
    input_ids = []
    loss_masks = []
    max_length = block_size + 1

    for source, target in zip(sources, targets):
        if is_custom:
            prompt_ids = tokenizer.encode(source, bos=True, eos=False, max_length=-1, pad=False)
            token_ids = tokenizer.encode(
                source + target, bos=True, eos=True, max_length=max_length, pad=True
            )
            pad_id = tokenizer.pad_id
        else:
            prompt_ids = tokenizer.encode(f"<s> {source}", add_special_tokens=False)
            token_ids = tokenizer.encode(
                f"<s> {source}{target} </s>",
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
                padding="max_length",
            )
            pad_id = tokenizer.pad_token_id

        prompt_length = min(len(prompt_ids), len(token_ids))
        mask = np.zeros(len(token_ids), dtype=np.uint8)
        for position in range(prompt_length, len(token_ids)):
            if token_ids[position] != pad_id:
                mask[position] = 1
        input_ids.append(token_ids)
        loss_masks.append(mask)

    return input_ids, loss_masks


class PromptDataset(Dataset):
    def __init__(self, data_path: Optional[str] = "", tokenizer: Union[Tokenizer, GPT2TokenizerFast] = None,
                 block_size: Optional[int] = None, cache_dir: str = "", mode: str = "train",
                 device="auto", pad_token_id: Optional[int] = None):
        super().__init__()
        self.device = utils.resolve_device(device)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(tokenizer, "pad_id", None)
            if pad_token_id is None and cache_dir:
                pad_token_id = 1
        self.pad_token_id = pad_token_id
        self.loss_masks = None

        if cache_dir == "":
            with open(data_path, "r", encoding="utf-8") as f:
                list_data_dict = json.load(f)

            prompt_input = PROMPT_DICT["prompt_input"]
            prompt_no_input = PROMPT_DICT["prompt_no_input"]
            sources = [
                prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
                for example in list_data_dict
            ]
            targets = [str(example["output"]) for example in list_data_dict]
            self.input_ids, self.loss_masks = _preprocess_with_completion_masks(
                sources, targets, tokenizer, block_size, isinstance(tokenizer, Tokenizer)
            )
        else:
            with h5py.File(cache_dir, "r") as h5f:
                self.input_ids = h5f[f"/{mode}"][:].tolist()
                mask_key = f"{mode}_loss_mask"
                if mask_key in h5f:
                    self.loss_masks = h5f[mask_key][:]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        text = self.input_ids[idx]
        x = torch.tensor(text[:-1], dtype=torch.long, device=self.device)
        y = torch.tensor(text[1:], dtype=torch.long, device=self.device)

        if self.loss_masks is not None:
            mask = torch.tensor(self.loss_masks[idx][1:], dtype=torch.bool, device=self.device)
            if mask.shape != y.shape:
                raise ValueError("Cached loss mask and token sequence have different lengths")
            y = y.masked_fill(~mask, -1)
        elif self.pad_token_id is not None:
            y = y.masked_fill(y == self.pad_token_id, -1)

        return x, y
