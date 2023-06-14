import json

import torch
from torch.utils.data import Dataset
import numpy as np
import tqdm
import time
from pathlib import Path
import numpy as np
import h5py
import utils
from transformers import AutoTokenizer, GPT2TokenizerFast
from typing import Optional, Union

from jamo import Tokenizer

class IterablDataset(Dataset):
    def __init__(self, corpus: Path, tokenizer: Union[Tokenizer, AutoTokenizer], block_size: int, cache_dir=""):
        self.block_size = block_size
        self.tokenizer = tokenizer

        self.pool_size = 4
        self.from_cache = cache_dir != ""
        
        self.texts = []
        if not self.from_cache:
            num_lines = sum(1 for _ in open(str(corpus), "r", buffering=100000))
            start = time.time()
            with open(corpus, "r", buffering=100000) as f:
                print("Loading Enormous Line by Line")
                for line in tqdm.tqdm(f, total=num_lines):
                    self.texts.append(line.strip())

            print(f"Loading Done in {time.time() - start:.4f}s")
            self.num_subsets = len(self.texts)
        else: 
            h5f = h5py.File(cache_dir, "r")
            self.tokens = h5f["tokens"][:]
            h5f.close()
            self.num_subsets = self.tokens.shape[0]

    @utils.profile
    def save_cache(self, save_dir):
        h5f = h5py.File(str(save_dir), "w")
        
        self.tokens = np.array([self._collate_fn(t) for t in self.texts], dtype=np.int8)
        del self.texts
        h5f.create_dataset("tokens", data=self.tokens)
        h5f.close()

        self.from_cache = True

    def _collate_fn(self, text):
        is_custom = isinstance(self.tokenizer, Tokenizer)
        kwargs = {"bos": True, "eos": True, "max_length": self.block_size + 1, "pad": True} if is_custom else {
            "max_length": 200, "truncation": True}
        token = self.tokenizer.encode(text, **kwargs)
        return token

    def __getitem__(self, idx):
        token = None
        # start, end = idx * (self.block_size+1), (idx+1) * (self.block_size)
        if not self.from_cache:
            text = self.texts[idx]
            token = self._collate_fn(text)
        else:
            token = self.tokens[idx]

        x = torch.tensor(token[:-1], dtype=torch.long, device="cuda")
        y = torch.tensor(token[1:], dtype=torch.long, device="cuda")

        return x, y

    def __len__(self):
        return self.num_subsets
    
    def __repr__(self) -> str:
        return f"Total {self.num_subsets} subsets."


PROMPT_DICT = {
    "prompt_input": (
        "요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        "### 명령어:\n{instruction}\n\n### 입력:\n{input}\n\n### 응답:"
    ),
    "prompt_no_input": (
        "명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        "### 명령어:\n{instruction}\n\n### 응답:"
    ),
}


def _preprocess_hg(strings, tokenizer:GPT2TokenizerFast, block_size):
    tokenized_list = [
        tokenizer(
            text,
            padding="longest",
            truncation=True
        )["input_ids"]
        for text in strings
    ]

    return tokenized_list

def _preprocess_spm(strings, tokenizer: Tokenizer, block_size):
    tokened_list = [tokenizer.encode(text, bos=False, eos=False, max_length=block_size + 1, pad=True) for text in strings]
    return tokened_list


class PromptDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: Union[Tokenizer, GPT2TokenizerFast], block_size):
        super().__init__()
        with open(data_path, "r", "utf-8") as f:
            list_data_dict = json.load(f)

        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        data = [source+target for source, target in zip(sources, targets)]
        _preprocess = _preprocess_spm if isinstance(tokenizer, Tokenizer) else _preprocess_hg
        self.input_ids = _preprocess(data, tokenizer, block_size)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx:int):
        text = self.input_ids[idx]
        x = torch.LongTensor(text[:-1], device="cuda")
        y = torch.LongTensor(text[1:], device="cuda")

        return x, y
