import json
import os

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
import codecs
from multiprocessing import Pool
import copy

from jamo import Tokenizer

class IterablDataset(Dataset):
    def __init__(self, corpus: Path, tokenizer: Union[Tokenizer, AutoTokenizer], block_size: int, cache_dir=""):
        self.block_size = block_size
        self.tokenizer = tokenizer

        self.pool_size = 4
        self.from_cache = cache_dir != ""
        self.texts = []
        self.tokenizer_is_custom = isinstance(self.tokenizer, Tokenizer)

        if not self.from_cache:
            # num_lines = sum(1 for _ in codecs.open(str(corpus), "r", encoding="utf-8", buffering=10000, errors="ignore"))
            start = time.time()

            # self.texts = self.load_corpus(corpus)
            with codecs.open(corpus, "r", encoding="utf-8", buffering=100000, errors="ignore") as f:
                print("Loading Enormous Line by Line")

                for line in tqdm.tqdm(f, total=7653985):
                    if len(line) < 200:
                        return
                    self.texts.append(line.strip())

            print(f"Loading Done in {time.time() - start:.4f}s")
            self.num_subsets = len(self.texts)
        else: 
            h5f = h5py.File(cache_dir, "r")
            self.tokens = h5f["tokens"][:]
            h5f.close()
            self.num_subsets = self.tokens.shape[0]


    def process_chunk(self, chunk):
        # Process each chunk of lines using multiprocessing
        processed_chunk = []
        for line in chunk:
            if len(line) < 400:
                continue
            processed_chunk.append(line)
        return processed_chunk

    def load_corpus(self, file_path, chunk_size=10000):
        pool = Pool(os.cpu_count() - 1)
        file_size = os.path.getsize(file_path)
        num_chunks = file_size // chunk_size

        with codecs.open(file_path, "r", encoding="utf-8", buffering=100000, errors="ignore") as file:
            chunks = []

            print("Read the chunk by chunk")
            lines = file.readlines(chunk_size)
            while lines:
                chunks.append(lines)
                lines = file.readlines(chunk_size)

            results = []
            with tqdm.tqdm(total=num_chunks) as pbar:
                for chunk in chunks:
                    result = pool.apply_async(self.process_chunk, args=(chunk,))
                    results.append(result)
                    pbar.update(1)

        pool.close()
        pool.join()

        # Get the processed chunks from the results
        processed_chunks = [result.get() for result in results]

        # Flatten the processed chunks into a single list
        processed_corpus = [line for chunk in processed_chunks for line in chunk]
        return processed_corpus

    @utils.profile
    def save_cache(self, save_dir):
        h5f = h5py.File(str(save_dir), "w")
        
        self.tokens = np.array([self._collate_fn(t) for t in self.texts], dtype=np.int8)
        del self.texts
        h5f.create_dataset("tokens", data=self.tokens)
        h5f.close()

        self.from_cache = True

    def _collate_fn(self, text):

        kwargs = {"bos": True, "eos": True, "max_length": self.block_size + 1, "pad": True} if self.tokenizer_is_custom else {
            "max_length": self.block_size+1, "truncation": True, "padding": "max_length", "return_tensors": "pt"}

        text = text if self.tokenizer_is_custom else f"<s> {text} </s>"
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

        if self.tokenizer_is_custom:
            x = torch.tensor(token[:-1], dtype=torch.long, device="cuda")
            y = torch.tensor(token[1:], dtype=torch.long, device="cuda")
        else:
            token = token[0].to("cuda")
            x = token[:-1].clone()
            y = token[1:].clone()

        return x, y

    def __len__(self):
        return self.num_subsets
    
    def __repr__(self) -> str:
        return f"Total {self.num_subsets} subsets."


PROMPT_DICT = {
    "prompt_input": (
        "요청을 적절히 완료하는 응답을 작성하세요.\n"
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
    def __init__(self, data_path: Optional[str]="", tokenizer: Union[Tokenizer, GPT2TokenizerFast]=None, block_size: Optional[int]=None, cache_dir:str="", mode: str="train"):
        super().__init__()

        if cache_dir == "":
            with open(data_path, "r", "utf-8") as f:
                list_data_dict = json.load(f)

            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
            sources = [
                prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
                for example in list_data_dict
            ]
            targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

            data = [source + target for source, target in zip(sources, targets)]
            _preprocess = _preprocess_spm if isinstance(tokenizer, Tokenizer) else _preprocess_hg
            self.input_ids = _preprocess(data, tokenizer, block_size)
        else:
            h5f = h5py.File(cache_dir, "r")
            self.input_ids = h5f[f"/{mode}"][:].tolist()
            h5f.close()

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx:int):
        text = self.input_ids[idx]
        x = torch.tensor(text[:-1], dtype=torch.long, device="cuda")
        y = torch.tensor(text[1:], dtype=torch.long, device="cuda")

        return x, y
