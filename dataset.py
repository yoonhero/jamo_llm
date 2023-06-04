import torch
from torch.utils.data import Dataset
import numpy as np
import tqdm
import time
from pathlib import Path
import numpy as np
import h5py
import utils

from jamo import Tokenizer

class IterablDataset(Dataset):
    def __init__(self, corpus: Path, tokenizer: Tokenizer, block_size: int, cache_dir=""):
        self.block_size = block_size
        self.tokenizer = tokenizer

        self.pool_size = 4
        self.from_cache = cache_dir == ""
        
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
        token = self.tokenizer.encode(text, bos=False, eos=False, max_length=self.block_size+1, pad=True)
        return token

    def __getitem__(self, idx):
        token = None
        # start, end = idx * (self.block_size+1), (idx+1) * (self.block_size)
        if not self.from_cache:
            text = self.texts[idx]
            token = self._collate_fn(text)
        else:
            token = self.tokens[idx]
        
        x = torch.tensor(token[:-1], dtype=torch.long)
        y = torch.tensor(token[1:], dtype=torch.long)

        return x, y

    def __len__(self):
        return self.num_subsets 
    
    def __repr__(self) -> str:
        return f"Total {self.num_subsets} subsets."