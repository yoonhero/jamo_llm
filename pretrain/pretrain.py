import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
import random
import numpy as np
import math
from pathlib import Path
import torch
import sys
from transformers import AutoTokenizer

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from sophia import SophiaG
from jamo.trainer import Trainer
from jamo import JAMO, Tokenizer
from generate import generate
import utils
from dataset import IterablDataset


class PreTrainer(Trainer):
    def __init__(self, train_mode: str, batch_size: int, corpus_path: str, checkpoint_dir: str, tokenizer_path: str,
                 save_interval: int, eval_interval: int, gradient_accumulate: int,
                 load: bool = False):
        Trainer.__init__(self, batch_size, corpus_path, checkpoint_dir, tokenizer_path, save_interval, eval_interval, gradient_accumulate)
        self.pretrain = train_mode == "pretrain"
        self.max_iters = 300000
        self.warmup_iters = 4000
        self.lr_decay_iters = self.max_iters
        self.min_lr = 2e-5

        if load:
            self.model, self.optimizer, _ = utils.prepare_for_resuming(self.checkpoint_dir, "small", self.learning_rate,
                                                                     best=True, pretrain=self.pretrain)
        else:
            self.checkpoint_dir.mkdir(exist_ok=True)
            self.model: nn.Module = JAMO.from_name("small").to(torch.device("cuda"))
            self.model: nn.Module = torch.compile(self.model, mode="reduce-overhead")
            optim_group = self.model.configure_optimizers(weight_decay=1e-1)
            self.optimizer: optim.Optimizer = SophiaG(optim_group, lr=self.learning_rate, betas=(0.965, 0.99), rho=0.02)

        # self.tokenizer: Tokenizer = Tokenizer(self.tokenizer_path)
        self.tokenizer = AutoTokenizer.from_pretrained("hg_tokenizer")
        self.train_loader: DataLoader = self.create_dataloader(tokenizer=self.tokenizer,
                                                               block_size=self.model.config.block_size)

        Trainer.init_logger(self)

    def create_dataloader(self, tokenizer, block_size):
        g = torch.Generator()
        g.manual_seed(1231928)
        dataset = IterablDataset(self.corpus_path, tokenizer, block_size)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, generator=g)
        self.logger.info("Finishing Loading the DataLoader")

        return train_loader

    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return self.learning_rate * it / self.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    utils.set_seed()

    parser = argparse.ArgumentParser(description='Pretrain My Custom GPT 🚀!!!')

    parser.add_argument("--train_mode", type=str, default="pretrain")
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument("--save_interval", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=5000)
    parser.add_argument("--gradient_accumulate", type=int, default=6)
    parser.add_argument("--output_dir", type=str, default="../tmp/checkpoint")
    parser.add_argument("--corpus_path", type=str, default="../tmp/cleaned/512.txt")
    parser.add_argument("--tokenizer_path", type=str, default="hg_tokenizer")
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument("--with_lr_scheduler", action="store_true")

    args = parser.parse_args()

    trainer = PreTrainer(
        train_mode=args.train_mode,
        batch_size=args.batch_size,
        corpus_path=args.corpus_path,
        checkpoint_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        gradient_accumulate=args.gradient_accumulate,
        load=args.load_model
    )

    trainer.train()
