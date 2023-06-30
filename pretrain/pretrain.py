import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
import math
from pathlib import Path
import torch
import sys
from transformers import AutoTokenizer
import time

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from sophia import SophiaG
from jamo.trainer import Trainer
from jamo import JAMO, Tokenizer
import utils
from dataset import IterablDataset


class PreTrainer(Trainer):
    def __init__(self, model_size:str, learning_rate: float, min_lr: float, batch_size: int, corpus_path: str, checkpoint_dir: str, tokenizer_path: str,
                 max_iters: int, warmup_iters: int, save_interval: int, eval_interval: int, gradient_accumulate: int,
                 load: bool = False, with_lr_scheduler: bool=False):
        Trainer.__init__(self, learning_rate, batch_size, corpus_path, checkpoint_dir, tokenizer_path, save_interval, eval_interval, gradient_accumulate)
        self.pretrain = True
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = self.max_iters
        self.min_lr = min_lr
        self.with_lr_scheduler = with_lr_scheduler

        if load:
            self.model, self.optimizer, _ = utils.prepare_for_resuming(self.checkpoint_dir, model_size, self.learning_rate,
                                                                     best=True, pretrain=self.pretrain)
        else:
            self.checkpoint_dir.mkdir(exist_ok=True)
            self.model: nn.Module = JAMO.from_name(model_size).to(torch.device("cuda"))
            self.model: nn.Module = torch.compile(self.model, mode="reduce-overhead")
            optim_group = self.model.configure_optimizers(weight_decay=1e-1)
            self.optimizer: optim.Optimizer = SophiaG(optim_group, lr=self.learning_rate, betas=(0.965, 0.99), rho=0.01)

        if self.model.config.vocab_size == 8000 or self.model.config.vocab_size == 32000:
            utils.tokenizer_setting()
            self.tokenizer = AutoTokenizer.from_pretrained("hg_tokenizer")
        else:
            self.tokenizer: Tokenizer = Tokenizer(self.tokenizer_path)

        self.train_loader: DataLoader = self.create_dataloader(tokenizer=self.tokenizer,
                                                               block_size=self.model.config.block_size)

        Trainer.init_logger(self)

    def create_dataloader(self, tokenizer, block_size, seed:int=1231928):
        g = torch.Generator()
        g.manual_seed(seed)

        t0 = time.time()
        dataset = IterablDataset(self.corpus_path, tokenizer, block_size)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, generator=g)
        self.logger.info(f"Finishing Loading the DataLoader!!! in {time.time() - t0}")

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

    parser = argparse.ArgumentParser(description='Pretraining your own custom LLM 🚀!!!')

    parser.add_argument("--model_size", type=str, default="tiny")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--max_iters", type=int, default=50000)
    parser.add_argument("--warmup_iters", type=int, default=750)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--gradient_accumulate", type=int, default=6)
    parser.add_argument("--checkpoint_dir", type=str, default="../tmp/checkpoint")
    parser.add_argument("--corpus_path", type=str, default="../tmp/dataset.txt")
    parser.add_argument("--tokenizer_path", type=str, default="hg_tokenizer")
    parser.add_argument("--load_model", action='store_true')
    parser.add_argument("--with_lr_scheduler", action="store_true")

    args = parser.parse_args()

    trainer = PreTrainer(
        model_size=args.model_size,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        corpus_path=args.corpus_path,
        checkpoint_dir=args.checkpoint_dir,
        tokenizer_path=args.tokenizer_path,
        max_iters=args.max_iters,
        warmup_iters=args.warmup_iters,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        gradient_accumulate=args.gradient_accumulate,
        load=args.load_model,
        with_lr_scheduler=args.with_lr_scheduler
    )

    trainer.train()
