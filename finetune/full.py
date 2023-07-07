from torch.utils.data import DataLoader
from pathlib import Path
import torch
import sys
import argparse
import torch.nn as nn
import torch.optim as optim

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from dataset import PromptDataset
from sophia import SophiaG
from jamo import Tokenizer
import utils
from jamo.trainer import Trainer


class FullTrainer(Trainer):
    def __init__(self, model_path: str, model_size:str, learning_rate: float, batch_size: int, cache_path: str, checkpoint_dir: str, tokenizer_path: str,
                 max_iters: int, warmup_iters: int, save_interval: int, eval_interval: int, gradient_accumulate: int, with_lr_scheduler: bool):
        Trainer.__init__(self, learning_rate, batch_size, "", checkpoint_dir, tokenizer_path, save_interval, eval_interval, gradient_accumulate)

        model_path = Path(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = utils.load_model(model_path, model_size=model_size, device=device)
        self.cache_path = cache_path

        self.model: nn.Module = torch.compile(self.model, mode="default")
        optim_group = self.model.configure_optimizers(weight_decay=2e-1)
        self.optimizer: optim.Optimizer = SophiaG(optim_group, lr=self.learning_rate, betas=(0.965, 0.99), rho=0.01)

        self.tokenizer = None
        self.train_loader, self.eval_loader = self.create_dataloader()

        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.with_lr_scheduler = with_lr_scheduler

    def get_lr(self, iteration: int):
        lr = self.learning_rate * iteration / self.warmup_iters if self.warmup_iters > iteration else self.learning_rate
        return lr

    def create_dataloader(self):
        g = torch.Generator()
        g.manual_seed(1231928)
        train_dataset = PromptDataset(cache_dir=self.cache_dir)
        eval_dataset = PromptDataset(cache_dir=self.cache_dir, mode="eval")
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, generator=g)
        if len(eval_dataset) == 0:
            eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        else: eval_loader = None
        self.logger.info("Finishing Loading the DataLoader")

        return train_loader, eval_loader
    
    @torch.no_grad()
    def eval(self, iteration):
        losses = []
        for _, (x, y) in enumerate(self.eval_loader):
            logits = self.model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1), ignore_index=-1)
            losses.append(loss.item())

        min_loss = sum(losses) / len(losses)
        self.writer.add_scalar("Loss/eval", min_loss, iteration)
        self.logger.info(f"Iter {iteration}: Eval Loss = {min_loss}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(description='Pretraining your own custom LLM 🚀!!!')

    parser.add_argument("--model_path", type=str, default="tmp/checkpoint")
    parser.add_argument("--model_size", type=str, default="small")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=60)
    parser.add_argument("--max_iters", type=int, default=6000)
    parser.add_argument("--warmup_iters", type=int, default=300)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--gradient_accumulate", type=int, default=6)
    parser.add_argument("--checkpoint_dir", type=str, default="../tmp/finetuned")
    parser.add_argument("--cache_path", type=str, default="../tmp/sft-cache.hdf5")
    parser.add_argument("--tokenizer_path", type=str, default="hg_tokenizer")
    parser.add_argument("--with_lr_scheduler", action="store_true")

    args = parser.parse_args()

    trainer = FullTrainer(
        model_path=args.model_path,
        model_size=args.model_size,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        warmup_iters=args.warmup_iters,
        cache_path=args.cache_path,
        checkpoint_dir=args.checkpoint_dir,
        tokenizer_path=args.tokenizer_path,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        gradient_accumulate=args.gradient_accumulate,
        with_lr_scheduler=args.with_lr_scheduler
    )

    trainer.train()
