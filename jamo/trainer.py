import logging
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Union

import torch
import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2TokenizerFast

try:
    import wandb
except ImportError:
    wandb = None

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from jamo import Tokenizer
import utils


class Trainer:
    model: Optional[torch.nn.Module] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    train_loader: Optional[DataLoader] = None
    tokenizer: Optional[Union[GPT2TokenizerFast, Tokenizer]] = None

    def __init__(self, learning_rate: float, batch_size: int, corpus_path: str, checkpoint_dir: str,
                 tokenizer_path: str, save_interval: int, eval_interval: int,
                 gradient_accumulate: int, device="auto", amp: bool = True):
        if batch_size <= 0 or gradient_accumulate <= 0:
            raise ValueError("batch_size and gradient_accumulate must be positive")
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = utils.resolve_device(device)
        self.amp = amp

        self.corpus_path = Path(corpus_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer_path = Path(tokenizer_path)
        self.gradient_accumulate = gradient_accumulate
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.with_lr_scheduler = False
        self.log_histograms = True
        self.wandb_enabled = False

    def create_dataloader(self, tokenizer, block_size):
        raise NotImplementedError

    def get_lr(self, iteration: int):
        raise NotImplementedError

    def init_logger(self) -> None:
        self.writer = SummaryWriter(comment=utils.current())
        self.logger = logging.getLogger(f"jamo-trainer-{id(self)}")
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s | %(filename)s : %(lineno)s] >> %(message)s"
        )
        file_handler = logging.FileHandler(filename="../training.log")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(level=logging.INFO)

        # W&B is opt-in now: the historical empty project name could abort a
        # local run when wandb happened to be installed.
        if wandb is not None and os.environ.get("WANDB_PROJECT"):
            try:
                wandb.init(
                    project=os.environ["WANDB_PROJECT"],
                    config={
                        "learning_rate": self.learning_rate,
                        "architecture": "GPT",
                        "dataset": "Custom Korean Corpus",
                        "epochs": getattr(self, "max_iters", None),
                    },
                )
                wandb.watch(self.model, log_freq=100)
                self.wandb_enabled = True
            except Exception as exc:
                self.logger.warning("W&B disabled: %s", exc)

    def _autocast_context(self):
        if not self.amp:
            return nullcontext()
        if self.device.type == "cuda":
            return torch.cuda.amp.autocast()
        if self.device.type == "mps":
            return torch.autocast(device_type="mps", dtype=torch.float16)
        return nullcontext()

    def _create_scaler(self):
        enabled = self.amp and self.device.type == "cuda"
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler(enabled=enabled)

    @staticmethod
    def _loss(logits, targets):
        if not torch.any(targets != -1):
            return logits.sum() * 0.0
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-1,
        )

    @staticmethod
    def _unpack_batch(batch):
        """Accept both historical (x, y, mask) and current (x, y) batches."""
        if len(batch) == 2:
            return batch

        if len(batch) != 3:
            raise ValueError(f"Expected a 2- or 3-tensor batch, got {len(batch)} values")
        x, y, mask = batch
        mask = mask.to(dtype=torch.bool)
        while mask.ndim > y.ndim and mask.shape[-2] == 1:
            mask = mask.squeeze(-2)
        if mask.ndim == y.ndim and mask.shape[-1] == y.shape[-1] + 1:
            mask = mask[..., 1:]
        if mask.shape != y.shape:
            raise ValueError(
                f"Batch mask shape {tuple(mask.shape)} does not match labels {tuple(y.shape)}"
            )
        return x, y.masked_fill(~mask, -1)

    def train(self):
        if len(self.train_loader) == 0:
            raise ValueError("The training DataLoader is empty; reduce batch_size or add data")
        self.model.train()
        self.scaler = self._create_scaler()
        if hasattr(self.optimizer, "set_batch_size"):
            self.optimizer.set_batch_size(self.batch_size * self.gradient_accumulate)
        self.optimizer.zero_grad(set_to_none=True)
        train_iter = iter(self.train_loader)

        pbar = tqdm.tqdm(range(1, self.max_iters + 1))
        for iteration in pbar:
            lr = self.get_lr(iteration) if self.with_lr_scheduler else self.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            microbatch_losses = []
            for _ in range(self.gradient_accumulate):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)
                x, y = self._unpack_batch(batch)

                with self._autocast_context():
                    logits = self.model(x)
                    loss = self._loss(logits, y)
                self.scaler.scale(loss / self.gradient_accumulate).backward()
                microbatch_losses.append(loss.detach())

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            mean_loss = torch.stack(microbatch_losses).mean().item()
            if self.wandb_enabled:
                wandb.log({"loss": mean_loss})
            self.writer.add_scalar("Loss/train", mean_loss, iteration)
            self.writer.add_scalar("LR/train", lr, iteration)
            self.logger.info(f"Iter {iteration}: Training Loss = {mean_loss:.4f}")

            if iteration % self.save_interval == 0:
                utils.save_model(iteration, self.model, self.optimizer, self.checkpoint_dir)
            if iteration % self.eval_interval == 0:
                self.eval(iteration)

        self.writer.close()
        if self.wandb_enabled:
            wandb.finish()

    @torch.no_grad()
    def eval(self, iteration):
        if self.eval_loader is None:
            self.logger.info(f"Iter {iteration}: Eval skipped (no evaluation data)")
            return None

        was_training = self.model.training
        self.model.eval()
        losses = []
        for batch in self.eval_loader:
            x, y = self._unpack_batch(batch)
            with self._autocast_context():
                logits = self.model(x)
                losses.append(self._loss(logits, y).item())

        if not losses:
            self.logger.info(f"Iter {iteration}: Eval skipped (empty evaluation data)")
            if was_training:
                self.model.train()
            return None

        loss_mu = sum(losses) / len(losses)
        self.writer.add_scalar("Loss/eval", loss_mu, iteration)
        self.logger.info(f"Iter {iteration}: Eval Loss = {loss_mu}")

        if self.log_histograms:
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(name, param, iteration)
        if was_training:
            self.model.train()
        return loss_mu

    def sampling(self):
        is_custom = isinstance(self.tokenizer, Tokenizer)
        token_ids = self.tokenizer.encode("" if is_custom else "<s>", bos=True) if is_custom else self.tokenizer.encode("<s>")
        token = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        eos_id = self.tokenizer.eos_id if is_custom else self.tokenizer.eos_token_id
        self.model.reset_cache()
        output = generate(self.model, token, max_new_tokens=100, temperature=0.8, top_k=20, eos_id=eos_id)
        self.model.reset_cache()
        result = self.tokenizer.decode(output)

        self.logger.info(result)
        with open("../result.txt", "a") as f:
            f.write(result + "\n")
        return result
