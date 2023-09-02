import tqdm
import logging
from pathlib import Path
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from typing import Optional, Union
from transformers import GPT2TokenizerFast
import gc
try:
    import wandb
except: pass

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from jamo import Tokenizer
import utils

class Trainer():
    model: Optional[torch.nn.Module] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    train_loader: Optional[DataLoader] = None
    tokenizer: Optional[Union[GPT2TokenizerFast, Tokenizer]] = None

    def __init__(self, learning_rate: float, batch_size: int, corpus_path: str, checkpoint_dir: str, tokenizer_path: str,
                 save_interval: int, eval_interval: int, gradient_accumulate: int
                 ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        # self.max_iters = 2000

        self.corpus_path: Path = Path(corpus_path)
        self.checkpoint_dir: Path = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.tokenizer_path: Path = Path(tokenizer_path)
        self.gradient_accumulate = gradient_accumulate
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.with_lr_scheduler = False

    def create_dataloader(self, tokenizer, block_size):
        return NotImplementedError

    def get_lr(self, iteration: int):
        return NotImplementedError

    def init_logger(self) -> None:
        self.writer = SummaryWriter(comment=utils.current())
        self.logger = logging.getLogger(__name__)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s | %(filename)s : %(lineno)s] >> %(message)s')
        fileHandler = logging.FileHandler(filename="../training.log")
        fileHandler.setFormatter(formatter)
        self.logger.addHandler(fileHandler)
        self.logger.setLevel(level=logging.INFO)

        try: 
            wandb.init(
                # set the wandb project where this run will be logged
                project="",
                
                # track hyperparameters and run metadata
                config={
                    "learning_rate": self.learning_rate,
                    "architecture": "GPT",
                    "dataset": "Custom Korean Corpus",
                    "epochs": self.max_iters,
                }
            )
            wandb.watch(self.model, log_freq=100) 
        except ImportError: pass

    def train(self):
        self.scaler = torch.cuda.amp.GradScaler()

        pbar = tqdm.tqdm(range(1, self.max_iters+1))
        for iteration in pbar:
            lr = self.get_lr(iteration) if self.with_lr_scheduler else self.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            for _ in range(self.gradient_accumulate):
                x, y, mask = next(iter(self.train_loader))

                def minibatch(x, y, mask):
                    with torch.cuda.amp.autocast():
                        logits = self.model(x)
                        logits[mask==0] = float("-inf")
                        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1),
                                                                 ignore_index=-1)
                        self.scaler.scale(loss / self.gradient_accumulate).backward()
                        return loss

                loss = minibatch(x, y, mask)
                wandb.log({"loss": loss.item()})

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            self.writer.add_scalar("Loss/train", loss.item(), iteration)
            self.writer.add_scalar("LR/train", lr, iteration)
            self.logger.info(f"Iter {iteration}: Training Loss = {loss.item():.4f}")

            if iteration % self.save_interval == 0:
                utils.save_model(iteration, self.model, self.optimizer, self.checkpoint_dir)

            if iteration % self.eval_interval == 0:
                self.eval(iteration)

        self.writer.close()
        wandb.finish()

    @torch.no_grad()
    def eval(self, iteration):
        losses = []
        for (x, y, mask) in self.eval_loader:
            logits = self.model(x)
            # Exclude the masked position during trainig process for good result.
            logits[mask==0] = float("-inf")
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1), ignore_index=-1)
            losses.append(loss.item())

        loss_mu = sum(losses) / len(losses)
        self.writer.add_scalar("Loss/eval", loss_mu, iteration)
        self.logger.info(f"Iter {iteration}: Eval Loss = {loss_mu}")

        # self.writer.add_text("jamo", result, iteration)

        for name, param in self.model.named_parameters():
            self.writer.add_histogram(name, param, iteration)

    def sampling(self):
        # is_custom = isinstance(self.tokenizer, Tokenizer)
        # kwargs = {"bos": True} if is_custom else {}
        # token = self.tokenizer.encode("" if is_custom else "<s>", **kwargs)
        # token = torch.tensor(token, dtype=torch.long, device="cuda")
        token = self.tokenizer.encode("<s>", return_tensors="pt").to("cuda")
        # eos_id = self.tokenizer.eos_id if is_custom else self.tokenizer.encode("</s>")
        eos_id = self.tokenizer.encode("</s>")[0]
        output = generate(self.model, token, max_new_tokens=100, temperature=0.8, top_k=20, eos_id=eos_id)
        self.model.reset_cache()
        result = self.tokenizer.decode(output)

        self.logger.info(result)
        with open("../result.txt", "a") as f:
            f.write(result + "\n")

        return result
