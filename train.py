import torch
import torch.nn as nn
import torch.optim as optim
from sophia import SophiaG
import argparse
from torch.utils.data import DataLoader
import tqdm
import random
import numpy as np
import time
import logging
import math
from pathlib import Path
# import deepspeed
import os
from dotenv import load_dotenv
import torch
from torch.utils.tensorboard import SummaryWriter

from jamo import JAMO, Tokenizer
from generate import generate
import utils 
from dataset import IterablDataset

load_dotenv()

os.environ["WANDB_API_KEY"] = os.environ.get('wandb')
os.environ["WANDB_MODE"] = "offline"

logger = logging.getLogger(__name__)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s | %(filename)s : %(lineno)s] >> %(message)s')
fileHandler = logging.FileHandler(filename="./training.log")
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
logger.setLevel(level=logging.INFO)

def set_seed(seed=12346):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else: 
            params_with_wd.append(p)
    return [{"params":params_with_wd, "weight_decay": 0.1}, {"params": params_without_wd, "weight_decay":0.0}]

class Trainer():
    def __init__(self, train_mode:str, batch_size:int, corpus_path: str, checkpoint_dir:str, tokenizer_path:str, save_interval:int, gradient_accumulate:int, is_wandb:bool=False, with_lr_scheduler:bool=True, load:bool=False):
        self.pretrain = train_mode == "pretrain"
        self.learning_rate = 3e-4
        self.batch_size = batch_size
        self.max_iters = 100000
        self.grad_clip = 2.0
        self.warmup_iters = 2000
        self.lr_decay_iters = self.max_iters
        self.min_lr = 1.5e-5 

        self.corpus_path = Path(corpus_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.tokenizer_path = Path(tokenizer_path)
        self.gradient_accumulate = gradient_accumulate
        self.save_interval = save_interval
        self.is_wandb = is_wandb
        self.with_lr_scheduler = with_lr_scheduler

        if self.is_wandb:
            import wandb
            wandb.init(
                project="JAMO",
                config={
                    "architecture": "GPT",
                    "dataset": "Custom Corpus Dataset",
                    "max_iters": self.max_iters,
                }
            )
            logger.info("Initiate the WANDB")

        if load: 
            self.model, self.optimizer, _ = utils.prepare_for_resuming(self.checkpoint_dir, self.learning_rate, model_size="supersmall", best=True, pretrain=self.pretrain)
        else:
            self.checkpoint_dir.mkdir(exist_ok=True)
            self.model = JAMO.from_name("supersmall").to(torch.device("cuda"))
            self.model:nn.Module = torch.compile(self.model, mode="reduce-overhead")
            # optimizer = optim.AdamW(model.parameters(), weight_decay=1e-1, betas=(0.9, 0.95))
            optim_group = self.model.configure_optimizers(weight_decay=2e-1)
            self.optimizer:optim.Optimizer = SophiaG(optim_group, lr=self.learning_rate, betas=(0.965, 0.99), rho = 0.03)

        # model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
        #               model=model,
        #               model_parameters=params)    
        if self.is_wandb:
            import wandb
            wandb.watch(self.model)

        self.tokenizer:Tokenizer = Tokenizer(tokenizer_path)
        self.train_loader:DataLoader = self.create_dataloader(tokenizer=self.tokenizer, block_size=self.model.config.block_size)

        self.writer = SummaryWriter(comment=utils.current())

    def create_dataloader(self, tokenizer, block_size):
        g = torch.Generator()
        g.manual_seed(1231928)

        dataset = IterablDataset(str(self.corpus_path), tokenizer, block_size)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, generator=g)
        logger.info("Finishing Loading the DataLoader")

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
    
    def train(self):
        self.scaler = torch.cuda.amp.GradScaler()
        
        pbar = tqdm.tqdm(range(1, self.max_iters+1))
        for iteration in pbar:
            if self.with_lr_scheduler:
                lr = self.get_lr(iteration)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

            for _ in range(self.gradient_accumulate):
                x, y = next(iter(self.train_loader))
                loss = self.step(x, y)
               
                self.writer.add_scalar("Loss/train", loss.item(), iteration)
                logger.info(f"Iter {iteration}: Training Loss = {loss.item():.4f}")
    
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            if iteration % self.save_interval == 0:
                utils.save_model(iteration, self.model, self.optimizer, self.checkpoint_dir)
            
            if iteration % 1000 == 0:
                self.model.eval()
                result = self.sampling(self.model)
                self.writer.add_text("jamo", result, iteration)
                self.model.train()

            # Log model weight histograms
            for name, param in self.model.named_parameters():
                self.writer.add_histogram(name, param, iteration)

            if self.is_wandb:
                import wandb
                wandb.log({
                    "iter": iter,
                    "train/loss": f"{loss.item():.6f}",
                    "lr": lr
                })
            
        self.writer.close()
        if self.is_wandb: 
            import wandb
            wandb.finish()

    def step(self, x:torch.tensor, y:torch.tensor):
        with torch.cuda.amp.autocast():
            # torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            logits = self.model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1), ignore_index=-1)
            self.scaler.scale(loss / self.gradient_accumulate).backward()

        return loss

    def sampling(self):
        token = self.tokenizer.encode("", bos=True)
        token = torch.tensor(token, dtype=torch.long, device="cuda")
        output = generate(self.model, token, max_new_tokens=60, temperature=0.8, top_k=4, eos_id=self.tokenizer.eos_id)
        result = self.tokenizer.decode(output)

        logger.info(result)
        with open("result.txt", "a") as f:
            f.write(result+"\n")
        
        return result


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    set_seed()
    # torch.multiprocessing.set_start_method("spawn")
    # torch.set_default_device('cuda')

    parser = argparse.ArgumentParser(description='Train My Custom GPT 🚀!!!')

    parser.add_argument("--train_mode", type=str, default="pretrain")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument("--save_interval", type=int, default=10000)
    parser.add_argument("--gradient_accumulate", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./tmp/checkpoint")
    parser.add_argument("--corpus_path", type=str, default="./tmp/512_chunk.txt")
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer/corpus.model")
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--with_lr_scheduler", action="store_true")

    args = parser.parse_args()

    trainer = Trainer(
        train_mode=args.train_mode,
        batch_size=args.batch_size,
        corpus_path=args.corpus_path,
        checkpoint_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path,
        save_interval=args.save_interval,
        gradient_accumulate=args.gradient_accumulate,
        is_wandb=args.wandb,
        with_lr_scheduler=args.with_lr_scheduler,
        load=args.load_model
    )

    trainer.train()