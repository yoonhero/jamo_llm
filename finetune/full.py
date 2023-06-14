from torch.utils.data import DataLoader
from pathlib import Path
import torch
from jamo import Tokenizer
import utils
from jamo.trainer import Trainer
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from dataset import PromptDataset


class LoRATrainer(Trainer):
    def __init__(self, batch_size: int, corpus_path: str, checkpoint_dir: str, tokenizer_path: str,
                 save_interval: int, eval_interval: int, gradient_accumulate: int, r:int=2, alpha:int=1, dropout:0.1=float):
        Trainer.__init__(self, batch_size, corpus_path, checkpoint_dir, tokenizer_path, save_interval, eval_interval, gradient_accumulate)

        self.model, self.optimizer, _ = utils.prepare_for_resuming("../tmp/", self.learning_rate,
                                                                       model_size="supersmall", best=True,
                                                                       pretrain=False)

        self.tokenizer: Tokenizer = Tokenizer(self.tokenizer_path)
        self.train_loader: DataLoader = self.create_dataloader(tokenizer=self.tokenizer, block_size=self.model.config.block_size)

        self.warmup_iters = 100

    def get_lr(self, iteration: int):
        lr = self.learning_rate * iteration / self.warmup_iters
        return lr

    def create_dataloader(self, tokenizer, block_size: int):
        g = torch.Generator()
        g.manual_seed(1231928)
        dataset = PromptDataset(str(self.corpus_path), tokenizer, block_size)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, generator=g)
        self.logger.info("Finishing Loading the DataLoader")

        return train_loader


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    trainer = Trainer(
        batch_size=128,
        corpus_path="../tmp/ko_alpaca_data.json",
        checkpoint_dir="../tmp/checkpoint",
        tokenizer_path="../tokenizer/corpus.vocab",
        save_interval=100,
        eval_interval=100,
        gradient_accumulate=4,
    )

    trainer.train()
