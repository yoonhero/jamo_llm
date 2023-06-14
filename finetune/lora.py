import argparse
from torch.utils.data import DataLoader
from pathlib import Path
import torch
from jamo import Tokenizer
from generate import generate
import utils
from jamo.trainer import Trainer
import sys

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from jamo import lora



class LoRATrainer(Trainer):
    def __init__(self, batch_size: int, corpus_path: str, checkpoint_dir: str, tokenizer_path: str,
                 save_interval: int, eval_interval: int, gradient_accumulate: int, r:int=2, alpha:int=1, dropout:0.1=float):
        Trainer.__init__(self, batch_size, corpus_path, checkpoint_dir, tokenizer_path, save_interval, eval_interval, gradient_accumulate)

        with lora(r=r, alpha=alpha, dropout=dropout):
            self.model, self.optimizer, _ = utils.prepare_for_resuming(self.checkpoint_dir, self.learning_rate,
                                                                       model_size="supersmall", best=True,
                                                                       pretrain=False)
        self.tokenizer: Tokenizer = Tokenizer(self.tokenizer_path)
        self.train_loader: DataLoader = self.create_dataloader(tokenizer=self.tokenizer)

    def create_dataloader(self, tokenizer, block_size):
        g = torch.Generator()
        g.manual_seed(1231928)
        dataset = PromptDataset(str(self.corpus_path), tokenizer, block_size)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, generator=g)
        self.logger.info("Finishing Loading the DataLoader")

        return train_loader

    def sampling(self):
        token = self.tokenizer.encode("", bos=True)
        token = torch.tensor(token, dtype=torch.long, device="cuda")
        output = generate(self.model, token, max_new_tokens=60, temperature=0.8, top_k=4, eos_id=self.tokenizer.eos_id)
        result = self.tokenizer.decode(output)

        self.logger.info(result)
        with open("result.txt", "a") as f:
            f.write(result + "\n")

        return result


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(description='FineTuning with LoRA🚀!!!')

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--gradient_accumulate", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="./tmp/checkpoint")
    parser.add_argument("--corpus_path", type=str, default="./tmp/512_chunk.txt")
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer/corpus.model")
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument("--wandb", action="store_true")

    args = parser.parse_args()

    trainer = Trainer(
        batch_size=args.batch_size,
        corpus_path=args.corpus_path,
        checkpoint_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path,
        save_interval=args.save_interval,
        gradient_accumulate=args.gradient_accumulate,
        is_wandb=args.wandb,
        load=args.load_model
    )

    trainer.train()
