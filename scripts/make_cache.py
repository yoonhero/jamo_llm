"""Build a backward-compatible instruction-tuning HDF5 cache."""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import PROMPT_DICT, _preprocess_with_completion_masks
import utils


def build_cache(dataset_path: str, output_path: str, tokenizer_path: str,
                block_size: int = 256, eval_size: int = 600,
                seed: int = 1231928) -> Path:
    tokenizer = utils.load_hg_tokenizer(tokenizer_path)

    with open(dataset_path, "r", encoding="utf8") as f:
        data = json.load(f)
    if not data:
        raise ValueError(f"The instruction dataset is empty: {dataset_path}")

    prompt_input = PROMPT_DICT["prompt_input"]
    prompt_no_input = PROMPT_DICT["prompt_no_input"]
    sources = [
        prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        for example in data
    ]
    targets = [str(example["output"]) for example in data]

    print(f"Sample Item:\n {sources[0] + targets[0]}")
    input_ids, loss_masks = _preprocess_with_completion_masks(
        sources, targets, tokenizer, block_size, is_custom=False
    )
    np_ids = np.asarray(input_ids, dtype=np.int16)
    np_masks = np.asarray(loss_masks, dtype=np.uint8)

    total_dataset_size = np_ids.shape[0]
    eval_size = min(max(eval_size, 0), max(total_dataset_size - 1, 0))
    train_size = total_dataset_size - eval_size
    print(f"Total Dataset: {total_dataset_size} | Train: {train_size} | Eval: {eval_size}")

    order = np.random.default_rng(seed).permutation(total_dataset_size)
    np_ids = np_ids[order]
    np_masks = np_masks[order]
    training_ds, eval_ds = np_ids[:train_size], np_ids[train_size:]
    training_masks, eval_masks = np_masks[:train_size], np_masks[train_size:]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("train", data=training_ds, dtype=np.int16)
        f.create_dataset("eval", data=eval_ds, dtype=np.int16)
        f.create_dataset("train_loss_mask", data=training_masks, dtype=np.uint8)
        f.create_dataset("eval_loss_mask", data=eval_masks, dtype=np.uint8)

    print(f"Wrote cache: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./tmp/finetuning_dataset.json")
    parser.add_argument("--output_path", type=str, default="./tmp/sft-cache.hdf5")
    parser.add_argument("--tokenizer_path", type=str, default="hg_tokenizer")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--eval_size", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1231928)
    args = parser.parse_args()

    build_cache(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        tokenizer_path=args.tokenizer_path,
        block_size=args.block_size,
        eval_size=args.eval_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
