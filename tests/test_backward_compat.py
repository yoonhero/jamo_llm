import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

import utils
from dataset import IterablDataset, PromptDataset
from sophia import SophiaG


class BackwardCompatibilityTests(unittest.TestCase):
    def test_historical_checkpoint_iteration_name(self):
        self.assertEqual(utils.get_epoch("2023-06-30T03_42_25-iter-300000-002.tar"), 300000)
        self.assertEqual(utils.get_epoch("2023-07-04T00_00_00-iter-20.tar"), 20)

    def test_local_hg_tokenizer_loads_with_current_transformers(self):
        tokenizer = utils.load_hg_tokenizer("hg_tokenizer")
        self.assertEqual(tokenizer.pad_token_id, 1)
        self.assertEqual(tokenizer.eos_token_id, 2)
        self.assertGreater(len(tokenizer.encode("안녕")), 0)

    def test_legacy_prompt_cache_masks_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "legacy.hdf5"
            with h5py.File(cache_path, "w") as h5f:
                h5f.create_dataset("train", data=np.array([[0, 9, 1, 1, 2]], dtype=np.int16))
                h5f.create_dataset("eval", data=np.empty((0, 5), dtype=np.int16))

            dataset = PromptDataset(cache_dir=str(cache_path), device="cpu")
            x, y = dataset[0]
            self.assertEqual(x.tolist(), [0, 9, 1, 1])
            self.assertEqual(y.tolist(), [9, -1, -1, 2])

    def test_iterable_cache_keeps_historical_three_value_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "pretrain.hdf5"
            with h5py.File(cache_path, "w") as h5f:
                h5f.create_dataset("tokens", data=np.array([[0, 9, 1, 2]], dtype=np.int16))

            tokenizer = utils.load_hg_tokenizer("hg_tokenizer")
            dataset = IterablDataset(cache_path, tokenizer, block_size=3,
                                     cache_dir=str(cache_path), device="cpu")
            x, y, mask = dataset[0]
            self.assertEqual(x.tolist(), [0, 9, 1])
            self.assertEqual(y.tolist(), [9, -1, 2])
            self.assertEqual(mask.tolist(), [True, False, True])

    def test_new_completion_mask_is_consumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "masked.hdf5"
            with h5py.File(cache_path, "w") as h5f:
                h5f.create_dataset("train", data=np.array([[0, 9, 8, 1, 2]], dtype=np.int16))
                h5f.create_dataset("eval", data=np.empty((0, 5), dtype=np.int16))
                h5f.create_dataset("train_loss_mask", data=np.array([[0, 0, 1, 0, 1]], dtype=np.uint8))
                h5f.create_dataset("eval_loss_mask", data=np.empty((0, 5), dtype=np.uint8))

            dataset = PromptDataset(cache_dir=str(cache_path), device="cpu")
            _, y = dataset[0]
            self.assertEqual(y.tolist(), [-1, 8, -1, 2])

    def test_sophia_updates_hessian_and_uses_batch_override(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = SophiaG([parameter], lr=0.1, rho=0.01, batch_size=1)
        optimizer.set_batch_size(4)
        before = parameter.detach().clone()
        parameter.square().sum().backward()
        optimizer.step()

        self.assertGreater(optimizer.state[parameter]["hessian"].item(), 0.0)
        self.assertNotEqual(parameter.item(), before.item())
        self.assertEqual(optimizer.batch_size, 4)


if __name__ == "__main__":
    unittest.main()
