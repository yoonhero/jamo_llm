import cProfile
import io
import pstats
from pstats import SortKey
import glob 
import torch
import torch.nn as nn
from pathlib import Path
import datetime
import os
import random
import numpy as np
import re
from typing import Optional, Union
from transformers import AutoTokenizer, GPT2TokenizerFast

from sophia import SophiaG
from jamo import JAMO


def resolve_device(requested: Optional[Union[str, torch.device]] = "auto") -> torch.device:
    """Prefer CUDA, then MPS, then CPU while keeping old device arguments valid."""
    if isinstance(requested, torch.device):
        requested = requested.type

    requested = (requested or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch build has no CUDA device.")
    if requested == "mps":
        if not torch.backends.mps.is_available() or not torch.backends.mps.is_built():
            raise RuntimeError("MPS was requested, but this PyTorch build has no MPS device.")
    if requested not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"Unknown device '{requested}'. Choose auto, cuda, mps, or cpu.")
    return torch.device(requested)


def load_hg_tokenizer(tokenizer_path: Union[str, Path]):
    """Load the local GPT-2 tokenizer despite the historical config key clash."""
    tokenizer_path = Path(tokenizer_path)
    try:
        return AutoTokenizer.from_pretrained(str(tokenizer_path))
    except AttributeError as exc:
        if "add_special_tokens" not in str(exc):
            raise

        vocab_file = tokenizer_path / "vocab.json"
        merges_file = tokenizer_path / "merges.txt"
        if not vocab_file.is_file() or not merges_file.is_file():
            raise

        return GPT2TokenizerFast(
            vocab_file=str(vocab_file),
            merges_file=str(merges_file),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            add_prefix_space=False,
        )

# Save the model.
def save_model(epoch: int, model, optimizer, PATH: Path) -> None:
    PATH = Path(PATH)
    PATH.mkdir(parents=True, exist_ok=True)
    model_state_dict = {
        "format_version": 2,
        "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_name": optimizer.__class__.__name__,
        "epoch": epoch
    }   
    save_dir = PATH / f"{current()}-iter-{epoch}.tar"
    torch.save(model_state_dict, str(save_dir))

def get_last_epoch(path: Path) -> int:
    """Get the last epoch and TAR file"""
    files = glob.glob(f"{str(path)}/*.tar")
    if len(files) == 0:
        return None
    
    epochs = [get_epoch(filename) for filename in files]
    return max(epochs)

def get_epoch(filename: str) -> int:
    match = re.search(r"-iter-(\d+)(?:-\d+)?\.tar$", Path(filename).name)
    if match is None:
        raise ValueError(f"Cannot find an iteration number in checkpoint name: {filename}")
    return int(match.group(1))


def _checkpoint_files(path: Path):
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    return sorted(path.glob("*.tar"), key=get_epoch)

def prepare_for_resuming(path: Path, model_size:str, learning_rate:float, best=True, pretrain=True, device=None):
    checkpoint_files = _checkpoint_files(path)
    assert checkpoint_files, "Please Check the model is existed."
    checkpoint_path = checkpoint_files[-1] if best else checkpoint_files[0]
    model_state_dict = torch.load(str(checkpoint_path), map_location="cpu")

    model = JAMO.from_name(model_size, pretrain=pretrain)
    if device is not None:
        model = model.to(device)

    optimizer_state = model_state_dict.get("optimizer")
    if optimizer_state is not None:
        checkpoint_group = optimizer_state.get("param_groups", [{}])[0]
        optimizer_name = model_state_dict.get("optimizer_name")
        if optimizer_name is None:
            optimizer_name = "sophia" if "rho" in checkpoint_group else "adamw"

        if optimizer_name.lower().startswith("sophia") or "rho" in checkpoint_group:
            optimizer = SophiaG(
                model.parameters(),
                lr=learning_rate,
                betas=tuple(checkpoint_group.get("betas", (0.965, 0.99))),
                rho=checkpoint_group.get("rho", 0.03 if pretrain else 0.01),
                weight_decay=checkpoint_group.get("weight_decay", 0.1),
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                betas=tuple(checkpoint_group.get("betas", (0.9, 0.95))),
                weight_decay=checkpoint_group.get("weight_decay", 0.1),
            )
        optimizer.load_state_dict(optimizer_state)
    else:
        optimizer = SophiaG(model.parameters(), lr=learning_rate)

    state_dict = model_state_dict["model"]
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    start_epoch = model_state_dict["epoch"]

    return model, optimizer, start_epoch

def load_model(model_path: Path, model_size:str, device):
    model_path = Path(model_path)
    if not model_path.is_file():
        model_dirs = _checkpoint_files(model_path)
        assert model_dirs, "There're no checkpoints in that directory."
        model_path = model_dirs[-1]

    model = JAMO.from_pretrained(model_size, model_path, device=device)
    return model

def is_torch_2():
    return torch.__version__[0] == "2"

def tokenizer_setting():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

def set_seed(seed=12346):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def profile(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE  # 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return wrapper
    

def current():
    date = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    return date
