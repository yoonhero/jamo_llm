import cProfile
import io
import pstats
from pstats import SortKey
import glob 
import torch
import torch.nn as nn
from pathlib import Path
import datetime

from sophia import SophiaG
from jamo import JAMO

# Save the model.
def save_model(epoch: int, model, optimizer, PATH: Path) -> None:
    model_state_dict = {
        "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }   
    save_dir = PATH / f"{current()}-iter-{epoch}.tar"
    torch.save(model_state_dict, str(save_dir))

def get_last_epoch(PATH: str) -> int:
    """Get the last epoch and TAR file"""
    path = Path(PATH)
    files = glob.glob(f"{str(path)}/*")
    if len(files) == 0:
        return None
    
    epochs = [int(filename.split("/")[-1].split(".")[0].split("-")[-1]) for filename in files]
    return max(epochs)

def load_model(path: Path, model_size:str, learning_rate:float, best=True, pretrain=True):
    model = JAMO.from_name(model_size, pretrain=pretrain)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95))
    rho = 0.03 if pretrain else 0.01
    weight_decay = 0.2 if pretrain else 0.1
    optimizer = SophiaG(model.parameters(), lr=learning_rate, betas=(0.965, 0.99), rho = rho, weight_decay=weight_decay)

    if best:
        last_epoch = get_last_epoch(str(path))
        path = path / f"iter-{last_epoch}.tar"
        model_state_dict = torch.load(str(path))
    else:
        assert path.exists(), "Please Check the model is existed."
        model_state_dict = torch.load(str(path))

    state_dict = model_state_dict["model"]
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    optimizer.load_state_dict(model_state_dict["optimizer"])
    start_epoch = model_state_dict["epoch"]

    return model, optimizer, start_epoch

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