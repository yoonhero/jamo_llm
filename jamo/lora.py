import torch
import torch.nn as nn
import torch.nn.functional as F
import jamo.model as jamo
import math
from contextlib import contextmanager
from dataclasses import dataclass


## LoRA
class LoRALayer():
    def __init__(self, r, lora_alpha, lora_dropout):
        self.r = r
        self.lora_alpha = lora_alpha

        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        self.merged = False

class LoRALinear(nn.Linear, LoRALayer):
    def __init__(self, fan_in, fan_out, r, lora_alpha, lora_dropout, **kwargs):
        nn.Linear.__init__(self, fan_in, fan_out, **kwargs)
        LoRALayer.__init__(self, r, lora_alpha, lora_dropout)

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros((fan_in, r))) # (in, rank)
            self.lora_B = nn.Parameter(torch.zeros((fan_out, r))) # (out, rank)
            self.scailing = lora_alpha / r
            self.weight.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode:bool=True):
        nn.Linear.train(self, mode)

        if mode: # for the training mode
            # lora enabled and merged -> unmerge the weight
            if self.r > 0. and self.merged:
                self.weight.data -= self.lora_A @ self.lora_B.T * self.scailing # (in, rank) @ (rank, out) => (in, out)
                self.merged = False
        else: # for the eval mode
            # lora disabled and not merged -> merge
            if self.r > 0. and not self.merged:
                self.weight.data += self.lora_A @ self.lora_B.T * self.scailing
                self.merged = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r > 0. and not self.merged:
            result = F.linear(x, self.weight, bias=self.bias)
            ab = (self.lora_dropout(x) @ self.lora_A @ self.lora_B.T)*self.scailing # (_, in) @ (in, r) @ (r, out)
            result = result + ab
        else:
            result = F.linear(x, self.weight, bias=self.bias)

        return result


class CasualAttention(jamo.CasualAttention):
    def __init__(self, config: jamo.JamoConfig) -> None:
        nn.Module.__init__(self)

        assert config.n_embd % config.n_heads == 0, "Please Check Embedding and Heads Number Config."
        self.head_size = config.n_embd // config.n_heads

        self.c_attn = LoRALinear(config.n_embd, config.n_embd*3, r=2, lora_alpha=1, lora_dropout=0.1, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_heads
        self.n_embd = config.n_embd
        self.block_size = config.block_size

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        if not jamo.is_torch_2():
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

@dataclass
class LoRAConfig:
    r: float = 0.0
    alpha: float = 1.0
    dropout: float = 0.0

@contextmanager
def lora(r, alpha, dropout):
    CasualAttention.lora_config = LoRAConfig(r=r, alpha=alpha, dropout=dropout)
    causal_self_attention = jamo.CasualAttention
    jamo.CausalSelfAttention = CasualAttention
    yield
    jamo.CausalSelfAttention = causal_self_attention
    CasualAttention.lora_config = None


if __name__ == "__main__":
    lora = LoRALinear(20, 10, 2, 1, 0)

    input = torch.randn((10, 20))
    output = lora(input)
    print(output, output.shape)