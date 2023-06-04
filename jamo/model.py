import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing_extensions import Self

@dataclass 
class JamoConfig: 
    n_embd: int
    n_heads: int
    n_layer: int
    vocab_size: int=10000
    block_size:int=512
    dropout: int = 0.2

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(**jamo_configs[name])


jamo_configs = {
    "small": dict(n_layer=24, n_heads=16, n_embd=1024),
    "medium": dict(n_layer=30, n_heads=16, n_embd=2048),
    "large": dict(n_layer=40, n_heads=26, n_embd=6656),
    "enormous": dict(n_layer=50, n_heads=64, n_embd=8192),
}

class JAMO(nn.Module):
    def __init__(self, config: JamoConfig) -> None:
        super().__init__()
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = RMSNorm(config.n_embd)

        self.apply(self._init_weights)

        self.rope_cache = None
        self.mask_cache = None
        self.kv_caches = []

        print(f"Number of parameters: {human_format(self.get_num_params())}")

    def get_num_params(self):
        n_params = [p.nelement() for p in self.parameters()]
        num = sum(n_params)
        return num

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))

    def forward(self, idx,  max_seq_length=None, input_pos=None):
        B, T = idx.shape

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        assert T <= max_seq_length, f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        assert max_seq_length <= block_size, f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert T <= block_size, f"Cannot forward sequence of length {T}, block size is only {block_size}"

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx)
        if self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(idx)

        if input_pos is not None:
            rope = self.rope_cache.index_select(0, input_pos)
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
        else:
            rope = self.rope_cache[:T]
            mask = self.mask_cache[:, :, :T, :T]


        x = self.wte(idx)

        if input_pos is None:  # proxy for use_cache=False
            for block in self.blocks:
                x, _ = block(x, rope, mask, max_seq_length)
        else:
            if not self.kv_caches:
                head_size = self.config.n_embd // self.config.n_heads
                cache_shape = (B, self.config.n_heads, max_seq_length, head_size)
                self.kv_caches = [
                    (torch.zeros(cache_shape, device=x.device, dtype=x.dtype), torch.zeros(cache_shape, device=x.device, dtype=x.dtype))
                    for _ in range(self.config.n_layer)
                ]
            for i, block in enumerate(self.blocks):
                x, self.kv_caches[i] = block(x, rope, mask, max_seq_length, input_pos, self.kv_caches[i])

        x = self.ln_f(x)

        logits = self.lm_head(x) 

        return logits

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(JamoConfig.from_name(name))

    def build_rope_cache(self, idx: torch.Tensor) -> torch.Tensor:
        return build_rope_cache(
            seq_len=self.config.block_size,
            n_elem=self.config.n_embd // self.config.n_heads,
            dtype=idx.dtype,
            device=idx.device,
        )

    def build_mask_cache(self, idx: torch.Tensor) -> torch.Tensor:
        ones = torch.ones((self.config.block_size, self.config.block_size), device=idx.device, dtype=torch.bool)
        return torch.tril(ones).unsqueeze(0).unsqueeze(0)

    def reset_cache(self) -> None:
        self.kv_caches.clear()
        if self.mask_cache.device.type == "xla":
            # https://github.com/Lightning-AI/lit-parrot/pull/83#issuecomment-1558150179
            self.rope_cache = None
            self.mask_cache = None

    def __repr__(self):
        return f">> {self.get_num_params()} Paramters <<"


class Block(nn.Module):
    def __init__(self, config: JamoConfig):
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.sa = CasualAttention(config)
        self.rms_2 = RMSNorm(config.n_embd)
        self.ffwd = FeedForward(config)
    
    def forward(self, x: torch.Tensor, rope: torch.Tensor, mask: torch.Tensor, max_seq_length: int, input_pos=None, kv_cache=None):
        B, T, C = x.shape
        h, new_kv_cache = self.attn(self.rms_1(x), rope, mask, max_seq_length, input_pos, kv_cache)
        x = x + h
        x = x + self.ffwd(self.rms_2(x))
        return x, new_kv_cache


class CasualAttention(nn.Module):
    def __init__(self, config: JamoConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_heads == 0, "Please Check Embedding and Heads Number Config."
        self.head_size = config.n_embd//config.n_heads

        self.c_attn = nn.Linear(config.n_embd, config.n_embd*3, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
    
        self.n_head = config.n_heads
        self.n_embd = config.n_embd
        self.block_size = config.block_size

        self.dropout = config.dropout
        if config.dropout:
            self.attn_dropout = nn.Dropout(self.dropout)
            self.resid_drop = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor, rope: torch.Tensor, mask: torch.Tensor, max_seq_length: int, input_pos=None, kv_cache=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        head_size = C//self.n_head
        k = k.view(B, T, self.n_head, head_size)
        q = q.view(B, T, self.n_head, head_size)
        v = v.view(B, T, self.n_head, head_size)

        q = apply_rope(q, rope)
        k = apply_rope(k, rope)

        k = k.transpose(1, 2)  # (B, nh, T, hs)
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            # check if reached token limit
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=2)
                cache_v = torch.roll(cache_v, -1, dims=2)
            k = cache_k.index_copy(2, input_pos, k)
            v = cache_v.index_copy(2, input_pos, v)
            kv_cache = k, v

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)
        y = self.resid_drop(self.c_proj(y)) if self.dropout else self.c_proj(y)

        return y, kv_cache


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4*config.n_embd
        n_hidden = int(2 * hidden_dim / 3)

        N = 256
        n_hidden = ((n_hidden - 1) // N) * N + N

        self.c_fc1 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, config.n_embd, bias=False)
        
    def forward(self, x):
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    num = int(num * 10) / 10
    return f"{f'{num:f}'.rstrip('0').rstrip('.')}{['', 'k', 'M', 'B', 'T'][magnitude]}"


def build_rope_cache(
    seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
):
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).float()

    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        cache = cache.half()
    return cache


def apply_rope(x: torch.Tensor, rope_cache) -> torch.Tensor:
    # truncate to support variable sizes
    T = x.size(1)
    rope_cache = rope_cache[:T]

    # cast because the reference does
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)