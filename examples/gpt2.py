"""
references:
1) Andrej Karpathy's nanoGPT: 
https://github.com/karpathy/nanoGPT
2) Andrej Karpathy's youtube lecture:
https://youtu.be/kCc8FmEb1nY?si=tlORBqCQQI7Xyljs
"""
from dataclasses import dataclass
from cranberry import Tensor, nn
import math

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class LayerNorm:
    def __init__(self, ndim, bias):
        self.weight = Tensor.ones(ndim)
        self.bias = Tensor.zeros(ndim) if bias else None

    def __call__(self, input: Tensor):
        # TODO: implement layer_norm
        return input.layer_norm(weight=self.weight, bias=self.bias, eps=1e-5)
    
class CausualSelfAttention:
    def __init__(self, config: GPTConfig):
        assert config.n_embd % config.n_head == 0
        # key, query, value projection for all heads
        self.c_key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_query = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.p = config.dropout

    def __call__(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        q, k, v = self.c_query(x), self.c_key(x), self.c_value(x)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # manual implementation of attention
        # TODO: transpose(-2, -1)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    
        # TODO: not implemented
        NotImplementedError

class MLP:
    def __init__(self, config: GPTConfig):
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.p = config.dropout
    
    def __call__(self, x):
        x = self.c_fc(x).gelu()
        x = self.c_proj(x).dropout(self.p)
        return x
    
class Block:
    def __init__(self, config: GPTConfig):
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausualSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT:
    def __init__(self, config: GPTConfig):
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        raise NotImplementedError