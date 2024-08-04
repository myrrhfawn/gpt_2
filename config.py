from dataclasses import dataclass
@dataclass
class GPTConfig:
    block_size: int = 1024      # max sequence length
    vocab_size: int = 50257     # number of tokens: 50,000 BPE merges + 256 bytes token + 1 <|endoftext|> token
    n_layer: int = 12           # nuber of layers
    n_head: int = 12            # number of heads
    n_embd: int = 768           # embedding dimensions

@dataclass
class GPTHyperParameters:
    opt_betas = (0.9, 0.95)
    opt_eps = 1e-8
    gl_norm_gradient = 1.0
