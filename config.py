from dataclasses import dataclass


@dataclass
class GPTHyperParameters:
    opt_betas = (0.9, 0.95)
    opt_eps = 1e-8
    gl_norm_gradient = 1.0

@dataclass
class DDPInfo:
    is_available: int = 0
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    master_process: bool = True

@dataclass
class LrConfig:
    max_steps: int = 50
    warmup_steps: int = 10
    max_lr: float = 6e-4
    min_lr: float = max_lr * 0.1

@dataclass
class GPTConfig:
    # model parmas
    block_size: int = 1024      # max sequence length
    vocab_size: int = 50257     # number of tokens: 50,000 BPE merges + 256 bytes token + 1 <|endoftext|> token
    n_layer: int = 12           # nuber of layers
    n_head: int = 12            # number of heads
    n_embd: int = 768           # embedding dimensions

    # train params
    val_interval = 100                  # validation interval
    val_loss_steps = 20                  # validation loss calculation steps
    total_batch_size: int = 524288      # 2**19 ~0,5M in number of tokens
    micro_batch: int = 2                # micro batch size
    sequence_length = 1024              # sequence length


    lr_config: LrConfig = LrConfig(     # learning_rate settings
        max_steps=19073,
        warmup_steps=715
    )
