# GPT-2 White Paper:
# https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
# Hyperparameters are taken from GPT-3 White Paper
# https://arxiv.org/pdf/2005.14165
# Attention Is All You Need:
# https://arxiv.org/pdf/1706.03762
# Flash Attention
# https://arxiv.org/pdf/2205.14135
# Flash Attention 2
# https://arxiv.org/pdf/2307.08691

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import GPTConfig

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projection for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really `bias`, more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        b, t, c = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values, for all heads in batch and move head forward to be the batch
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so C=768 channels in transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)     # (B, nh, T, hs)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)     # (B, nh, T, hs)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)     # (B, nh, T, hs)
        # attention (materializes the large (T, T) matrix for all the queries and keys)
        # Causal:
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :t, :t] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v     # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Flash Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(b, t, c)    # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class TanhGELU(nn.Module):
    """Tanh Gaussian Error Linear Unit"""
    def forward(self, input):
        # same as nn.GELU(approximate='tanh')
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class MLP(nn.Module):
    """MultiLayer Perceptron"""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1


    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape(B, T)
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # forward the token position embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)   # shape (T)
        #   forward the GPT model itself
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        # forward the blocks of transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"Loading weights from pretrained gpt: {model_type}")

        # n_layer, n_head and n_embd are determined from model type
        config_params = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),    # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),   # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),   # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),   # 1558M params
        }[model_type]
        config_params['vocab_size'] = 50257     # always 50257 for GPT model checkpoints
        config_params['block_size'] = 1024      # always 1024 for GPT model checkpoints

        # create a from scratch initialized minGPT model
        config = GPTConfig(**config_params)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

        # init huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all the parameters are aligned and match in names shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]     # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]            # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for  the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())

            else:
                # vanilla copy over the parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        for k, v in sd.items():
            print(f"{k} {v.shape}")
        return model

def get_lr(it, max_steps=50):
    """Returns learning rate with 10 steps of linear warmup and decay down"""
    max_lr = 3e-4
    min_lr = max_lr * 0.1
    warmup_steps = 10

    # 1) linear warm-up for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, returns min learning rates
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min lr_rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))       # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

