# Tiny Shakespeare Dataset
# https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
import os

import tiktoken
import torch
import numpy as np

from config import DDPInfo


def load_tokens(filename: str) -> torch.tensor:
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T):
        self.b = B
        self.t = T

        # at init load tokens form disk and store them in memory
        with open("tinyshakespeare.txt", 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded: {len(self.tokens)} tokens")
        print(f"1 epochs: {len(self.tokens) // (B * T)} batches")
        self.reset()

    def reset(self):
        # state
        self.current_position = 0


    def next_batch(self):
        b, t = self.b, self.t
        buf = self.tokens[self.current_position:self.current_position + b*t + 1]
        x = (buf[:-1]).view(b, t)   # inputs
        y = (buf[1:]).view(b, t)    # targets
        # advance the position in the tensor
        self.current_position += b * t
        # if loading next batch would be out of bounds, reset
        if self.current_position + (b * t + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

class DistributedDataLoader:
    def __init__(self, B: int, T: int, ddp: DDPInfo, split, data_root: str = "datasets/fineweb"):
        self.b = B
        self.t = T
        self.process_rank = ddp.rank
        self.num_processes = ddp.world_size
        assert split in {'train', 'val'}

        # get the shard filenames
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"No shards found for split {split}"
        if ddp.master_process:
            print(f"Found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.b * self.t * self.process_rank

    def next_batch(self):
        b, t = self.b, self.t
        buf = self.tokens[self.current_position:self.current_position + b*t + 1]
        x = (buf[:-1]).view(b, t)   # inputs
        y = (buf[1:]).view(b, t)    # targets
        # advance the position in the tensor
        self.current_position += b * t * self.num_processes
        # if loading next batch would be out of bounds, reset
        if self.current_position + (b * t * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_position + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.b * self.t * self.process_rank
        return x, y
