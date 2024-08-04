import tiktoken
import torch


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
