import math
import os
import time

import tiktoken
import torch.distributed as dist

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from dataloaders import DataLoaderLite, DistributedDataLoader
from config import GPTConfig, GPTHyperParameters, DDPInfo, LrConfig
from model import GPT, get_lr
from torch.distributed import init_process_group, destroy_process_group

print(f"Torch is available: {torch.cuda.is_available()}")
print(f"Torch version: {torch.__version__}")

FIXED_SEED = True

class ModelTrainer:

    def __init__(
            self,
            config: GPTConfig,
            dataloder: str = 'distributed',
            use_compile: bool = False

    ):
        self.config = config
        self.use_compile = use_compile
        self.dataloader_type = dataloder
        self.enc = tiktoken.get_encoding('gpt2')

        if FIXED_SEED:
            torch.manual_seed(1337)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(1337)
        self.setup_ddp()

        assert self.config.total_batch_size % \
               (self.config.micro_batch * self.config.sequence_length * self.ddp.world_size) == 0, \
            "Make sure total_batch_size is divisible by B * T * ddp.world_size"
        self.grad_accum_steps = self.config.total_batch_size // \
                                (self.config.micro_batch * self.config.sequence_length * self.ddp.world_size)
        self.logger(f"Total desired batch size: {self.config.total_batch_size}")
        self.logger(f"=> calculated gradient accumulation steps: {self.grad_accum_steps}")
        self.create_dataloaders()
        torch.set_float32_matmul_precision('high')  # TF32, 1.8x performance, default is 'highest' (FP32)
        self.init_model()

    def init_model(self):
        # create model
        self.model = GPT(self.config)
        self.model.to(self.device)
        if self.use_compile:
            self.model = torch.compile(self.model)  # 2x performance
            # compile model (to don`t use python interpretation and use something like compiled code)
            # transfer data from GPU HBM(High bandwidth memory) to GPU SRAM
        if self.ddp.is_available:
            self.model = DDP(self.model, device_ids=[self.ddp.local_rank])

        # always contains the "raw" unwrapped model
        self.raw_model = self.model.module if self.ddp.is_available else self.model

        # optimizer
        self.optimizer = self.raw_model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=self.device)

    def setup_ddp(self):
        self.ddp = DDPInfo(
            is_available=int(os.environ.get('RANK', -1) != -1),  # this is a ddp run?
        )
        if self.ddp.is_available:
            # torchrun --standalone --nproc_per_node=1 train_gpt2.py
            # use of DDP atm demands CUDA, we set the device appropriately according to rank
            assert torch.cuda.is_available(), "We need CUDA for DDP"
            init_process_group(backend='nccl')
            self.ddp.rank = int(os.environ['RANK'])
            self.ddp.local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp.world_size = int(os.environ['WORLD_SIZE'])
            self.device = f'cuda:{self.ddp.local_rank}'
            torch.cuda.set_device(self.device)
            self.ddp.master_process = self.ddp.rank == 0  # this process will do logging, checkpoints etc.
        else:
            # vanilla settings already in ddp info, non-DDP run
            self.device = self.get_available_device()

    def create_dataloaders(self):
        if 'distributed' in self.dataloader_type:
            self.train_dataloader = DistributedDataLoader(
                B=self.config.micro_batch,
                T=self.config.sequence_length,
                ddp=self.ddp,
                split="train"
            )
            self.val_dataloader = DistributedDataLoader(
                B=self.config.micro_batch,
                T=self.config.sequence_length,
                ddp=self.ddp,
                split="val"
            )
        else:
            self.train_dataloader = DataLoaderLite(B=self.config.micro_batch, T=self.config.sequence_length)
            self.val_dataloader = DataLoaderLite(B=self.config.micro_batch, T=self.config.sequence_length)

    def get_available_device(self):
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        self.logger(f"Using device: {device}")
        return device

    def logger(self, text):
        """Print log only in master process"""
        if self.ddp.master_process:
            print(text)

    def val_step(self):
        """Makes validation step for model"""
        self.model.eval()
        self.val_dataloader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            for _ in range(self.config.val_loss_steps):
                x, y = self.val_dataloader.next_batch()
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):  # 1.2x performance
                    logits, loss = self.model(x, y)
                loss = loss / self.config.val_loss_steps
                val_loss_accum += loss.detach()
        if self.ddp.is_available:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        self.logger(f"Step(val) - loss: {val_loss_accum.item():.4f}")

    def train_step(self, step):
        """"""
        self.model.train()
        self.optimizer.zero_grad()
        t0 = time.time()
        loss_accum = 0.0
        for micro_step in range(self.grad_accum_steps):
            x, y = self.train_dataloader.next_batch()
            x, y = x.to(self.device), y.to(self.device)
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):  # 1.2x performance
                logits, loss = self.model(x, y)
            loss = loss / self.grad_accum_steps
            loss_accum += loss.detach()
            if self.ddp.is_available:
                self.model.require_backward_grad_sync = (micro_step == self.grad_accum_steps - 1)
            loss.backward()
        if self.ddp.is_available:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        # to prevent model getting too big of shocks in terms of the gradient magnitude
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        # determine and set the learning rate for this iteration
        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000  # time difference in milliseconds
        tokens_per_seconds = (self.train_dataloader.b * self.train_dataloader.t * self.grad_accum_steps) / (t1 - t0)
        self.logger(
            f"Step({step}/{self.config.lr_config.max_steps}), loss: {loss_accum.item():.6f} lr: {lr:.4e}, "
            f"dt: {dt:.2f}ms, tok/sec: {tokens_per_seconds:.2f}, norm: {norm:.4f}"
        )

    def train(self):
        """Train model using GPTConfig"""
        for step in range(self.config.lr_config.max_steps):
            # once in a while evaluate our validation loss
            if step % self.config.val_interval == 0:
                self.val_step()
                self.generate()
            self.train_step(step)
        self.logger("Train finish successfully")

    def get_lr(self, step):
        """Returns learning rate with 10 steps of linear warmup and decay down"""
        # 1) linear warm-up for warmup_iters steps
        if step < self.config.lr_config.warmup_steps:
            return self.config.lr_config.max_lr * (step + 1) / self.config.lr_config.warmup_steps
        # 2) if step > lr_decay_iters, returns min learning rates
        if step > self.config.lr_config.max_steps:
            return self.config.lr_config.min_lr
        # 3) in between, use cosine decay down to min lr_rate
        decay_ratio = (step - self.config.lr_config.warmup_steps) / (
                    self.config.lr_config.max_steps - self.config.lr_config.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
        return self.config.lr_config.min_lr + coeff * (self.config.lr_config.max_lr - self.config.lr_config.min_lr)

    def generate(self):
        """Generate text using trained model"""
        self.model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = self.enc.encode("Hello, I`m language model, ")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(self.device)
        sample_rng = torch.Generator(device=self.device)
        if FIXED_SEED:
            sample_rng.manual_seed(42 + self.ddp.rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                logits, loss = self.model(xgen)
                # take the logits at the last position
                logits = logits[:, -1, :]   # B (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # to do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
                # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = self.enc.decode(tokens)
            print(f"Rank {self.ddp.rank} - sample[{i}]: {decoded}")


    def __del__(self):
        if self.ddp.is_available:
            destroy_process_group()






if __name__ == "__main__":
    # runner = ModelTrainer(config=GPTConfig(lr_config=LrConfig(max_steps=10, warmup_steps=1)), dataloder='lite')
    runner = ModelTrainer(config=GPTConfig(lr_config=LrConfig(max_steps=10, warmup_steps=1)))
    runner.train()
