import time
import torch
from torch.nn import functional as F
from dataloaders import DataLoaderLite
from config import GPTConfig, GPTHyperParameters
from model import GPT, get_lr

print(f"Torch is available: {torch.cuda.is_available()}")
print(f"Torch version: {torch.__version__}")

def test_pretrained():
    device = get_available_device()
    num_return_sequences = 5
    max_length = 30

    model = GPT.from_pretrained('gpt2')
    model.eval()
    model.to(device)

    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode("Hello, I`m language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)  # (8, )
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5 ,8)
    x = tokens.to(device)

    # generate! right now x is (B, T) where  B = 5,  T = 8
    # set the seed to 42
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    while x.size(1) < max_length:
        # forward the model to get the logits
        with torch.no_grad():
            logits = model(x)  # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :]  # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token  from top-k probabilities
            ix = torch.multinomial(topk_probs, 1)  # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
            # append to the sequence
            x = torch.cat((x, xcol), dim=1)

    # print the generated text
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)
    print("SUCCESS, yeah!")

def get_available_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")
    return device




if __name__ == "__main__":

    device = get_available_device()
    num_return_sequences = 5
    max_length = 30
    max_steps = 50

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    train_loader = DataLoaderLite(2, 1024)
    torch.set_float32_matmul_precision('high')      # TF32, 1.8x performance, default is 'highest' (FP32)
    # get logits
    params = GPTHyperParameters()
    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)
    # compile model (to dont use python interpretation and use something like compiled code)
    # transfer data from GPU HBM(High bandwidth memory) to GPU SRAM
    #model = torch.compile(model)    # 2x performance
    #logits, loss = model(x, y)

    # optimazer
    optimazer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=params.opt_betas, eps=params.opt_eps)
    for step in range(max_steps):
        t0 = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimazer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):  # 1.2x performance
            logits, loss = model(x, y)
            # import code; code.interact(local=locals())
        loss.backward()
        # to prevent model getting too big of shocks in terms of the gradient magnitude
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # determine and set the learning rate for this iteration
        lr = get_lr(step, max_steps)
        for param_group in optimazer.param_groups:
            param_group['lr'] = lr
        optimazer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000   # time difference in milliseconds
        tokens_per_seconds = (train_loader.b * train_loader.t) / (t1 - t0)
        print(f"Step({step}/{max_steps}), loss: {loss.item()} lr: {lr:.4e} dt: {dt:.2f}ms tok/sec: {tokens_per_seconds}, norm: {norm:.4f}")
    print(loss)

    print(logits.shape)
    print("SUCCESS, yeah!")