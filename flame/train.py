import os
import torch
import numpy as np
from typing import Optional
from dataclasses import dataclass

from model import Transformer
from tokenizer import Tokenizer

# training configuration
epochs = 5
eval_iters = 1
dataset = "poem"
batch_size = 32
block_size = 1080
learning_rate = 6e-4

dim: int = 256
n_layers: int = 8
n_heads: int = 8
vocab_size: int = -1
n_kv_heads: Optional[int] = None
hidden_dim: int = 1024
dropout = 0.2

device = 'cuda'
device_type = 'cuda' if 'cuda' in device else 'cpu'

@dataclass
class ModelArgs:
    dim: int = 256
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = 5000
    n_kv_heads: Optional[int] = None
    hidden_dim: int = 1024
    dropout = 0.2
    head_dim: int = dim // n_heads

    batch_size: int = 32
    block_size = 1080
    max_seq_len: int = 1080

# loading dataset
tokenizer = Tokenizer()
data_dir = os.path.join("data", dataset)
def get_batch(split):
    if split == "train":
        data = tokenizer.encode(open(os.path.join(data_dir, 'train.txt')).read(), sos=False, eos=False)
        data = torch.tensor(data, dtype=torch.long)
    else:
        data = tokenizer.encode(open(os.path.join(data_dir, 'val.txt')).read(), sos=False, eos=False)
        data = torch.tensor(data, dtype=torch.long)

    # encoding data into number
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y

# loading model
model = Transformer(ModelArgs)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# estimating loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def save_checkpoint(epoch, model, optimizer, filename):
    """Saves the model and optimizer state after each epoch."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(filename):
    """Loads a model and optimizer state from a checkpoint file."""
    checkpoint = torch.load(filename)
    return checkpoint['epoch'], checkpoint['model_state_dict'], checkpoint['optimizer_state_dict']

def train(model_dir, ex_model_dir = None):
    # if train from scratch
    model_path = os.path.join(model_dir, "flame_sm_10.pt")
    if ex_model_dir == None:
        for epoch in range(epochs):
            # every once in a while evaluate the loss on train and val sets
            if epoch % eval_iters == 0 or epoch == epochs - 1:
                losses = estimate_loss()
                print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            xb, yb = get_batch('train')
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0:
                save_checkpoint(epoch, model, optimizer, model_path)
    else:
        start_epoch, model_state_dict, _ = load_checkpoint(model_path)
        model.load_state_dict(model_state_dict)

        for epoch in range(start_epoch, epochs):
            if epoch % eval_iters == 0 or epoch == epochs - 1:
                losses = estimate_loss()
                print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            xb, yb = get_batch('train')
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0:
                save_checkpoint(epoch, model, optimizer, model_path)
