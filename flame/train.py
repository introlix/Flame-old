import os
import torch
import numpy as np
from typing import Optional

from model import Transformer, ModelArgs
from tokenizer import Tokenizer

# training configuration
epochs = 5
eval_iters = 1
dataset = "corpus"
batch_size = 6
block_size = 1080
learning_rate = 6e-4

dim: int = 256
n_layers: int = 4
n_heads: int = 4
vocab_size: int = -1
n_kv_heads: Optional[int] = None
hidden_dim: int = 1024
dropout = 0.2
head_dim = dim // n_heads

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# loading the datasets
data_dir = os.path.join("data", dataset)
tokenizer = Tokenizer()
data = torch.tensor(tokenizer.encode(open(os.path.join(data_dir, 'train.txt')).read(), sos=False, eos=False))
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# getting the batch and input & target
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data

    # get input and target
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    x, y = x.to(device), y.to(device)

    return x, y

# loading model
model_args = dict(dim = dim, n_layers = n_layers, n_heads = n_heads, vocab_size=tokenizer.n_words, n_kv_heads = None, hidden_dim = hidden_dim, head_dim=head_dim, batch_size = batch_size, max_seq_len=1080)
args = ModelArgs(**model_args)
model = Transformer(args)
model = model.to(device)
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

# training the model
def train(model_dir, ex_model_dir = None):
    # if train from scratch
    model_path = os.path.join(model_dir, f"flame_sm_{str(model.get_num_params())[:2]}M.pt")
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
                
train("./model")