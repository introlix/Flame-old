import os
import torch
import numpy as np

from model import Transformer
from tokenizer import Tokenizer

# training configuration
epochs = 5
eval_iters = 1
dataset = "poem"
batch_size = 32
block_size = 1080
learning_rate = 6e-4
device = 'cuda'
device_type = 'cuda' if 'cuda' in device else 'cpu'

# loading dataset
tokenizer = Tokenizer()
data_dir = os.path.join("data", dataset)
def get_batch(split):
    if split == "train":
        data = np.memmap(os.path.join(data_dir, 'train.txt'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.txt'), dtype=np.uint16, mode='r')

    # encoding data into number
    data = tokenizer.encode(data, sos=False, eos=False)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y

# loading model
model = Transformer()
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
        for epoch in epochs:
            # every once in a while evaluate the loss on train and val sets
            if iter % eval_iters == 0 or iter == epochs - 1:
                losses = estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

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
            if iter % eval_iters == 0 or iter == epochs - 1:
                losses = estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            xb, yb = get_batch('train')
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0:
                save_checkpoint(epoch, model, optimizer, model_path)