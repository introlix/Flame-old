import math
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class ModelArgs:
    dim: int = 256
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1
    n_kv_heads: Optional[int] = None
    hidden_dim: int = 1024
    dropout = 0.2

    batch_size: int = 32
    block_size = 1080
    max_seq_len: int = 1080


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads = args.n_heads
        self.att_dropout = args.dropout
        self.resid_dropout = args.dropout
        self.head_dim = args.dim // args.n_heads
        self.n_kv_heads = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seqlen, n_embd = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x) # (batch_size, seqlen, n_head, head_dim)

        xq = xq.view(batch_size, seqlen, self.n_heads, self.head_dim).transpose(1, 2) # (batch_size, n_head, seqlen, head_dim)
        xk = xk.view(batch_size, seqlen, self.n_heads, self.head_dim).transpose(1, 2) # (batch_size, n_head, seqlen, head_dim)
        xv = xv.view(batch_size, seqlen, self.n_heads, self.head_dim).transpose(1, 2) # (batch_size, n_head, seqlen, head_dim)

        att = xq @ xk.transpose(-2, -1) # (batch_size, n_head, seqlen, head_dim) @ (batch_size, n_head, head_dim, seqlen) -> (batch_size, n_head, seqlen, seqlen)
        att = att * (1.0 / math.sqrt(xk.size(-1)))
        att = att.masked_fill(self.bias[:,:,:seqlen,:seqlen] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.att_dropout(att)
        y = att @ xv # (batch_size, n_head, seqlen, seqlen) x (batch_size, n_head, seqlen, head_dim) -> (batch_size, n_head, seqlen, head_dim)

        y = y.transpose(1, 2).contiguous().view(batch_size, seqlen, n_embd)

        y = self.resid_dropout(self.wo(y))

        return y
    

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.gelu = nn.GELU()
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.dropout = args.dropout

    def forward(self, x) -> torch.Tensor:
        x = self.w1(x)
        x = self.gelu(x)
        x = self.w2(x)
        x = self.dropout(x)

        return x
    
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__
        self.ln_1 = LayerNorm(args.dim, bias=False)
        self.attn = Attention(args)
        self.ln_2 = LayerNorm(args.dim, bias=False)
        self.feed_forward = FeedForward(args)

    def forward(self, x) -> torch.Tensor:
        r = self.attn(self.ln_1(x))
        h = x + r
        r = self.feed_forward(self.ln_2(h))
        out = h + r
        return out
    
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__
        self.args = args

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(args.vocab_size, args.dim),
            wpe = nn.Embedding(args.block_size, args.dim),
            drop = nn.Dropout(args.dropout),
            h = nn.ModuleList([Transformer(args) for _ in range(args.n_layers)]),
            ln_f = LayerNorm(args.dim, bias=True),
        ))

        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)


    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets = None):
        device = idx.device
        _, seqlen = idx.size()

        assert seqlen <= self.args.block_size, f"Cannot forward sequence of length {seqlen}, block size is only {self.args.block_size}"

        pos = torch.arange(0, seqlen, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.args.block_size else idx[:, -self.args.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx