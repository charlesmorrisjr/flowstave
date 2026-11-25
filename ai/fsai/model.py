"""A small GPT-style transformer for music tokens.

forward(idx, targets)       -> training (whole sequences, causal attention)
forward_cached(idx, pk, pv) -> generation one step at a time with a key/value cache; this is the
                               function exported to ONNX for the Flowstave server.
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

PRESETS = {
    "tiny":  dict(n_layer=2, n_head=4, d_model=128),   # quick tests only
    "small": dict(n_layer=4, n_head=4, d_model=256),   # ~3M parameters
    "base":  dict(n_layer=6, n_head=6, d_model=384),   # ~11M parameters (recommended)
    "large": dict(n_layer=8, n_head=8, d_model=512),   # ~25M parameters
}


@dataclass
class Config:
    vocab_size: int
    max_len: int = 768
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    dropout: float = 0.1

    def to_dict(self):
        return asdict(self)


class Attention(nn.Module):
    def __init__(self, c, sdpa=True):
        super().__init__()
        self.n_head, self.d_head = c.n_head, c.d_model // c.n_head
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model)
        self.proj = nn.Linear(c.d_model, c.d_model)
        self.drop = c.dropout
        self.sdpa = sdpa

    def _split(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).split(self.n_head * self.d_head, dim=2)
        return [t.view(B, T, self.n_head, self.d_head).transpose(1, 2) for t in (q, k, v)]

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self._split(x)
        if self.sdpa:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop if self.training else 0.0)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
            att = att.masked_fill(mask, float("-inf")).softmax(-1)
            att = F.dropout(att, self.drop, self.training)
            y = att @ v
        return self.proj(y.transpose(1, 2).reshape(B, T, C))

    def forward_cached(self, x, pk, pv):
        """x: [B,T,C]; pk/pv: [B,H,P,Dh]. Returns output and the extended cache."""
        B, T, C = x.shape
        q, k, v = self._split(x)
        k = torch.cat([pk, k], dim=2)
        v = torch.cat([pv, v], dim=2)
        P = pk.shape[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        i = torch.arange(T, device=x.device).unsqueeze(1) + P
        j = torch.arange(k.shape[2], device=x.device).unsqueeze(0)
        att = att.masked_fill(j > i, -1e9).softmax(-1)
        y = att @ v
        return self.proj(y.transpose(1, 2).reshape(B, T, C)), k, v


class Block(nn.Module):
    def __init__(self, c, sdpa=True):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.attn = Attention(c, sdpa)
        self.mlp = nn.Sequential(nn.Linear(c.d_model, 4 * c.d_model), nn.GELU(), nn.Linear(4 * c.d_model, c.d_model), nn.Dropout(c.dropout))
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        return x + self.mlp(self.ln2(x))

    def forward_cached(self, x, pk, pv):
        a, k, v = self.attn.forward_cached(self.ln1(x), pk, pv)
        x = x + a
        return x + self.mlp(self.ln2(x)), k, v


class MusicGPT(nn.Module):
    def __init__(self, c: Config, sdpa=True):
        super().__init__()
        self.config = c
        self.tok = nn.Embedding(c.vocab_size, c.d_model)
        self.pos = nn.Embedding(c.max_len, c.d_model)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([Block(c, sdpa) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # tied embeddings
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self):
        return sum(p.numel() for p in self.parameters()) - self.pos.weight.numel()

    def forward(self, idx, targets=None, ignore_index=0):
        B, T = idx.shape
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=ignore_index)
        return logits, loss

    def forward_cached(self, idx, past_k, past_v):
        """idx [1,T]; past_k/past_v [L,1,H,P,Dh]. Returns last-position logits [1,V] and new caches."""
        T = idx.shape[1]
        P = past_k.shape[3]
        pos = torch.arange(T, device=idx.device) + P
        x = self.tok(idx) + self.pos(pos)
        ks, vs = [], []
        for li, b in enumerate(self.blocks):
            x, k, v = b.forward_cached(x, past_k[li], past_v[li])
            ks.append(k)
            vs.append(v)
        logits = self.head(self.ln_f(x[:, -1, :]))
        return logits, torch.stack(ks), torch.stack(vs)

    @torch.no_grad()
    def sample(self, prompt, max_new=600, temperature=1.0, eos=2):
        """Plain (unconstrained) sampling, used to show progress during training."""
        c = self.config
        dev = next(self.parameters()).device
        L, H, Dh = c.n_layer, c.n_head, c.d_model // c.n_head
        pk = torch.zeros(L, 1, H, 0, Dh, device=dev)
        pv = torch.zeros(L, 1, H, 0, Dh, device=dev)
        seq = list(prompt)
        x = torch.tensor([seq], device=dev)
        for _ in range(max_new):
            if len(seq) >= c.max_len:
                break
            logits, pk, pv = self.forward_cached(x, pk, pv)
            probs = (logits.float() / temperature).softmax(-1)
            nxt = int(torch.multinomial(probs, 1))
            seq.append(nxt)
            if nxt == eos:
                break
            x = torch.tensor([[nxt]], device=dev)
        return seq
