"""A small decoder-only GPT, self-contained. ~60 lines of your own PyTorch.

This is the R4 anatomy: token + position embeddings -> N blocks -> final norm -> LM head.
Each block is pre-norm attention and pre-norm MLP with residual connections.

Attention itself is one library call. You proved in R3 that your hand-written version
matches `F.scaled_dot_product_attention`; this week the point is the training loop, so
use the library call with `is_causal=True` and spend your time on the loop.

    python model.py       builds the model and runs one forward pass on random tokens.
                          Loss at init must be close to ln(vocab_size) ≈ 4.17 for vocab 65.

Shapes (B batch, T block_size, C n_embd, H n_head, V vocab_size):
    idx          (B, T)        int64 token ids
    tok_emb      (B, T, C)     + pos_emb (T, C) broadcast
    inside attn  q, k, v       (B, H, T, C // H)   after reshape + transpose
    logits       (B, T, V)
    loss         scalar        cross-entropy over all B*T positions
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 256     # T: max context the model is trained on
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128         # C
    dropout: float = 0.0      # leave at 0 until the pathology runs; then try 0.2 on `diverge`


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        # TODO(1): one Linear producing q, k, v together: nn.Linear(C, 3 * C).
        #          One output projection nn.Linear(C, C). Store n_head and n_embd.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # TODO(2): qkv = self.qkv(x).split(C, dim=2); reshape each to (B, T, H, C // H) and
        #          transpose(1, 2) -> (B, H, T, C // H).
        #          y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=...)
        #          Merge heads back: y.transpose(1, 2).contiguous().view(B, T, C); return self.proj(y).
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # TODO(3): Linear(C, 4C) -> GELU -> Linear(4C, C) -> Dropout.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # TODO(4): ln1, attn, ln2, mlp. Pre-norm: x = x + attn(ln1(x)); x = x + mlp(ln2(x)).
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        # TODO(5): tok_emb = nn.Embedding(V, C); pos_emb = nn.Embedding(T, C);
        #          blocks = nn.ModuleList([Block(cfg) for _ in range(n_layer)]);
        #          ln_f = nn.LayerNorm(C); head = nn.Linear(C, V, bias=False).
        #          Optional but standard: tie head.weight to tok_emb.weight.
        raise NotImplementedError

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence longer than block_size"
        # TODO(6): pos = torch.arange(T, device=idx.device)
        #          x = tok_emb(idx) + pos_emb(pos); for block in blocks: x = block(x)
        #          logits = head(ln_f(x))                                   -> (B, T, V)
        #          loss = None if targets is None else
        #                 F.cross_entropy(logits.view(-1, V), targets.view(-1))
        #          return logits, loss
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        """Autoregressive sampling: the R1 loop, now with your own model."""
        # TODO(7): for each step: crop idx to the last block_size tokens; logits, _ = self(idx_cond);
        #          take logits[:, -1, :] / temperature; probs = softmax; next = torch.multinomial(probs, 1);
        #          idx = torch.cat([idx, next], dim=1). Return idx.
        raise NotImplementedError

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    cfg = GPTConfig()
    model = GPT(cfg)
    print(f"{model.num_params() / 1e6:.2f}M parameters")
    idx = torch.randint(0, cfg.vocab_size, (4, 32))
    logits, loss = model(idx, idx)
    print("logits", tuple(logits.shape), "loss at init", round(loss.item(), 3),
          "(expect ~", round(torch.log(torch.tensor(float(cfg.vocab_size))).item(), 3), ")")
