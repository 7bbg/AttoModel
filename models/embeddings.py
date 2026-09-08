"""
Embeddings, Tied Input/Output Projections, Learnable Meta-Tokens, RMSNorm, and RoPE.

Designed for sub-20M parameter allocation where parameter efficiency is paramount:
- Tied V x d_model embeddings matrix (V = 8192, d_model = 512 -> 4.19M parameters).
- Learnable Meta-Tokens (6 x 512) acting as dedicated attention-sink targets.
- Rotary Position Embeddings (RoPE) for long context scaling.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    More computationally efficient than standard LayerNorm with identical stabilization.
    """

    def __init__(self, dim: int, eps: float = 1.0e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) applied to attention queries and keys.
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequency inverse table
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        return cos, sin

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies RoPE to q, k tensors of shape (B, H, T, D).
        """
        cos, sin = self._compute_cos_sin(seq_len + offset, q.device, q.dtype)
        cos = cos[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
        sin = sin[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)

        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        return q_rot, k_rot


class TiedEmbedding(nn.Module):
    """
    Tied Input and Output Embeddings with Learnable Meta-Token Offloading.
    
    Vocabulary matrix W_emb (V x d_model) is reused for language modeling head projection:
      logits = F.linear(x, W_emb)
    
    Learnable meta-tokens M (6 x d_model) are prepended to prompt sequences to act
    as dedicated attention sinks, eliminating background activation noise.
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 512,
        n_meta_tokens: int = 6,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_meta_tokens = n_meta_tokens

        # Main token embedding matrix
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Learnable meta-tokens (M in R^{n_meta x d_model})
        if n_meta_tokens > 0:
            self.meta_tokens = nn.Parameter(torch.randn(n_meta_tokens, d_model) * 0.02)
        else:
            self.meta_tokens = None

        # Initialize weights with standard normal / Xavier
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def embed_tokens(
        self,
        input_ids: torch.Tensor,
        include_meta_tokens: bool = True,
    ) -> torch.Tensor:
        """
        Embeds input token IDs and optionally prepends learnable meta-token vectors.
        
        Args:
            input_ids: Tensor of shape (B, T)
            include_meta_tokens: If True, prepends n_meta_tokens to the sequence -> (B, n_meta + T, d_model)
        """
        token_emb = self.embedding(input_ids)  # (B, T, d_model)
        
        if include_meta_tokens and self.meta_tokens is not None:
            batch_size = input_ids.shape[0]
            # Expand meta tokens across batch
            meta_expanded = self.meta_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # (B, n_meta, d_model)
            return torch.cat([meta_expanded, token_emb], dim=1)
            
        return token_emb

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Tied language modeling head projection.
        Computes logits via linear transformation using the transpose of embedding weights.
        """
        return F.linear(hidden_states, self.embedding.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        include_meta_tokens: bool = True,
    ) -> torch.Tensor:
        return self.embed_tokens(input_ids, include_meta_tokens=include_meta_tokens)
