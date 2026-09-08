"""
Differential Attention Branch with Multi-Head Softmax Differencing & Cross-Layer KV Sharing.

Differential Attention Formula:
    DiffAttn(Q, K, V) = ( Softmax(Q1 @ K1^T / sqrt(d_k)) - lambda * Softmax(Q2 @ K2^T / sqrt(d_k)) ) @ V
where lambda is a learnable scalar parameter initialized to lambda = 0.8.

Cross-Layer KV Sharing:
    Layers 0-5 compute and maintain their own KV caches.
    Layers 6-11 share and reuse KV representations from their corresponding lower layer (layer_idx - 6),
    enabling the < 1.2 MB Zero-Memory KV-Cache footprint at C = 4,096.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embeddings import RMSNorm, RotaryPositionEmbedding


class DifferentialAttention(nn.Module):
    """
    Multi-Head Differential Attention Branch with MLA (Multi-Head Latent Attention) Compressive Head.
    Eliminates background activation noise and head redundancy by computing the difference
    between two softmax attention distributions.
    
    Includes MLA compressive KV head restricting KV-cache memory usage to < 1.2 MB at C = 4,096.
    """

    def __init__(
        self,
        d_in: int = 256,
        d_out: int = 256,
        n_heads: int = 4,
        d_head: int = 64,
        layer_idx: int = 0,
        lambda_init: float = 0.8,
        share_kv: bool = True,
        kv_share_distance: int = 6,
        max_seq_len: int = 4096,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        use_mla: bool = True,
        d_latent_kv: int = 24,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.n_heads = n_heads
        self.d_head = d_head
        self.layer_idx = layer_idx
        self.share_kv = share_kv
        self.kv_share_distance = kv_share_distance
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        self.use_mla = use_mla
        self.d_latent_kv = d_latent_kv

        # Determine if this layer owns its KV projections or reuses from lower layer
        self.is_kv_owner = (not share_kv) or (layer_idx < kv_share_distance)

        # Query projections (Q1 and Q2) -> each of dimension (n_heads * d_head = d_out)
        self.q1_proj = nn.Linear(d_in, n_heads * d_head, bias=False)
        self.q2_proj = nn.Linear(d_in, n_heads * d_head, bias=False)

        # Key and Value projections (instantiated only in KV owner layers)
        if self.is_kv_owner:
            if self.use_mla:
                # MLA Compressive Head: compress input down to d_latent_kv (e.g. 24)
                self.w_dkv = nn.Linear(d_in, d_latent_kv, bias=False)
                self.kv_norm = RMSNorm(d_latent_kv)
                self.w_uk1 = nn.Linear(d_latent_kv, n_heads * d_head, bias=False)
                self.w_uk2 = nn.Linear(d_latent_kv, n_heads * d_head, bias=False)
                self.w_uv = nn.Linear(d_latent_kv, n_heads * d_head, bias=False)
                self.k1_proj = None
                self.k2_proj = None
                self.v_proj = None
            else:
                self.w_dkv = None
                self.kv_norm = None
                self.w_uk1 = None
                self.w_uk2 = None
                self.w_uv = None
                self.k1_proj = nn.Linear(d_in, n_heads * d_head, bias=False)
                self.k2_proj = nn.Linear(d_in, n_heads * d_head, bias=False)
                self.v_proj = nn.Linear(d_in, n_heads * d_head, bias=False)
        else:
            self.w_dkv = None
            self.kv_norm = None
            self.w_uk1 = None
            self.w_uk2 = None
            self.w_uv = None
            self.k1_proj = None
            self.k2_proj = None
            self.v_proj = None

        # Output projection back to d_out
        self.out_proj = nn.Linear(n_heads * d_head, d_out, bias=False)

        # Learnable lambda per head, parameterized as lambda = exp(log_lambda) or bounded sigmoid
        # Initialized such that lambda ≈ lambda_init (0.8)
        # Using per-head parameter for fine-grained differential control
        init_val = math.log(lambda_init / (1.0 - lambda_init + 1e-6)) if lambda_init < 1.0 else 0.8
        self.lambda_param = nn.Parameter(torch.full((n_heads,), init_val, dtype=torch.float32))

        # Sub-layer RMSNorm for query/key stabilization
        self.q_norm = RMSNorm(d_head)
        self.k_norm = RMSNorm(d_head)

        # Rotary Position Embedding
        if use_rope:
            self.rope = RotaryPositionEmbedding(d_head, max_seq_len=max_seq_len, theta=rope_theta)
        else:
            self.rope = None

        self._init_weights()

    def _init_weights(self) -> None:
        std = 0.02 / math.sqrt(2 * max(1, self.layer_idx + 1))
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=std)

    def get_lambda(self) -> torch.Tensor:
        """Returns the active differential cancellation scalar lambda per head."""
        return torch.sigmoid(self.lambda_param).view(1, self.n_heads, 1, 1)

    def forward(
        self,
        x: torch.Tensor,
        external_kv: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        attn_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        return_attn_maps: bool = False,
        return_aux_info: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Optional[Dict[str, torch.Tensor]], Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass for Differential Attention.

        Args:
            x: Input hidden states (B, T, d_model)
            external_kv: Tuple of (K1, K2, V) from paired lower layer if this is a shared layer
            kv_cache: Existing KV cache dict for autoregressive generation
            attn_mask: Optional causal or block attention mask
            use_cache: Whether to update and return KV cache
            return_attn_maps: Whether to return dense attention maps for L_Diff regularizer / tests
            return_aux_info: Whether to return lightweight differential regularizer info without dense maps

        Returns:
            out: Attention branch output (B, T, d_model)
            produced_kv: Tuple (K1, K2, V) produced by this layer (if owner)
            new_kv_cache: Updated KV cache dict (if use_cache)
            attn_info: Optional dictionary containing A1, A2 attention maps or diff_loss regularizer
        """
        B, T, _ = x.shape

        # Compute Q1 and Q2
        q1 = self.q1_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, D)
        q2 = self.q2_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, D)

        # Apply Q RMSNorm
        q1 = self.q_norm(q1)
        q2 = self.q_norm(q2)

        # Obtain K1, K2, V
        new_kv_cache = None
        if self.is_kv_owner:
            if self.use_mla:
                # MLA compressive head: compress hidden states to low-rank latent
                c_kv_current = self.kv_norm(self.w_dkv(x))  # (B, T, d_latent_kv)
                if use_cache:
                    if kv_cache is not None and "c_kv" in kv_cache:
                        full_c_kv = torch.cat([kv_cache["c_kv"], c_kv_current], dim=1)
                        cache_offset = kv_cache["c_kv"].shape[1]
                    else:
                        full_c_kv = c_kv_current
                        cache_offset = 0
                    new_kv_cache = {"c_kv": full_c_kv}
                else:
                    full_c_kv = c_kv_current
                    cache_offset = 0

                S = full_c_kv.shape[1]
                # Reconstruct full K1, K2, V from compressed latent
                k1 = self.w_uk1(full_c_kv).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
                k2 = self.w_uk2(full_c_kv).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
                v = self.w_uv(full_c_kv).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
                k1 = self.k_norm(k1)
                k2 = self.k_norm(k2)

                # Apply RoPE
                if self.rope is not None:
                    q1, _ = self.rope(q1, q1, seq_len=T, offset=cache_offset)
                    q2, _ = self.rope(q2, q2, seq_len=T, offset=cache_offset)
                    _, k1 = self.rope(k1, k1, seq_len=S, offset=0)
                    _, k2 = self.rope(k2, k2, seq_len=S, offset=0)
            else:
                assert self.k1_proj is not None and self.k2_proj is not None and self.v_proj is not None
                k1 = self.k1_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
                k2 = self.k2_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
                v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
                k1 = self.k_norm(k1)
                k2 = self.k_norm(k2)

                cache_offset = kv_cache["k1"].shape[2] if (kv_cache is not None and "k1" in kv_cache) else 0
                if self.rope is not None:
                    q1, k1 = self.rope(q1, k1, seq_len=T, offset=cache_offset)
                    q2, k2 = self.rope(q2, k2, seq_len=T, offset=cache_offset)

                if use_cache:
                    if kv_cache is not None and "k1" in kv_cache:
                        k1 = torch.cat([kv_cache["k1"], k1], dim=2)
                        k2 = torch.cat([kv_cache["k2"], k2], dim=2)
                        v = torch.cat([kv_cache["v"], v], dim=2)
                    new_kv_cache = {"k1": k1, "k2": k2, "v": v}
        else:
            # Reusing shared KV from lower paired layer
            if external_kv is None:
                raise ValueError(
                    f"Layer {self.layer_idx} is configured with cross-layer KV sharing but received no external_kv."
                )
            k1, k2, v = external_kv
            S = k1.shape[2]
            cache_offset = max(0, S - T)
            if self.rope is not None:
                q1, _ = self.rope(q1, q1, seq_len=T, offset=cache_offset)
                q2, _ = self.rope(q2, q2, seq_len=T, offset=cache_offset)

        produced_kv = (k1, k2, v)
        S = k1.shape[2]

        is_causal = (attn_mask is None) and (T > 1) and (T == S)
        lambda_val = self.get_lambda().to(dtype=q1.dtype)

        if not return_attn_maps:
            # High-throughput, memory-efficient differential attention via FlashAttention / SDPA.
            # Avoids O(T^2) materialization of dense attention matrices, eliminating 90+ GB OOM.
            # By linearity of matrix multiplication:
            # (A1 - lambda * A2) @ V == (A1 @ V) - lambda * (A2 @ V)
            attn1 = F.scaled_dot_product_attention(
                q1, k1, v,
                attn_mask=attn_mask,
                is_causal=is_causal,
            )
            attn2 = F.scaled_dot_product_attention(
                q2, k2, v,
                attn_mask=attn_mask,
                is_causal=is_causal,
            )
            attn_out = attn1 - lambda_val * attn2
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)
            out = self.out_proj(attn_out)

            attn_info = None
            if return_aux_info:
                # Efficient cosine similarity regularizer between Q1 and Q2 representations (O(T) memory)
                diff_sim = F.cosine_similarity(q1, q2, dim=-1).clamp(min=0.0).mean()
                attn_info = {"diff_loss": diff_sim, "lambda": lambda_val}

            return out, produced_kv, new_kv_cache, attn_info
        else:
            scale = 1.0 / math.sqrt(self.d_head)

            # Explicit dense matrix multiplication for attention map inspection / testing
            scores1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale  # (B, H, T, S)
            scores2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale  # (B, H, T, S)

            # Causal masking
            if attn_mask is not None:
                scores1 = scores1 + attn_mask
                scores2 = scores2 + attn_mask
            elif is_causal:
                # Standard causal mask
                causal_mask = torch.triu(torch.full((T, S), float("-inf"), device=x.device, dtype=scores1.dtype), diagonal=1)
                scores1 = scores1 + causal_mask.unsqueeze(0).unsqueeze(0)
                scores2 = scores2 + causal_mask.unsqueeze(0).unsqueeze(0)

            # Softmax distributions
            a1 = F.softmax(scores1, dim=-1, dtype=torch.float32).type_as(x)  # (B, H, T, S)
            a2 = F.softmax(scores2, dim=-1, dtype=torch.float32).type_as(x)  # (B, H, T, S)

            # Differential subtraction: A_diff = A1 - lambda * A2
            a_diff = a1 - lambda_val.type_as(a1) * a2  # (B, H, T, S)

            # Attention output: A_diff @ V
            attn_out = torch.matmul(a_diff, v)  # (B, H, T, D)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)

            # Output projection
            out = self.out_proj(attn_out)

            sim = (a1 * a2).sum(dim=-1).mean()
            attn_info = {"a1": a1, "a2": a2, "lambda": lambda_val, "diff_loss": sim}

            return out, produced_kv, new_kv_cache, attn_info
