"""
Parallel Hybrid-Head Block (Mamba-2 SSM + Differential Attention) with Granular Sparse MoE.

Executes continuous state-space sequence modeling and differential attention in parallel,
followed by Granular Sparse Mixture-of-Experts with 1 Shared + 4 Routed Experts (Top-2).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.attention_branch import DifferentialAttention
from models.embeddings import RMSNorm
from models.moe import GranularSparseMoE
from models.ssm_branch import Mamba2SSMBranch


class HybridBlock(nn.Module):
    """
    Parallel Hybrid Layer Block.
    
    Structure:
        x -> RMSNorm_1 -> [Mamba-2 SSD || Diff-Attn] -> Add & Residual
          -> RMSNorm_2 -> Granular Sparse MoE -> Add & Residual
    """

    def __init__(
        self,
        d_model: int = 512,
        d_ssm: int = 256,
        d_state: int = 16,
        d_conv: int = 4,
        d_attn: int = 256,
        n_heads: int = 4,
        d_head: int = 64,
        d_ff: int = 72,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        top_k: int = 2,
        layer_idx: int = 0,
        lambda_init: float = 0.8,
        share_kv: bool = True,
        kv_share_distance: int = 6,
        max_seq_len: int = 4096,
        use_rope: bool = True,
        router_jitter_noise: float = 0.0,
        use_mla: bool = True,
        d_latent_kv: int = 24,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = d_model
        self.d_ssm = d_ssm
        self.d_attn = d_attn

        # Pre-Branch Normalization
        self.norm1 = RMSNorm(d_model)

        # Parallel Branch 1: Mamba-2 SSM (operates on d_ssm channels)
        self.ssm_branch = Mamba2SSMBranch(
            d_in=d_ssm,
            d_out=d_ssm,
            d_ssm=d_ssm,
            d_state=d_state,
            d_conv=d_conv,
            layer_idx=layer_idx,
        )

        # Parallel Branch 2: Differential Attention (operates on d_attn channels)
        self.attn_branch = DifferentialAttention(
            d_in=d_attn,
            d_out=d_attn,
            n_heads=n_heads,
            d_head=d_head,
            layer_idx=layer_idx,
            lambda_init=lambda_init,
            share_kv=share_kv,
            kv_share_distance=kv_share_distance,
            max_seq_len=max_seq_len,
            use_rope=use_rope,
            use_mla=use_mla,
            d_latent_kv=d_latent_kv,
        )

        # Learnable parallel branch fusion weighting
        self.ssm_scale = nn.Parameter(torch.ones(1))
        self.attn_scale = nn.Parameter(torch.ones(1))

        # Pre-MoE Normalization
        self.norm2 = RMSNorm(d_model)

        # Stage 2: Granular Sparse MoE
        self.moe = GranularSparseMoE(
            d_model=d_model,
            d_ff=d_ff,
            n_routed_experts=n_routed_experts,
            n_shared_experts=n_shared_experts,
            top_k=top_k,
            router_jitter_noise=router_jitter_noise,
        )

    def forward(
        self,
        x: torch.Tensor,
        external_kv: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        ssm_state: Optional[torch.Tensor] = None,
        conv_state: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        return_aux_info: bool = False,
        return_attn_maps: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Optional[Dict[str, torch.Tensor]],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[Dict[str, torch.Tensor]],
    ]:
        """
        Forward pass for the Hybrid Block.

        Returns:
            out: (B, T, d_model)
            produced_kv: (K1, K2, V)
            new_kv_cache: dict with updated KV cache
            new_ssm_state: updated SSM recurrent state
            new_conv_state: updated 1D conv buffer
            aux_info: dict containing routing info and differential attention maps
        """
        # Pre-branch RMSNorm
        x_norm1 = self.norm1(x)

        # Split hidden state channels in parallel between SSM and DiffAttn branches
        x_ssm_in = x_norm1[..., : self.d_ssm]
        x_attn_in = x_norm1[..., self.d_ssm : self.d_ssm + self.d_attn]

        # Parallel execution: SSM + DiffAttn
        ssm_out, new_ssm_state, new_conv_state = self.ssm_branch(
            x_ssm_in,
            ssm_state=ssm_state,
            conv_state=conv_state,
            use_cache=use_cache,
        )

        attn_out, produced_kv, new_kv_cache, attn_info = self.attn_branch(
            x_attn_in,
            external_kv=external_kv,
            kv_cache=kv_cache,
            attn_mask=attn_mask,
            use_cache=use_cache,
            return_attn_maps=return_attn_maps,
            return_aux_info=return_aux_info,
        )

        # Concatenate parallel branches along channel dimension and add residual
        x_hybrid = torch.cat([self.ssm_scale * ssm_out, self.attn_scale * attn_out], dim=-1)
        x = x + x_hybrid

        # Pre-MoE RMSNorm and Granular MoE
        x_norm2 = self.norm2(x)
        moe_out, routing_info = self.moe(
            x_norm2,
            return_routing_info=return_aux_info,
        )

        x = x + moe_out

        aux_info = None
        if return_aux_info:
            aux_info = {
                "routing_info": routing_info,
                "attn_info": attn_info,
            }

        return x, produced_kv, new_kv_cache, new_ssm_state, new_conv_state, aux_info
