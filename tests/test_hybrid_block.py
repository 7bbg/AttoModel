"""Tests for Parallel Hybrid-Head Block (Mamba-2 SSM + Differential Attention + Granular MoE)."""

import pytest
import torch

from models.hybrid_block import HybridBlock


def test_hybrid_block_forward():
    block = HybridBlock(
        d_model=512,
        d_ssm=256,
        d_attn=256,
        n_heads=4,
        d_head=64,
        d_ff=72,
        n_routed_experts=4,
        n_shared_experts=1,
        top_k=2,
        layer_idx=0,
    )

    B, T, D = 2, 16, 512
    x = torch.randn(B, T, D)

    out, produced_kv, next_kv, next_ssm, next_conv, aux = block(x, return_aux_info=True)

    assert out.shape == (B, T, D)
    assert produced_kv is not None
    assert len(produced_kv) == 3  # (K1, K2, V)
    assert aux is not None
    assert "routing_info" in aux
    assert "attn_info" in aux


def test_hybrid_block_cross_layer_kv():
    owner_layer = HybridBlock(d_model=512, d_ssm=256, d_attn=256, layer_idx=0, share_kv=True, kv_share_distance=6)
    shared_layer = HybridBlock(d_model=512, d_ssm=256, d_attn=256, layer_idx=6, share_kv=True, kv_share_distance=6)

    B, T, D = 2, 8, 512
    x = torch.randn(B, T, D)

    # Owner produces KV
    out0, produced_kv, _, _, _, _ = owner_layer(x)
    assert produced_kv is not None

    # Shared layer consumes external KV
    out6, _, _, _, _, _ = shared_layer(x, external_kv=produced_kv)
    assert out6.shape == (B, T, D)
