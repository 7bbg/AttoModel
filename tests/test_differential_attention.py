"""Tests for Differential Attention Branch."""

import pytest
import torch

from models.attention_branch import DifferentialAttention


def test_differential_attention_forward():
    attn = DifferentialAttention(
        d_in=256,
        d_out=256,
        n_heads=4,
        d_head=64,
        layer_idx=0,
        lambda_init=0.8,
    )

    B, T, D = 2, 12, 256
    x = torch.randn(B, T, D)

    out, produced_kv, next_cache, attn_info = attn(x, return_attn_maps=True)

    assert out.shape == (B, T, D)
    assert len(produced_kv) == 3
    assert attn_info is not None
    assert "a1" in attn_info
    assert "a2" in attn_info
    assert "lambda" in attn_info
    assert attn_info["a1"].shape == (B, 4, T, T)


def test_differential_attention_step_cache():
    # Test MLA compressive KV caching (< 1.2 MB footprint)
    attn_mla = DifferentialAttention(d_in=256, d_out=256, n_heads=4, d_head=64, layer_idx=0, use_mla=True, d_latent_kv=24)
    x_pre = torch.randn(1, 4, 256)
    out_pre, _, kv_cache_mla, _ = attn_mla(x_pre, use_cache=True)
    assert kv_cache_mla["c_kv"].shape == (1, 4, 24)

    x_step = torch.randn(1, 1, 256)
    out_step, _, new_kv_cache_mla, _ = attn_mla(x_step, kv_cache=kv_cache_mla, use_cache=True)
    assert out_step.shape == (1, 1, 256)
    assert new_kv_cache_mla["c_kv"].shape == (1, 5, 24)

    # Test standard uncompressed KV caching
    attn_std = DifferentialAttention(d_in=256, d_out=256, n_heads=4, d_head=64, layer_idx=0, use_mla=False)
    out_pre2, _, kv_cache_std, _ = attn_std(x_pre, use_cache=True)
    assert kv_cache_std["k1"].shape == (1, 4, 4, 64)

    out_step2, _, new_kv_cache_std, _ = attn_std(x_step, kv_cache=kv_cache_std, use_cache=True)
    assert out_step2.shape == (1, 1, 256)
    assert new_kv_cache_std["k1"].shape == (1, 4, 5, 64)
