"""Tests for Granular Sparse Mixture-of-Experts (MoE)."""

import pytest
import torch

from models.moe import GranularSparseMoE, SwiGLUExpert


def test_swiglu_expert():
    expert = SwiGLUExpert(d_model=512, d_ff=72)
    x = torch.randn(4, 16, 512)
    out = expert(x)
    assert out.shape == (4, 16, 512)


def test_granular_moe_routing():
    moe = GranularSparseMoE(
        d_model=512,
        d_ff=72,
        n_routed_experts=4,
        n_shared_experts=1,
        top_k=2,
    )

    B, T, D = 2, 10, 512
    x = torch.randn(B, T, D)

    out, routing_info = moe(x, return_routing_info=True)

    assert out.shape == (B, T, D)
    assert routing_info is not None
    assert "router_logits" in routing_info
    assert "topk_indices" in routing_info
    assert "moe_loss" in routing_info
    assert routing_info["topk_indices"].shape == (B * T, 2)
    assert routing_info["moe_loss"].item() >= 0.0
