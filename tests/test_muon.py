"""Tests for Muon Matrix Orthogonalization Optimizer and Combined Dual Optimizer."""

import pytest
import torch
import torch.nn as nn

from optimizers.muon import CombinedMuonAdamW, Muon, create_optimizer, zeropower_via_newtonschulz5


def test_newton_schulz5_orthogonalization():
    # Random 2D matrix
    G = torch.randn(64, 32)
    O = zeropower_via_newtonschulz5(G, steps=5)

    assert O.shape == G.shape
    # O^T @ O should approximate identity of size 32 x 32 within polar decomposition tolerance
    product = torch.matmul(O.transpose(-2, -1), O)
    identity = torch.eye(32)
    diff = (product - identity).abs().max().item()
    assert diff < 0.5, f"Expected approx identity, got max diff {diff}"


def test_combined_optimizer_step():
    linear = nn.Linear(512, 512, bias=True)
    opt = CombinedMuonAdamW(linear, muon_lr=0.02, adamw_lr=1e-3)

    x = torch.randn(2, 512)
    y = linear(x).sum()
    y.backward()

    opt.step()
    opt.zero_grad()
    assert len(opt.param_groups) == 2  # 1 for Muon weight, 1 for AdamW bias
