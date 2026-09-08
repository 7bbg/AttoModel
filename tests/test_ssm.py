"""Tests for Mamba-2 Chunked SSM Branch."""

import pytest
import torch

from models.ssm_branch import Mamba2SSMBranch


def test_ssm_branch_forward():
    ssm = Mamba2SSMBranch(d_in=256, d_out=256, d_ssm=256, d_state=16, d_conv=4)

    B, T, D = 2, 16, 256
    x = torch.randn(B, T, D)

    out, _, _ = ssm(x)
    assert out.shape == (B, T, D)


def test_ssm_step_state():
    ssm = Mamba2SSMBranch(d_in=256, d_out=256, d_ssm=256, d_state=16, d_conv=4)

    # Initial token step
    x1 = torch.randn(1, 1, 256)
    out1, state1, conv1 = ssm(x1, use_cache=True)
    assert out1.shape == (1, 1, 256)
    assert state1.shape == (1, 256, 16)
    assert conv1.shape == (1, 256, 4)

    # Second token step
    x2 = torch.randn(1, 1, 256)
    out2, state2, conv2 = ssm(x2, ssm_state=state1, conv_state=conv1, use_cache=True)
    assert out2.shape == (1, 1, 256)
    assert state2.shape == (1, 256, 16)
