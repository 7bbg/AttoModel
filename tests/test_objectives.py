"""Tests for Compound Training Objectives."""

import pytest
import torch

from models.objectives import CompoundObjective


def test_compound_objective_loss():
    obj = CompoundObjective(
        clm_weight=1.0,
        fim_weight=0.25,
        moe_weight=0.01,
        diff_weight=0.005,
        vocab_size=8192,
    )

    B, T, V = 2, 16, 8192
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    moe_loss = torch.tensor(1.2)
    diff_loss = torch.tensor(0.5)

    total_loss, loss_dict = obj(
        logits=logits,
        targets=targets,
        moe_loss=moe_loss,
        diff_loss=diff_loss,
        is_fim_batch=False,
    )

    assert total_loss.item() > 0.0
    assert "loss_total" in loss_dict
    assert "loss_clm" in loss_dict
    assert "loss_moe" in loss_dict
    assert "loss_diff" in loss_dict
    assert "token_perplexity" in loss_dict
