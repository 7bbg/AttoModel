"""
Compound Training Objectives for AttoModel.

Combines:
- Causal Language Modeling Loss (L_CLM)
- Fill-In-The-Middle Structural Infilling Loss (L_FIM)
- Granular Mixture-of-Experts Load Balancing Auxiliary Loss (L_MoE)
- Differential Attention Orthogonalization Regularizer (L_Diff)

Formula:
    L_Total = L_CLM + 0.25 * L_FIM + 0.01 * L_MoE + 0.005 * L_Diff
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompoundObjective(nn.Module):
    """
    Computes the compound multi-task pre-training loss for the sub-20M micro-agent model.
    """

    def __init__(
        self,
        clm_weight: float = 1.0,
        fim_weight: float = 0.25,
        moe_weight: float = 0.01,
        diff_weight: float = 0.005,
        vocab_size: int = 8192,
    ):
        super().__init__()
        self.clm_weight = clm_weight
        self.fim_weight = fim_weight
        self.moe_weight = moe_weight
        self.diff_weight = diff_weight
        self.vocab_size = vocab_size

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        moe_loss: Optional[torch.Tensor] = None,
        diff_loss: Optional[torch.Tensor] = None,
        is_fim_batch: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes weighted total loss and detailed metric breakdown.

        Args:
            logits: (B, T, vocab_size) prediction logits
            targets: (B, T) ground truth token IDs with -100 for ignored/padding tokens
            moe_loss: Scalar auxiliary load balancing loss
            diff_loss: Scalar differential attention orthogonalization loss
            is_fim_batch: Whether the current batch consists of FIM-transformed samples

        Returns:
            total_loss: Scalar loss tensor for backprop
            loss_dict: Dictionary containing detached metrics for logging
        """
        device = logits.device

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous().view(-1, self.vocab_size)
        shift_targets = targets[:, 1:].contiguous().view(-1)

        # Cross-Entropy Token Loss
        ce_loss = F.cross_entropy(
            shift_logits,
            shift_targets,
            ignore_index=-100,
        )

        l_clm = ce_loss if not is_fim_batch else torch.tensor(0.0, device=device)
        l_fim = ce_loss if is_fim_batch else torch.tensor(0.0, device=device)
        l_moe = moe_loss if moe_loss is not None else torch.tensor(0.0, device=device)
        l_diff = diff_loss if diff_loss is not None else torch.tensor(0.0, device=device)

        # Weighted combination
        total_loss = (
            self.clm_weight * l_clm
            + self.fim_weight * l_fim
            + self.moe_weight * l_moe
            + self.diff_weight * l_diff
        )

        loss_dict = {
            "loss_total": total_loss.detach(),
            "loss_clm": l_clm.detach(),
            "loss_fim": l_fim.detach(),
            "loss_moe": l_moe.detach(),
            "loss_diff": l_diff.detach(),
            "token_perplexity": torch.exp(ce_loss.detach().clamp(max=20.0)),
        }

        return total_loss, loss_dict
