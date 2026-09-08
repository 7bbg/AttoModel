"""
Granular Sparse Mixture-of-Experts (MoE) with Shared Expert and Top-2 Routing.

Architecture:
    y_FFN = Expert_shared(x) + sum_{i in Top-2} G(x)_i * Expert_i(x)
where:
    G(x) = Softmax(Top-2(x @ W_gate))
    Expert_i(x) is a SwiGLU feed-forward network: (SiLU(x @ W_g) * (x @ W_u)) @ W_d

Auxiliary Load Balancing Loss (L_MoE):
    L_MoE = N_routed * sum_{i=1}^{N_routed} (f_i * P_i)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    """
    SwiGLU Feed-Forward Expert Network.
    Applies gated linear units with SiLU activation:
        out = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
    """

    def __init__(self, d_model: int = 512, d_ff: int = 72):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        std = 0.02 / math.sqrt(2)
        nn.init.normal_(self.w_gate.weight, mean=0.0, std=std)
        nn.init.normal_(self.w_up.weight, mean=0.0, std=std)
        nn.init.normal_(self.w_down.weight, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_model)
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)


class GranularSparseMoE(nn.Module):
    """
    Granular Sparse MoE containing 1 shared expert and 4 routed experts with Top-2 selection.
    
    Total active experts per token: 3 (1 shared + 2 routed), ensuring high parameter efficiency
    and granular specialization.
    """

    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 72,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        top_k: int = 2,
        router_jitter_noise: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.top_k = top_k
        self.router_jitter_noise = router_jitter_noise

        # Router gate projection
        self.router_gate = nn.Linear(d_model, n_routed_experts, bias=False)

        # 1 Shared Expert (always active)
        self.shared_expert = SwiGLUExpert(d_model=d_model, d_ff=d_ff * n_shared_experts)

        # 4 Routed Experts
        self.routed_experts = nn.ModuleList([
            SwiGLUExpert(d_model=d_model, d_ff=d_ff) for _ in range(n_routed_experts)
        ])

        self._init_router()

    def _init_router(self) -> None:
        nn.init.normal_(self.router_gate.weight, mean=0.0, std=0.02)

    def compute_routing_loss(
        self,
        router_logits: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the auxiliary load balancing loss (L_MoE) via switch / GShard formulation:
            L_MoE = N * sum_{i=1}^N (f_i * P_i)
        where:
            f_i = fraction of token dispatches to expert i
            P_i = average routing probability for expert i
        """
        # router_logits: (N_tokens, N_routed)
        # topk_indices: (N_tokens, top_k)
        num_tokens = router_logits.shape[0]
        if num_tokens == 0:
            return torch.tensor(0.0, device=router_logits.device, dtype=router_logits.dtype)

        # Softmax probabilities over all routed experts
        probs = F.softmax(router_logits, dim=-1)  # (N_tokens, N_routed)
        mean_probs = probs.mean(dim=0)            # (N_routed,)

        # Compute dispatch frequencies (f_i)
        # One-hot representation of top-k selections
        # topk_indices: (N_tokens, top_k)
        dispatch_mask = F.one_hot(topk_indices, num_classes=self.n_routed_experts).float()  # (N_tokens, top_k, N_routed)
        dispatch_count = dispatch_mask.sum(dim=[0, 1])  # (N_routed,)
        dispatch_freq = dispatch_count / (num_tokens * self.top_k)

        # Auxiliary loss: N_routed * sum(f_i * P_i)
        aux_loss = self.n_routed_experts * (dispatch_freq * mean_probs).sum()
        return aux_loss

    def forward(
        self,
        x: torch.Tensor,
        return_routing_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass for Granular Sparse MoE.

        Args:
            x: Input hidden state tensor of shape (B, T, d_model)
            return_routing_info: If True, returns router logits and top-k indices for loss computation

        Returns:
            out: Combined FFN output tensor of shape (B, T, d_model)
            routing_info: Optional dictionary containing router logits and aux loss
        """
        orig_shape = x.shape
        B, T, D = orig_shape
        x_flat = x.view(-1, D)  # (N_tokens, D)
        N_tokens = x_flat.shape[0]

        # 1. Compute Shared Expert output (always active for all tokens)
        shared_out = self.shared_expert(x_flat)  # (N_tokens, D)

        # 2. Compute Router Gating Logits
        router_logits = self.router_gate(x_flat)  # (N_tokens, n_routed_experts)

        if self.training and self.router_jitter_noise > 0.0:
            noise = torch.randn_like(router_logits) * self.router_jitter_noise
            router_logits_for_routing = router_logits + noise
        else:
            router_logits_for_routing = router_logits

        # Top-2 expert selection per token
        topk_weights, topk_indices = torch.topk(router_logits_for_routing, k=self.top_k, dim=-1)
        # Softmax over selected top-k weights
        topk_weights = F.softmax(topk_weights, dim=-1)  # (N_tokens, top_k)

        # 3. Compute Routed Experts output
        routed_out = torch.zeros_like(x_flat)

        # Efficient token dispatching by expert
        for expert_idx, expert_module in enumerate(self.routed_experts):
            # Find tokens routed to this expert
            # mask: (N_tokens, top_k)
            expert_mask = (topk_indices == expert_idx)
            if expert_mask.any():
                # Indices of tokens that selected this expert
                token_indices, k_positions = torch.where(expert_mask)
                expert_inputs = x_flat[token_indices]
                expert_outputs = expert_module(expert_inputs)
                
                # Weight by softmax router score
                weights = topk_weights[token_indices, k_positions].unsqueeze(-1)
                weighted_outputs = expert_outputs * weights
                
                # Accumulate into output
                routed_out.index_add_(0, token_indices, weighted_outputs)

        # Total FFN Output: Shared + Routed
        total_out = (shared_out + routed_out).view(B, T, D)

        routing_info = None
        if return_routing_info:
            moe_loss = self.compute_routing_loss(router_logits, topk_indices)
            routing_info = {
                "router_logits": router_logits,
                "topk_indices": topk_indices,
                "moe_loss": moe_loss,
            }

        return total_out, routing_info
