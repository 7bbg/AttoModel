"""
Muon Optimizer: Newton-Schulz Matrix Orthogonalization for 2D Weight Projections.

Implements:
- 5-step Newton-Schulz iterative orthogonalization for 2D weight tensors:
    M_t = mu * M_{t-1} + grad(W_{t-1})
    O_t = NewtonSchulz(M_t)
    W_t = W_{t-1} - eta_t * ( 0.2 * O_t / RMS(O_t) + lambda_wd * W_{t-1} )
- Unified CombinedMuonAdamW optimizer partitioning 2D weights to Muon and 1D/SSM/Norms to AdamW.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1.0e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration (degree 5 quintic polynomial) to compute the approximate
    orthogonalization / polar decomposition of matrix G (UV^T where G = U S V^T).
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype == torch.bfloat16 else G.float()

    # If matrix is tall (m > n), transpose to work on square n x n inside
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.transpose(-2, -1)
        transposed = True

    # Spectral norm approximation via Frobenius normalization
    norm = X.norm(p="fro", dim=(-2, -1), keepdim=True) + eps
    X = X / norm

    for _ in range(steps):
        A = torch.matmul(X, X.transpose(-2, -1))
        B = b * A + c * torch.matmul(A, A)
        X = a * X + torch.matmul(B, X)

    if transposed:
        X = X.transpose(-2, -1)

    return X.type_as(G)


class Muon(Optimizer):
    """
    Muon Optimizer for 2D parameter tensors (Linear weights, Projections).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.01,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.ndim < 2:
                    raise ValueError(f"Muon only supports 2D+ tensors, got shape {p.shape}")

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)

                if nesterov:
                    g = grad + momentum * buf
                else:
                    g = buf

                # 5-step Newton-Schulz orthogonalization
                g_orth = zeropower_via_newtonschulz5(g, steps=ns_steps)

                # Scale by 0.2 / RMS(O_t)
                rms = torch.sqrt(torch.mean(g_orth.pow(2)) + 1.0e-8)
                update = 0.2 * (g_orth / rms)

                if weight_decay > 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                p.data.add_(update, alpha=-lr)

        return loss


class CombinedMuonAdamW:
    """
    Unified dual-optimizer combining Muon for 2D projection matrices and AdamW
    for 1D parameters, embeddings, norms, biases, and SSM matrices.
    """

    def __init__(
        self,
        model: nn.Module,
        muon_lr: float = 0.02,
        muon_momentum: float = 0.95,
        muon_weight_decay: float = 0.01,
        muon_ns_steps: int = 5,
        adamw_lr: float = 1.2e-3,
        adamw_betas: Tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1.0e-8,
        adamw_weight_decay: float = 0.1,
    ):
        self.model = model
        muon_params: List[nn.Parameter] = []
        adamw_params: List[nn.Parameter] = []

        # Partition parameters:
        # 2D weight matrices (linear layers without embeddings/1D) -> Muon
        # Embeddings, 1D vectors (biases, norms, SSM log_A, D, dt_bias, lambda) -> AdamW
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 2 and "embedding" not in name.lower() and "meta_tokens" not in name.lower():
                muon_params.append(p)
            else:
                adamw_params.append(p)

        self.muon_opt = Muon(
            muon_params,
            lr=muon_lr,
            momentum=muon_momentum,
            weight_decay=muon_weight_decay,
            ns_steps=muon_ns_steps,
        ) if muon_params else None

        self.adamw_opt = torch.optim.AdamW(
            adamw_params,
            lr=adamw_lr,
            betas=adamw_betas,
            eps=adamw_eps,
            weight_decay=adamw_weight_decay,
        ) if adamw_params else None

        self.muon_params = muon_params
        self.adamw_params = adamw_params

    @property
    def param_groups(self) -> List[Dict[str, Any]]:
        groups = []
        if self.muon_opt:
            groups.extend(self.muon_opt.param_groups)
        if self.adamw_opt:
            groups.extend(self.adamw_opt.param_groups)
        return groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.muon_opt:
            self.muon_opt.zero_grad(set_to_none=set_to_none)
        if self.adamw_opt:
            self.adamw_opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        if self.muon_opt:
            self.muon_opt.step()
        if self.adamw_opt:
            self.adamw_opt.step()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "muon": self.muon_opt.state_dict() if self.muon_opt else None,
            "adamw": self.adamw_opt.state_dict() if self.adamw_opt else None,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self.muon_opt and state_dict.get("muon"):
            self.muon_opt.load_state_dict(state_dict["muon"])
        if self.adamw_opt and state_dict.get("adamw"):
            self.adamw_opt.load_state_dict(state_dict["adamw"])


def create_optimizer(
    model: nn.Module,
    muon_lr: float = 0.02,
    adamw_lr: float = 1.2e-3,
    muon_weight_decay: float = 0.01,
    adamw_weight_decay: float = 0.1,
) -> CombinedMuonAdamW:
    """Factory creating the unified Muon + AdamW dual optimizer."""
    return CombinedMuonAdamW(
        model=model,
        muon_lr=muon_lr,
        adamw_lr=adamw_lr,
        muon_weight_decay=muon_weight_decay,
        adamw_weight_decay=adamw_weight_decay,
    )
