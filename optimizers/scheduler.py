"""
Warmup-Stable-Decay (WSD) Learning Rate Scheduler with Minus-Square-Root Cooldown Annealing.

Schedule:
- Phase 1 (Warmup): Linear ramp from 0 to peak learning rate (eta_peak).
- Phase 2 (Stable): Constant learning rate at eta_peak.
- Phase 3 (Decay): Cooldown annealing: eta(t) = eta_min + (eta_peak - eta_min) * (1 - sqrt((t - t_decay) / (T - t_decay))).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Union

import torch
from optimizers.muon import CombinedMuonAdamW


class WSDScheduler:
    """
    Warmup-Stable-Decay (WSD) learning rate scheduler.
    Supports single optimizers or dual-optimizer wrappers like CombinedMuonAdamW.
    """

    def __init__(
        self,
        optimizer: Union[torch.optim.Optimizer, CombinedMuonAdamW],
        total_steps: int,
        warmup_ratio: float = 0.05,
        stable_ratio: float = 0.75,
        decay_ratio: float = 0.20,
        decay_type: str = "minus_sqrt",  # "minus_sqrt", "cosine", or "linear"
        min_lr_ratio: float = 0.01,
    ):
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.stable_steps = int(total_steps * stable_ratio)
        self.decay_start_step = self.warmup_steps + self.stable_steps
        self.decay_steps = total_steps - self.decay_start_step
        self.decay_type = decay_type
        self.min_lr_ratio = min_lr_ratio

        self.current_step = 0

        # Store base learning rates for all parameter groups
        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]

    def get_lr_factor(self, step: int) -> float:
        """Computes the multiplicative scaling factor in [min_lr_ratio, 1.0] for step."""
        if step < self.warmup_steps:
            # Linear warmup
            return float(step + 1) / float(max(1, self.warmup_steps))
        elif step < self.decay_start_step:
            # Stable plateau
            return 1.0
        else:
            # Cooldown decay phase
            decay_progress = float(step - self.decay_start_step) / float(max(1, self.decay_steps))
            decay_progress = min(max(decay_progress, 0.0), 1.0)

            if self.decay_type == "minus_sqrt":
                # 1 - sqrt(t)
                factor = 1.0 - math.sqrt(decay_progress)
            elif self.decay_type == "cosine":
                # 0.5 * (1 + cos(pi * t))
                factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            else:
                # Linear
                factor = 1.0 - decay_progress

            # Clamp between min_lr_ratio and 1.0
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * factor

    def step(self, step: Optional[int] = None) -> List[float]:
        """Advances the scheduler by one step and updates optimizer parameter groups."""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        factor = self.get_lr_factor(self.current_step)
        current_lrs = []

        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            new_lr = base_lr * factor
            group["lr"] = new_lr
            current_lrs.append(new_lr)

        return current_lrs

    def state_dict(self) -> Dict[str, Any]:
        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "decay_start_step": self.decay_start_step,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.current_step = state_dict.get("current_step", 0)
        self.total_steps = state_dict.get("total_steps", self.total_steps)
        self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
        self.decay_start_step = state_dict.get("decay_start_step", self.decay_start_step)
        self.base_lrs = state_dict.get("base_lrs", self.base_lrs)
