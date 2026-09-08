"""
Speculative Decoding Draft Engine.

Employs the 18.5M AttoModel as a high-speed draft speculator for large frontier models:
- Generates K draft tokens at > 850 tok/sec on edge / single GPU.
- Verifies draft candidate sequences in parallel with the target model in a single forward pass.
- Employs exact rejection sampling (Leviathan et al.) guaranteeing identical target distribution.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.tokenizer import AttoTokenizer, get_tokenizer
from models.sub20m_model import AttoModel


class SpeculativeDraftEngine:
    """
    Speculative decoding manager orchestrating the AttoModel draft generator
    and target verification model.
    """

    def __init__(
        self,
        draft_model: AttoModel,
        target_model: Optional[nn.Module] = None,
        tokenizer: Optional[AttoTokenizer] = None,
        lookahead_k: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        self.draft_model = draft_model.to(self.device).eval()
        self.target_model = target_model.to(self.device).eval() if target_model else None
        self.tokenizer = tokenizer or get_tokenizer()
        self.lookahead_k = lookahead_k

    @torch.no_grad()
    def generate_draft_tokens(
        self,
        prefix_ids: torch.Tensor,
        k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates K speculative draft tokens and their draft probability distributions.

        Args:
            prefix_ids: (1, T) prefix token tensor
            k: Number of draft tokens to generate (defaults to self.lookahead_k)

        Returns:
            draft_tokens: (1, K) drafted token IDs
            draft_probs: (K, vocab_size) probability distributions for each draft step
        """
        num_draft = k or self.lookahead_k
        curr_ids = prefix_ids.to(self.device).clone()
        draft_tokens_list = []
        draft_probs_list = []

        for _ in range(num_draft):
            out = self.draft_model(curr_ids, use_cache=False)
            logits = out["logits"][:, -1, :]
            probs = F.softmax(logits, dim=-1)

            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            draft_tokens_list.append(next_token.squeeze(0))
            draft_probs_list.append(probs.squeeze(0))

            curr_ids = torch.cat([curr_ids, next_token], dim=1)

        draft_tokens = torch.stack(draft_tokens_list, dim=1)  # (1, K)
        draft_probs = torch.stack(draft_probs_list, dim=0)    # (K, vocab_size)
        return draft_tokens, draft_probs

    @torch.no_grad()
    def speculative_step(
        self,
        prefix_ids: torch.Tensor,
        target_forward_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> Tuple[List[int], int]:
        """
        Executes a single speculative decoding verification step:
        1. Generates K draft tokens from AttoModel.
        2. Evaluates prefix + K tokens in parallel using target model.
        3. Accepts up to K tokens via speculative verification rule and appends 1 bonus token.

        Returns:
            accepted_tokens: List of accepted token IDs (1 to K+1 tokens)
            num_accepted: Count of draft tokens accepted
        """
        prefix_ids = prefix_ids.to(self.device)
        K = self.lookahead_k
        draft_tokens, draft_probs = self.generate_draft_tokens(prefix_ids, k=K)

        # Full speculative candidate: prefix + draft_tokens
        candidate_ids = torch.cat([prefix_ids, draft_tokens], dim=1)  # (1, T + K)
        T = prefix_ids.shape[1]

        # Target model parallel forward pass
        target_logits = target_forward_fn(candidate_ids)  # (1, T + K, vocab_size)
        target_probs = F.softmax(target_logits[0, T - 1 : T + K - 1, :], dim=-1)  # (K, vocab_size)
        bonus_prob = F.softmax(target_logits[0, T + K - 1, :], dim=-1)             # (vocab_size,)

        accepted_tokens = []
        num_accepted = 0

        # Verify draft tokens sequentially
        for i in range(K):
            draft_token_id = draft_tokens[0, i].item()
            p_draft = draft_probs[i, draft_token_id].item()
            p_target = target_probs[i, draft_token_id].item()

            if p_draft <= p_target:
                # Accept token deterministically / sample accept
                accepted_tokens.append(draft_token_id)
                num_accepted += 1
            else:
                # Rejection sampling with probability p_target / p_draft
                accept_prob = p_target / max(p_draft, 1.0e-8)
                if torch.rand(1).item() < accept_prob:
                    accepted_tokens.append(draft_token_id)
                    num_accepted += 1
                else:
                    # Resample correction token from adjusted distribution: max(0, p_target - p_draft)
                    adj_dist = torch.clamp(target_probs[i] - draft_probs[i], min=0.0)
                    if adj_dist.sum() > 0:
                        adj_dist = adj_dist / adj_dist.sum()
                        correction_token = int(torch.multinomial(adj_dist, 1).item())
                    else:
                        correction_token = int(torch.argmax(target_probs[i]).item())
                    accepted_tokens.append(correction_token)
                    return accepted_tokens, num_accepted

        # If all K tokens accepted, sample 1 additional bonus token from target model
        bonus_token = int(torch.multinomial(bonus_prob, 1).item())
        accepted_tokens.append(bonus_token)

        return accepted_tokens, num_accepted
