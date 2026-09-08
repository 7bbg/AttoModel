"""
Sub-20M Micro-Agent Language Model (AttoModel).

Unified Systems Architecture:
- 18.50M Total Parameters / 11.20M Active Parameters per Token
- 12 Hybrid Transformer/SSM Layers (Parallel Mamba-2 SSD + Differential Attention)
- Granular Sparse MoE (1 Shared Expert + 4 Routed Experts with Top-2 Routing)
- 6 Learnable Meta-Token Attention Sinks
- Tied Input/Output Embeddings (V = 8192, d_model = 512)
- Zero-Memory KV-Cache with Cross-Layer KV Sharing (Layers 0-5 share with Layers 6-11)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embeddings import RMSNorm, TiedEmbedding
from models.hybrid_block import HybridBlock


@dataclass
class AttoConfig:
    """Hyperparameter configuration for AttoModel (18.5M parameter budget)."""

    # Vocabulary and Embeddings
    vocab_size: int = 8192
    d_model: int = 512
    n_layers: int = 12
    max_seq_len: int = 4096
    n_meta_tokens: int = 6
    tie_embeddings: bool = True
    rms_norm_eps: float = 1.0e-5

    # SSM Branch (Mamba-2 SSD)
    d_ssm: int = 256
    d_state: int = 16
    d_conv: int = 4
    expand: int = 1
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1.0e-4

    # Differential Attention Branch
    d_attn: int = 256
    n_heads: int = 4
    d_head: int = 64
    lambda_init: float = 0.8
    share_kv_cross_layer: bool = True
    kv_share_distance: int = 6
    rope_theta: float = 10000.0
    use_rope: bool = True
    use_mla: bool = True
    d_latent_kv: int = 24

    # Granular Sparse MoE
    n_routed_experts: int = 4
    n_shared_experts: int = 1
    top_k: int = 2
    d_ff: int = 72
    router_jitter_noise: float = 0.0
    routed_scaling_factor: float = 1.0

    # Objectives & Loss Weights
    clm_loss_weight: float = 1.0
    fim_loss_weight: float = 0.25
    moe_loss_weight: float = 0.01
    diff_loss_weight: float = 0.005

    @classmethod
    def from_yaml(cls, yaml_path: str) -> AttoConfig:
        """Loads configuration from a YAML file."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        flat_config: Dict[str, Any] = {}
        for section in ["model", "ssm", "attention", "moe", "objectives"]:
            if section in data and isinstance(data[section], dict):
                flat_config.update(data[section])

        # Overwrite with any root-level keys
        for k, v in data.items():
            if not isinstance(v, dict) and k in cls.__dataclass_fields__:
                flat_config[k] = v

        return cls(**{k: v for k, v in flat_config.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AttoConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AttoModel(nn.Module):
    """
    18.5M Parameter Sub-20M Micro-Agent Language Model.
    """

    def __init__(self, config: Optional[AttoConfig] = None):
        super().__init__()
        self.config = config or AttoConfig()

        # Tied Input/Output Embedding with 6 Learnable Meta-Tokens
        self.embeddings = TiedEmbedding(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            n_meta_tokens=self.config.n_meta_tokens,
        )

        # 12 Hybrid Transformer/SSM Layers
        self.layers = nn.ModuleList([
            HybridBlock(
                d_model=self.config.d_model,
                d_ssm=self.config.d_ssm,
                d_state=self.config.d_state,
                d_conv=self.config.d_conv,
                d_attn=self.config.d_attn,
                n_heads=self.config.n_heads,
                d_head=self.config.d_head,
                d_ff=self.config.d_ff,
                n_routed_experts=self.config.n_routed_experts,
                n_shared_experts=self.config.n_shared_experts,
                top_k=self.config.top_k,
                layer_idx=layer_idx,
                lambda_init=self.config.lambda_init,
                share_kv=self.config.share_kv_cross_layer,
                kv_share_distance=self.config.kv_share_distance,
                max_seq_len=self.config.max_seq_len,
                use_rope=self.config.use_rope,
                router_jitter_noise=self.config.router_jitter_noise,
                use_mla=self.config.use_mla,
                d_latent_kv=self.config.d_latent_kv,
            )
            for layer_idx in range(self.config.n_layers)
        ])

        # Final Pre-Head Normalization
        self.final_norm = RMSNorm(self.config.d_model, eps=self.config.rms_norm_eps)

    def count_parameters(self) -> Dict[str, Union[int, float]]:
        """
        Calculates and returns total and active parameter counts across all model submodules.
        """
        total_params = sum(p.numel() for p in self.parameters())
        
        # Calculate active parameters per token
        # Tied Embeddings: V * d_model
        emb_params = self.embeddings.embedding.weight.numel()
        meta_params = self.embeddings.meta_tokens.numel() if self.embeddings.meta_tokens is not None else 0
        norms_params = self.final_norm.weight.numel() + sum(
            l.norm1.weight.numel() + l.norm2.weight.numel() for l in self.layers
        )

        # Active layer parameters:
        # SSM + DiffAttn + 1 Shared Expert + Top-2 Routed Experts
        active_layer_params = 0
        for l in self.layers:
            # SSM
            active_layer_params += sum(p.numel() for p in l.ssm_branch.parameters())
            # Attention
            active_layer_params += sum(p.numel() for p in l.attn_branch.parameters())
            # MoE Shared Expert
            active_layer_params += sum(p.numel() for p in l.moe.shared_expert.parameters())
            # MoE Router Gate
            active_layer_params += sum(p.numel() for p in l.moe.router_gate.parameters())
            # Top-2 Routed Experts (average active params)
            expert_params = sum(p.numel() for p in l.moe.routed_experts[0].parameters())
            active_layer_params += self.config.top_k * expert_params
            # Scales
            active_layer_params += l.ssm_scale.numel() + l.attn_scale.numel()

        active_params = emb_params + meta_params + norms_params + active_layer_params

        return {
            "total_parameters": total_params,
            "total_parameters_M": round(total_params / 1.0e6, 2),
            "active_parameters": active_params,
            "active_parameters_M": round(active_params / 1.0e6, 2),
            "active_reduction_percent": round((1.0 - active_params / total_params) * 100, 2),
            "vocab_size": self.config.vocab_size,
            "d_model": self.config.d_model,
            "n_layers": self.config.n_layers,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Dict[str, torch.Tensor]]] = None,
        ssm_states: Optional[List[torch.Tensor]] = None,
        conv_states: Optional[List[torch.Tensor]] = None,
        attn_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        return_aux_losses: bool = False,
        return_attn_maps: bool = False,
        include_meta_tokens: bool = True,
    ) -> Dict[str, Any]:
        """
        Forward pass for training and inference.

        Args:
            input_ids: (B, T) LongTensor of token IDs
            targets: (B, T) Optional target token IDs for loss calculation
            kv_caches: List of KV cache dicts per layer (for inference)
            ssm_states: List of SSM recurrent states per layer (for inference)
            conv_states: List of Conv1D state buffers per layer (for inference)
            attn_mask: Optional attention mask
            use_cache: If True, tracks and returns incremental states for decoding
            return_aux_losses: If True, computes MoE load balancing & differential losses
            include_meta_tokens: Whether to prepend the 6 learnable meta-tokens

        Returns:
            dict containing:
                - logits: (B, T, vocab_size) or (B, n_meta + T, vocab_size)
                - loss: Total scalar loss (if targets provided)
                - loss_clm: Causal LM loss
                - loss_moe: MoE load balancing auxiliary loss
                - loss_diff: Differential attention regularizer loss
                - new_kv_caches: Updated KV caches (if use_cache)
                - new_ssm_states: Updated SSM states (if use_cache)
                - new_conv_states: Updated conv buffers (if use_cache)
        """
        B, T = input_ids.shape
        n_meta = self.config.n_meta_tokens if include_meta_tokens else 0

        # 1. Embed tokens & prepend learnable meta-tokens
        h = self.embeddings(input_ids, include_meta_tokens=include_meta_tokens)  # (B, n_meta + T, d_model)

        # Cross-layer KV storage: maps owner layer index -> (K1, K2, V)
        produced_kvs: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        new_kv_caches_list: List[Dict[str, torch.Tensor]] = []
        new_ssm_states_list: List[torch.Tensor] = []
        new_conv_states_list: List[torch.Tensor] = []
        layer_aux_list: List[Dict[str, Any]] = []

        # 2. Iterate through 12 Hybrid Blocks
        for idx, layer in enumerate(self.layers):
            layer_kv_cache = kv_caches[idx] if (kv_caches is not None and idx < len(kv_caches)) else None
            layer_ssm_state = ssm_states[idx] if (ssm_states is not None and idx < len(ssm_states)) else None
            layer_conv_state = conv_states[idx] if (conv_states is not None and idx < len(conv_states)) else None

            # Cross-layer KV routing: if this layer is shared, retrieve KV from (layer_idx - kv_share_distance)
            external_kv = None
            if self.config.share_kv_cross_layer and idx >= self.config.kv_share_distance:
                owner_idx = idx - self.config.kv_share_distance
                external_kv = produced_kvs.get(owner_idx)

            h, produced_kv, next_kv, next_ssm, next_conv, aux_info = layer(
                h,
                external_kv=external_kv,
                kv_cache=layer_kv_cache,
                ssm_state=layer_ssm_state,
                conv_state=layer_conv_state,
                attn_mask=attn_mask,
                use_cache=use_cache,
                return_aux_info=return_aux_losses,
                return_attn_maps=return_attn_maps,
            )

            if layer.attn_branch.is_kv_owner:
                produced_kvs[idx] = produced_kv

            if use_cache:
                new_kv_caches_list.append(next_kv)
                new_ssm_states_list.append(next_ssm)
                new_conv_states_list.append(next_conv)

            if return_aux_losses and aux_info is not None:
                layer_aux_list.append(aux_info)

        # 3. Final Normalization
        h_norm = self.final_norm(h)

        # 4. Tied Language Modeling Head Projection
        # Extract token sequence (stripping meta-tokens if needed for generation / target alignment)
        if n_meta > 0:
            token_hidden = h_norm[:, n_meta :, :]  # (B, T, d_model)
        else:
            token_hidden = h_norm

        logits = self.embeddings.compute_logits(token_hidden)  # (B, T, vocab_size)

        output: Dict[str, Any] = {
            "logits": logits,
            "hidden_states": token_hidden,
        }

        if use_cache:
            output["new_kv_caches"] = new_kv_caches_list
            output["new_ssm_states"] = new_ssm_states_list
            output["new_conv_states"] = new_conv_states_list

        # 5. Loss Computations if targets or aux losses requested
        if targets is not None or return_aux_losses:
            loss_clm = torch.tensor(0.0, device=input_ids.device)
            loss_moe = torch.tensor(0.0, device=input_ids.device)
            loss_diff = torch.tensor(0.0, device=input_ids.device)

            if targets is not None:
                # Standard causal cross-entropy (shift logits & targets)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = targets[:, 1:].contiguous()
                loss_clm = F.cross_entropy(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_targets.view(-1),
                    ignore_index=-100,
                )

            if return_aux_losses and layer_aux_list:
                # Accumulate MoE load balancing losses across all layers
                moe_losses = []
                diff_losses = []
                for aux in layer_aux_list:
                    if "routing_info" in aux and aux["routing_info"] and "moe_loss" in aux["routing_info"]:
                        moe_losses.append(aux["routing_info"]["moe_loss"])
                    if "attn_info" in aux and aux["attn_info"] is not None:
                        # Diff regularizer: penalize high similarity between A1 and A2 representations
                        if "diff_loss" in aux["attn_info"]:
                            diff_losses.append(aux["attn_info"]["diff_loss"])
                        elif "a1" in aux["attn_info"] and "a2" in aux["attn_info"]:
                            a1 = aux["attn_info"]["a1"]
                            a2 = aux["attn_info"]["a2"]
                            sim = (a1 * a2).sum(dim=-1).mean()
                            diff_losses.append(sim)

                if moe_losses:
                    loss_moe = torch.stack(moe_losses).mean()
                if diff_losses:
                    loss_diff = torch.stack(diff_losses).mean()

            total_loss = (
                self.config.clm_loss_weight * loss_clm
                + self.config.moe_loss_weight * loss_moe
                + self.config.diff_loss_weight * loss_diff
            )

            output.update({
                "loss": total_loss,
                "loss_clm": loss_clm,
                "loss_moe": loss_moe,
                "loss_diff": loss_diff,
            })

        return output
