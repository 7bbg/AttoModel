"""Core neural network architecture modules for AttoModel (18.5M parameter Sub-20M Micro-Agent)."""

from models.attention_branch import DifferentialAttention
from models.embeddings import RMSNorm, RotaryPositionEmbedding, TiedEmbedding
from models.hybrid_block import HybridBlock
from models.moe import GranularSparseMoE, SwiGLUExpert
from models.objectives import CompoundObjective
from models.ssm_branch import Mamba2SSMBranch
from models.sub20m_model import AttoConfig, AttoModel

__all__ = [
    "AttoConfig",
    "AttoModel",
    "HybridBlock",
    "DifferentialAttention",
    "Mamba2SSMBranch",
    "GranularSparseMoE",
    "SwiGLUExpert",
    "TiedEmbedding",
    "RMSNorm",
    "RotaryPositionEmbedding",
    "CompoundObjective",
]
