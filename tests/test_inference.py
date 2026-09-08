"""Tests for Hybrid Inference Engine and Speculative Decoding."""

import pytest
import torch

from inference.engine import GenerationConfig, InferenceEngine
from inference.speculator import SpeculativeDraftEngine
from models.sub20m_model import AttoConfig, AttoModel


def test_inference_engine_generation():
    config = AttoConfig(n_layers=4, max_seq_len=256)
    model = AttoModel(config)
    engine = InferenceEngine(model)

    gen_cfg = GenerationConfig(max_new_tokens=8, temperature=0.0)
    output = engine.generate("Test prompt", gen_config=gen_cfg)
    assert isinstance(output, str)


def test_inference_engine_from_checkpoint(tmp_path):
    config = AttoConfig(n_layers=2, max_seq_len=128)
    model = AttoModel(config)
    ckpt_path = str(tmp_path / "test_model.pt")

    torch.save({
        "step": 100,
        "tokens_seen": 50000,
        "loss": 2.5,
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "type": "pretrained",
    }, ckpt_path)

    engine = InferenceEngine.from_checkpoint(ckpt_path)
    assert engine.is_trained is True
    status = engine.get_model_status()
    assert status["checkpoint_step"] == 100
    assert status["checkpoint_type"] == "pretrained"
    assert status["checkpoint_path"] == ckpt_path

    # Verify KV footprint satisfies < 1.2 MB specification
    kv_stats = engine.calculate_kv_memory_footprint(4096)
    assert kv_stats["kv_cache_MB"] < 1.2
    assert kv_stats["meets_zero_memory_spec"] is True


def test_speculative_draft_engine():
    config = AttoConfig(n_layers=2, max_seq_len=256)
    draft_model = AttoModel(config)
    speculator = SpeculativeDraftEngine(draft_model, lookahead_k=3)

    prefix_ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
    draft_tokens, draft_probs = speculator.generate_draft_tokens(prefix_ids, k=3)

    assert draft_tokens.shape == (1, 3)
    assert draft_probs.shape == (3, config.vocab_size)
