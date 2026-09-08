"""Tests for GRPO Verifiable Rewards Evaluator and Policy Alignment."""

import pytest
import torch

from training.grpo_rlvr import GRPOTrainer, VerifiableRewardEvaluator


def test_verifiable_rewards_python_and_json():
    evaluator = VerifiableRewardEvaluator()

    # Valid Python and Tool JSON
    valid_text = (
        "<|thought start|>Check calculation<|thought end|>"
        "<|call tool|>\n{\"tool\": \"python_interpreter\", \"parameters\": {\"code\": \"x = 10\\nprint(x)\"}}\n"
        "```python\ndef add(a, b):\n    return a + b\n```\n"
        "<|reflect|>Execution succeeded without errors."
    )
    res = evaluator.evaluate_completion("prompt", valid_text)
    assert res["syntax_reward"] > 0.0
    assert res["tool_reward"] > 0.0
    assert res["reflection_reward"] > 0.0
    assert res["format_penalty"] == 0.0
    assert res["total_reward"] > 0.0


def test_verifiable_rewards_malformed():
    evaluator = VerifiableRewardEvaluator()

    # Broken thought tag and invalid json
    invalid_text = (
        "<|thought start|>Unclosed thought"
        "<|call tool|>\n{broken_json: true\n"
    )
    res = evaluator.evaluate_completion("prompt", invalid_text)
    assert res["format_penalty"] < 0.0
