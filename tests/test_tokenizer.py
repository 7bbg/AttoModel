"""Tests for 8k Byte-Level Tiktoken Tokenizer & Special Tokens."""

import pytest

from data.tokenizer import SPECIAL_TOKENS, VOCAB_SIZE, get_tokenizer


def test_tokenizer_vocab_size():
    tok = get_tokenizer()
    assert tok.vocab_size == VOCAB_SIZE
    assert len(tok.special_tokens) == len(SPECIAL_TOKENS)


def test_tokenizer_special_tokens_roundtrip():
    tok = get_tokenizer()
    for token_str, expected_id in tok.special_tokens.items():
        encoded = tok.encode(token_str)
        assert len(encoded) == 1
        assert encoded[0] == expected_id
        decoded = tok.decode(encoded)
        assert decoded == token_str


def test_agentic_format():
    tok = get_tokenizer()
    agent_text = tok.format_agent_turn(
        thought="Calculating prime sum",
        tool_call='{"tool": "calc"}',
        tool_response='{"res": 455}',
        reflection="Verified",
        content="Final answer: 455",
    )
    assert "<|thought start|>" in agent_text
    assert "<|thought end|>" in agent_text
    assert "<|call tool|>" in agent_text
    assert "<|tool response|>" in agent_text
    assert "<|reflect|>" in agent_text
    
    encoded = tok.encode(agent_text)
    decoded = tok.decode(encoded)
    assert decoded == agent_text
