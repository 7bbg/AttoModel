"""Tests for Curriculum Data Mixer, Dataset Preparation, and Binary Memmap Streaming."""

import json
import os
import pytest
import torch

from data.dataset import BinaryMemmapDataset, PackedSequenceDataset, write_tokens_to_binary
from data.filters import MinHashLSHDeduplicator, QualityFilter
from data.synthetic_pipeline import (
    ASTJSONTraceGenerator,
    AgenticTraceGenerator,
    CodeASTGenerator,
    CurriculumDataMixer,
    EducationalCorpusGenerator,
    FormalLogicMathGenerator,
    RLVRPromptGenerator,
)
from data.tokenizer import get_tokenizer


def test_data_mix_config_loading():
    config = CurriculumDataMixer.load_config("configs/data_mix.yaml")
    assert "curriculum" in config
    assert "phases" in config["curriculum"]
    phases = config["curriculum"]["phases"]
    assert len(phases) == 3

    # Check phase shares sum to 1.0
    total_share = sum(p["share"] for p in phases)
    assert abs(total_share - 1.0) < 1e-5


def test_curriculum_document_generation():
    docs = CurriculumDataMixer.generate_curriculum_documents(total_samples=30, config_path="configs/data_mix.yaml")
    assert len(docs) == 30
    for doc in docs:
        assert isinstance(doc, str)
        assert len(doc) > 20


def test_quality_filter_and_educational_score():
    edu_text = (
        "# Chapter 1: Binary Search Trees\n\n"
        "## Formal Definition\n"
        "A binary search tree satisfies the search tree invariant. Therefore, all keys in the left "
        "subtree are strictly less than the root key. For example, consider algorithm traversal."
    )
    assert QualityFilter.passes_heuristics(edu_text) is True
    score = QualityFilter.calculate_educational_score(edu_text)
    assert score >= 3.0


def test_minhash_deduplication():
    dedup = MinHashLSHDeduplicator()
    text1 = "Mathematics is the foundation of scientific rigor. Theorem: The set of primes is infinite."
    text2 = "Mathematics is the foundation of scientific rigor. Theorem: The set of primes is infinite."
    text3 = "Computer science explores algorithms, computational complexity, and dynamic memory models."

    assert dedup.is_duplicate(0, text1) is False
    assert dedup.is_duplicate(1, text2) is True
    assert dedup.is_duplicate(2, text3) is False


def test_rlvr_prompt_generation():
    prompts = RLVRPromptGenerator.generate_rlvr_prompts(count=20)
    assert len(prompts) == 20
    for p in prompts:
        assert isinstance(p, str)
        assert "User:" in p


def test_binary_dataset_compilation_and_streaming(tmp_path):
    out_bin = str(tmp_path / "test_curriculum.bin")
    stats = CurriculumDataMixer.compile_binary_dataset(
        output_bin_path=out_bin,
        total_samples=15,
        config_path="configs/data_mix.yaml",
        max_seq_len=64,
        apply_dedup=False,
    )
    assert os.path.exists(out_bin)
    assert stats["total_tokens"] > 0
    assert stats["packed_samples"] > 0

    # Stream from binary memmap dataset
    dataset = BinaryMemmapDataset(out_bin, max_seq_len=64)
    assert len(dataset) == stats["packed_samples"]
    batch = next(iter(dataset))
    assert batch["input_ids"].shape == (64,)
    assert batch["labels"].shape == (64,)


def test_huggingface_curriculum_streamer():
    from data.hf_loader import HuggingFaceCurriculumStreamer
    assert HuggingFaceCurriculumStreamer.is_datasets_available() is True
    # Test streaming a small slice across all 3 phases
    docs = HuggingFaceCurriculumStreamer.stream_full_curriculum(total_samples=6)
    assert len(docs) == 6
    for d in docs:
        assert isinstance(d, str)
        assert len(d) > 10

    # Test extracting RLVR prompts
    prompts = HuggingFaceCurriculumStreamer.extract_rlvr_prompts_from_hf(count=6)
    assert len(prompts) >= 1
    for p in prompts:
        assert isinstance(p, str)
        assert "User:" in p

