#!/usr/bin/env python3
"""
Dataset Preparation & Binary Token Compilation Script for AttoModel.

Implements the 3-Phase Pretraining Curriculum from configs/data_mix.yaml:
- Phase 1: Syntax (50%): Cosmopedia v2 synthetic textbooks & FineWeb-Edu high-score text
- Phase 2: Logic (34%): Python ASTs / JSON traces (60%) & Math proofs and formal logic (40%)
- Phase 3: Agentic (16%): Tool invocation sequences (60%) & Self-play Chain-of-Thought (40%)

Outputs:
1. High-speed contiguous uint16 binary file for zero-copy streaming via BinaryMemmapDataset.
2. Formatted verifiable prompt suite JSON for post-training GRPO alignment (RLVR).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Optional, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from data.dataset import StreamingSequencePacker, write_tokens_to_binary
from data.filters import MinHashLSHDeduplicator, QualityFilter
from data.hf_loader import HuggingFaceCurriculumStreamer
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


def parse_token_count(val: Optional[Union[int, str]]) -> Optional[int]:
    """Parses string token expressions like '10M', '50M', '1B', '5B' into integer counts."""
    if val is None:
        return None
    if isinstance(val, int):
        return val
    val_str = str(val).strip().upper()
    if val_str.endswith("B"):
        return int(float(val_str[:-1]) * 1_000_000_000)
    elif val_str.endswith("M"):
        return int(float(val_str[:-1]) * 1_000_000)
    elif val_str.endswith("K"):
        return int(float(val_str[:-1]) * 1_000)
    return int(val_str)


def prepare_dataset(
    config_path: str = "configs/data_mix.yaml",
    output_bin: str = "data/curriculum_train.bin",
    rlvr_output: str = "data/rlvr_prompts.json",
    target_tokens: Optional[int] = None,
    num_samples: Optional[int] = None,
    num_rlvr_prompts: int = 100,
    max_seq_len: int = 4096,
    source: str = "huggingface",
    apply_dedup: bool = True,
    apply_filter: bool = True,
) -> None:
    # Determine target budget
    if target_tokens is None and num_samples is None:
        target_tokens = 10_000_000

    target_desc = f"{target_tokens:,} Tokens" if target_tokens else f"{num_samples:,} Documents"

    print("=" * 78)
    print(" ATTO-MODEL: CURRICULUM PRETRAINING & RLVR DATASET PREPARATION")
    print(f" Blueprint: AttoModel_Architecture.pdf (5.0B Token Plan & Section 5.2 RLVR)")
    print(f" Target Budget: {target_desc} | Max Context: {max_seq_len} tokens")
    print(f" Source Mode: {source.upper()} | Binary Output: {output_bin}")
    print(f" Config: {config_path} | RLVR Prompts: {rlvr_output}")
    print("=" * 78)

    os.makedirs(os.path.dirname(os.path.abspath(output_bin)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(rlvr_output)), exist_ok=True)

    tokenizer = get_tokenizer()

    # Load FIM hyperparameters
    cfg = CurriculumDataMixer.load_config(config_path) if os.path.exists(config_path) else {}
    fim_cfg = cfg.get("fim", {})
    fim_rate = fim_cfg.get("rate", 0.50)
    spm_prob = fim_cfg.get("spm_prob", 0.50)

    # 1. Initialize Zero-Memory Streaming Sequence Packer
    packer = StreamingSequencePacker(
        output_bin_path=output_bin,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        fim_rate=fim_rate,
        spm_prob=spm_prob,
        append=False,
    )

    # 2. Select document stream
    use_hf = (source == "huggingface" and HuggingFaceCurriculumStreamer.is_datasets_available())
    if use_hf:
        print("[*] Streaming real Hugging Face datasets specified in AttoModel_Architecture.pdf:")
        print("    - Phase 1 (50%): HuggingFaceTB/cosmopedia-v2 + HuggingFaceFW/fineweb-edu (>4.7)")
        print("    - Phase 2 (34%): flytech/python-codes-25k ASTs + HuggingFaceH4/MATH-500")
        print("    - Phase 3 (16%): glaiveai/glaive-function-calling-v2 + Self-Play CoT")
        doc_stream = HuggingFaceCurriculumStreamer.iter_curriculum_stream(config_path=config_path)
    else:
        print("[*] Streaming synthetic curriculum generator matching data_mix.yaml phase weights...")
        doc_stream = CurriculumDataMixer.iter_synthetic_stream(config_path=config_path)

    # 3. Stream, filter, tokenize, pack, and flush directly to disk
    print("[*] Beginning high-throughput streaming compilation...")
    dedup = MinHashLSHDeduplicator() if apply_dedup else None

    doc_count = 0
    dups_dropped = 0
    filter_dropped = 0
    start_time = time.time()
    last_log_time = time.time()

    try:
        for doc in doc_stream:
            # Heuristic quality filter
            if apply_filter and not QualityFilter.passes_heuristics(doc):
                filter_dropped += 1
                continue

            # MinHash deduplication
            if dedup and dedup.is_duplicate(doc_count, doc):
                dups_dropped += 1
                continue

            doc_count += 1
            packer.add_document(doc)

            now = time.time()
            is_target_reached = (
                (target_tokens and packer.tokens_written >= target_tokens)
                or (num_samples and doc_count >= num_samples)
            )

            if now - last_log_time >= 3.0 or is_target_reached:
                elapsed = now - start_time
                tok_sec = packer.tokens_written / max(elapsed, 1e-4)
                file_mb = os.path.getsize(output_bin) / (1024 * 1024)

                if target_tokens:
                    pct = min(100.0, (packer.tokens_written / target_tokens) * 100.0)
                    print(
                        f"[*] Tokens: {packer.tokens_written:,}/{target_tokens:,} ({pct:.1f}%) | "
                        f"Seqs: {packer.sequences_written:,} | Docs: {doc_count:,} | "
                        f"Size: {file_mb:.2f} MB | Speed: {tok_sec:,.0f} tok/s"
                    )
                else:
                    pct = min(100.0, (doc_count / num_samples) * 100.0) if num_samples else 0.0
                    print(
                        f"[*] Docs: {doc_count:,}/{num_samples:,} ({pct:.1f}%) | "
                        f"Tokens: {packer.tokens_written:,} | Seqs: {packer.sequences_written:,} | "
                        f"Size: {file_mb:.2f} MB | Speed: {tok_sec:,.0f} tok/s"
                    )
                last_log_time = now

            if is_target_reached:
                break
    finally:
        if hasattr(doc_stream, "close"):
            try:
                doc_stream.close()
            except Exception:
                pass

    # Finalize packer (pad and flush remaining tokens)
    total_tokens = packer.finish()
    total_elapsed = time.time() - start_time
    final_mb = os.path.getsize(output_bin) / (1024 * 1024)
    avg_speed = total_tokens / max(total_elapsed, 1e-4)

    print("=" * 78)
    print(f"[*] Pre-Training Dataset Compilation Finished in {total_elapsed:.1f}s")
    print(f"    - Output File: {output_bin}")
    print(f"    - Total Contiguous Tokens: {total_tokens:,}")
    print(f"    - Total Packed Sequences (C={max_seq_len}): {packer.sequences_written:,}")
    print(f"    - Total Documents Processed: {doc_count:,}")
    print(f"    - Duplicates Filtered: {dups_dropped:,} | Low-Quality Dropped: {filter_dropped:,}")
    print(f"    - File Size on Disk: {final_mb:.2f} MB")
    print(f"    - Average Compilation Throughput: {avg_speed:,.0f} tok/sec")
    print("=" * 78)

    # 4. Extract and save verifiable RLVR prompts for GRPO alignment
    if num_rlvr_prompts > 0:
        print(f"[*] Extracting {num_rlvr_prompts} verifiable RLVR prompts...")
        rlvr_prompts = []
        if use_hf:
            try:
                rlvr_prompts = HuggingFaceCurriculumStreamer.extract_rlvr_prompts_from_hf(count=num_rlvr_prompts)
            except Exception as e:
                print(f"[!] Warning: failed extracting RLVR prompts from Hugging Face: {e}")

        if len(rlvr_prompts) < num_rlvr_prompts:
            shortfall = num_rlvr_prompts - len(rlvr_prompts)
            rlvr_prompts.extend(RLVRPromptGenerator.generate_rlvr_prompts(count=shortfall))

        with open(rlvr_output, "w", encoding="utf-8") as f:
            json.dump(rlvr_prompts[:num_rlvr_prompts], f, indent=2)
        print(f"[+] Saved {len(rlvr_prompts[:num_rlvr_prompts])} verifiable RLVR prompts to: {rlvr_output}")

    print("=" * 78)
    print(" READY FOR TRAINING:")
    print(f" 1. Pretrain: python -m training.pretrain --data_bin {output_bin}")
    print(f" 2. RLVR GRPO: python -m training.grpo_rlvr --prompts_file {rlvr_output}")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="AttoModel Curriculum Dataset Preparation")
    parser.add_argument("--config", type=str, default="configs/data_mix.yaml", help="Path to data mix YAML config")
    parser.add_argument("--output_bin", type=str, default="data/curriculum_train.bin", help="Output path for uint16 binary dataset")
    parser.add_argument("--rlvr_output", type=str, default="data/rlvr_prompts.json", help="Output path for RLVR prompt JSON")
    parser.add_argument("--target_tokens", type=str, default=None, help="Target token budget (e.g. 10M, 50M, 100M, 1B, 5B, or integer)")
    parser.add_argument("--num_samples", type=int, default=None, help="Alternative: target number of documents (e.g. 1000, 50000)")
    parser.add_argument("--num_rlvr_prompts", type=int, default=100, help="Number of verifiable RLVR prompts for GRPO alignment")
    parser.add_argument("--max_seq_len", type=int, default=4096, help="Model context sequence length")
    parser.add_argument("--source", type=str, choices=["huggingface", "synthetic"], default="huggingface", help="Data source mode")
    parser.add_argument("--no_dedup", action="store_true", help="Disable MinHash deduplication")
    parser.add_argument("--no_filter", action="store_true", help="Disable quality heuristics")
    args = parser.parse_args()

    parsed_tokens = parse_token_count(args.target_tokens)

    prepare_dataset(
        config_path=args.config,
        output_bin=args.output_bin,
        rlvr_output=args.rlvr_output,
        target_tokens=parsed_tokens,
        num_samples=args.num_samples,
        num_rlvr_prompts=args.num_rlvr_prompts,
        max_seq_len=args.max_seq_len,
        source=args.source,
        apply_dedup=not args.no_dedup,
        apply_filter=not args.no_filter,
    )


if __name__ == "__main__":
    main()
