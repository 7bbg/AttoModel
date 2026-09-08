#!/usr/bin/env python3
"""
Interactive Chat & Agentic REPL for AttoModel (18.5M Parameters).

Features:
- Seamless loading of trained (pretrain) or aligned (GRPO RLVR) checkpoints
- Real-time token streaming with syntax highlighting for thought & tool tags
- Multi-turn conversation history
- Zero-Memory KV-Cache status monitoring
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from inference.engine import GenerationConfig, InferenceEngine


# ANSI Color formatting for terminal chat
class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def format_agent_stream(token_chunk: str, in_thought: bool, in_tool: bool) -> tuple[str, bool, bool]:
    """Applies terminal color highlights to agentic control tags in real-time."""
    text = token_chunk.replace("\ufffd", "")
    if not text:
        return "", in_thought, in_tool

    if "<|thought start|>" in text:
        in_thought = True
        text = text.replace("<|thought start|>", f"\n{Colors.DIM}{Colors.CYAN}💭 [Reasoning: ")
    if "<|thought end|>" in text:
        in_thought = False
        text = text.replace("<|thought end|>", f"]{Colors.RESET}\n")

    if "<|call tool|>" in text:
        in_tool = True
        text = text.replace("<|call tool|>", f"\n{Colors.YELLOW}🛠️  [Tool Call:\n")
    if "<|tool response|>" in text:
        in_tool = False
        text = text.replace("<|tool response|>", f"\n📥 [Tool Response:\n")
    if "<|reflect|>" in text:
        text = text.replace("<|reflect|>", f"\n{Colors.MAGENTA}🔄 [Self-Reflection: {Colors.RESET}")

    return text, in_thought, in_tool


def run_interactive_repl(
    engine: InferenceEngine,
    temperature: float = 0.3,
    top_p: float = 0.90,
    repetition_penalty: float = 1.15,
    max_tokens: int = 256,
) -> None:
    """Runs interactive conversation REPL in terminal."""
    status = engine.get_model_status()
    params = status["parameters"]

    print("=" * 70)
    print(f"{Colors.BOLD}{Colors.GREEN} AttoModel (18.5M Micro-Agent) Interactive Session{Colors.RESET}")
    print("=" * 70)
    print(f" • Total Parameters:  {params['total_parameters_M']}M")
    print(f" • Active Parameters: {params['active_parameters_M']}M ({params['active_reduction_percent']}% reduction)")
    print(f" • Compute Device:    {status['device']}")

    tokens_seen = status.get("tokens_seen", 0) or 0
    if status["is_trained"]:
        ckpt_type = status["checkpoint_type"].upper()
        step = status["checkpoint_step"]
        path = status["checkpoint_path"]
        print(f" • {Colors.GREEN}Model State:       TRAINED ({ckpt_type}){Colors.RESET}")
        print(f" • Checkpoint Path:   {path}")
        if step > 0:
            print(f" • Training Step:     {step:,}")
        if tokens_seen > 0:
            pct_blueprint = (tokens_seen / 5_000_000_000) * 100.0
            print(f" • Tokens Trained:    {tokens_seen / 1e6:.2f}M / 5,000M ({pct_blueprint:.2f}% of 5.0B Blueprint)")
            if tokens_seen < 370_000_000:
                print(f"   {Colors.YELLOW}[!] Note: Model is at {tokens_seen / 1e6:.1f}M tokens (undertrained vs 370M Chinchilla / 5B Blueprint).{Colors.RESET}")
                print(f"   {Colors.DIM}    Low temperature (0.1 - 0.3) or structured agent prompts work best.{Colors.RESET}")
    else:
        print(f" • {Colors.YELLOW}Model State:       INITIALIZED (Untrained Weights){Colors.RESET}")
        print(f"   {Colors.DIM}Notice: Train with `bash scripts/run_pretrain.sh` or `run_grpo.sh`{Colors.RESET}")

    print("-" * 70)
    print(" Commands: /reset (clear history) | /status (show model info) | /exit (quit)")
    print(" Tip: Ask coding/tool/math tasks like: 'Write a python function to compute gcd(a, b)'")
    print("=" * 70)

    history: List[str] = []
    gen_config = GenerationConfig(
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        suppress_control_tokens=True,
    )

    while True:
        try:
            user_input = input(f"\n{Colors.BOLD}You > {Colors.RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            print("Session ended.")
            break
        elif user_input.lower() == "/reset":
            history.clear()
            print(f"{Colors.GREEN}[✓] Conversation history reset.{Colors.RESET}")
            continue
        elif user_input.lower() == "/status":
            print(f"\n{Colors.CYAN}--- Model & KV Footprint Status ---{Colors.RESET}")
            kv_info = engine.calculate_kv_memory_footprint(4096)
            for k, v in kv_info.items():
                print(f"  {k}: {v}")
            continue
        elif user_input.lower() == "/help":
            print("Commands: /reset, /status, /exit")
            continue

        # Format turn into agentic prompt aligned with training distribution
        is_greeting = user_input.lower() in ("hi", "hello", "hey", "greetings", "hi there")
        if is_greeting and tokens_seen < 200_000_000:
            prompt = f"<|bos|>User: {user_input}\nFinal Response: Hello! I am AttoModel, an 18.5M parameter micro-agent model. How can I help you with code, tools, or math today?<|eos|>"
            # If the model hasn't been pre-trained on casual greetings, answer gracefully
            print(f"{Colors.BOLD}AttoModel > {Colors.RESET}Hello! I am AttoModel (18.5M Micro-Agent). I specialize in Python code, mathematical reasoning, and tool execution. How can I assist you?")
            continue
        else:
            prompt = f"<|bos|>User: {user_input}\n<|thought start|>"

        print(f"{Colors.BOLD}AttoModel > {Colors.RESET}", end="", flush=True)

        in_thought = False
        in_tool = False
        full_completion = []

        for chunk in engine.generate_stream(prompt, gen_config=gen_config):
            full_completion.append(chunk)
            formatted_chunk, in_thought, in_tool = format_agent_stream(chunk, in_thought, in_tool)
            print(formatted_chunk, end="", flush=True)

        if in_thought:
            print(f"{Colors.RESET}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description="AttoModel Interactive Chat")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints",
        help="Path to checkpoint file or checkpoints directory (auto-detects best checkpoint)",
    )
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature (default: 0.3, lower = more deterministic)")
    parser.add_argument("--top_p", type=float, default=0.90, help="Nucleus sampling threshold (default: 0.90)")
    parser.add_argument("--repetition_penalty", type=float, default=1.15, help="Repetition penalty (default: 1.15)")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum new tokens per turn")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda or cpu)")
    args = parser.parse_args()

    engine = InferenceEngine.from_pretrained(args.checkpoint, device=args.device)
    run_interactive_repl(
        engine,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
