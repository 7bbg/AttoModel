"""
Group Relative Policy Optimization (GRPO) with Verifiable Rewards (RLVR).

Aligns the sub-20M micro-agent model on deterministic code execution and tool-use tasks:
- Samples G = 8 candidate completions per prompt q: {o_1, o_2, ..., o_G}
- Computes group-normalized advantages: A_i = (R_i - mean(R)) / (std(R) + eps)
- Verifiable Rewards (RLVR):
    +1.0 for valid JSON / Python syntax
    +2.0 for successful tool call format and valid parameters
    -1.5 for malformed control tags or broken schemas
    +1.0 for self-correction / reflection loops
- PPO-clip / GRPO objective with KL penalty against reference policy.
"""

from __future__ import annotations

import ast
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.tokenizer import AttoTokenizer, get_tokenizer
from models.sub20m_model import AttoModel


class VerifiableRewardEvaluator:
    """
    Deterministic rule-based reward functions for agentic execution and syntax verification.
    """

    def __init__(self, tokenizer: Optional[AttoTokenizer] = None):
        self.tokenizer = tokenizer or get_tokenizer()

    def evaluate_completion(self, prompt: str, completion: str) -> Dict[str, float]:
        """
        Evaluates a single sampled completion and returns fine-grained reward components.
        """
        rewards = {
            "syntax_reward": 0.0,
            "tool_reward": 0.0,
            "format_penalty": 0.0,
            "reflection_reward": 0.0,
            "total_reward": 0.0,
        }

        # 1. Tag matching & structural format verification
        # Check that control tags are properly opened and closed
        thought_starts = completion.count("<|thought start|>")
        thought_ends = completion.count("<|thought end|>")
        if thought_starts != thought_ends:
            rewards["format_penalty"] -= 1.5

        # 2. Python syntax validity check
        # Search for ```python ... ``` blocks
        py_blocks = re.findall(r"```python\s*(.*?)\s*```", completion, re.DOTALL)
        for code in py_blocks:
            try:
                ast.parse(code)
                rewards["syntax_reward"] += 1.0
            except SyntaxError:
                rewards["format_penalty"] -= 1.0

        # 3. JSON validity and Tool Call syntax check
        # Search for <|call tool|> ... blocks or json blocks
        if "<|call tool|>" in completion:
            tool_sections = completion.split("<|call tool|>")[1:]
            for sec in tool_sections:
                sec_clean = sec.strip()
                start_brace = sec_clean.find("{")
                if start_brace != -1:
                    try:
                        decoder = json.JSONDecoder()
                        parsed, _ = decoder.raw_decode(sec_clean[start_brace:])
                        if isinstance(parsed, dict) and ("tool" in parsed or "name" in parsed):
                            rewards["tool_reward"] += 2.0
                        else:
                            rewards["syntax_reward"] += 0.5
                    except json.JSONDecodeError:
                        rewards["format_penalty"] -= 1.5
                else:
                    rewards["format_penalty"] -= 1.5

        # 4. Reflection token / self-correction reward
        if "<|reflect|>" in completion and len(completion.split("<|reflect|>")[-1].strip()) > 5:
            rewards["reflection_reward"] += 1.0

        rewards["total_reward"] = (
            rewards["syntax_reward"]
            + rewards["tool_reward"]
            + rewards["format_penalty"]
            + rewards["reflection_reward"]
        )

        return rewards


class GRPOTrainer:
    """
    GRPO RLVR Policy Trainer.
    """

    def __init__(
        self,
        policy_model: AttoModel,
        ref_model: Optional[AttoModel] = None,
        tokenizer: Optional[AttoTokenizer] = None,
        group_size: int = 8,
        kl_beta: float = 0.04,
        clip_eps: float = 0.2,
        lr: float = 5.0e-5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        self.policy_model = policy_model.to(self.device)
        self.ref_model = ref_model.to(self.device) if ref_model else None
        if self.ref_model:
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False

        self.tokenizer = tokenizer or get_tokenizer()
        self.group_size = group_size
        self.kl_beta = kl_beta
        self.clip_eps = clip_eps
        self.evaluator = VerifiableRewardEvaluator(self.tokenizer)
        self.optimizer = torch.optim.AdamW(self.policy_model.parameters(), lr=lr)

    def compute_group_advantages(self, rewards: List[float]) -> torch.Tensor:
        """
        Computes normalized group advantages: A_i = (R_i - mean(R)) / (std(R) + 1e-8).
        """
        r_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        mean = r_tensor.mean()
        std = r_tensor.std(unbiased=False)
        advantages = (r_tensor - mean) / (std + 1.0e-8)
        return advantages

    def sample_group_completions(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
    ) -> Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]:
        """
        Samples G = 8 candidate completions for a given prompt using the policy model.
        """
        self.policy_model.eval()
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        completions: List[str] = []
        completion_token_tensors: List[torch.Tensor] = []
        action_logprobs_list: List[torch.Tensor] = []

        with torch.no_grad():
            for _ in range(self.group_size):
                curr_ids = prompt_tensor.clone()
                gen_ids = []
                action_logprobs = []

                for _ in range(max_new_tokens):
                    out = self.policy_model(curr_ids, use_cache=False)
                    logits = out["logits"][:, -1, :] / temperature
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    token_id = next_token.item()

                    log_prob = F.log_softmax(logits, dim=-1)[0, token_id]
                    action_logprobs.append(log_prob)

                    gen_ids.append(token_id)
                    curr_ids = torch.cat([curr_ids, next_token], dim=1)

                    if token_id == self.tokenizer.eos_token_id:
                        break

                comp_text = self.tokenizer.decode(gen_ids)
                completions.append(comp_text)
                completion_token_tensors.append(torch.tensor(gen_ids, device=self.device))
                action_logprobs_list.append(torch.stack(action_logprobs))

        return completions, completion_token_tensors, action_logprobs_list

    def step(self, prompt: str) -> Dict[str, Any]:
        """
        Executes a single GRPO policy update step for a prompt.
        """
        self.policy_model.train()

        # 1. Sample G = 8 completions
        completions, comp_tokens, old_logprobs = self.sample_group_completions(prompt)

        # 2. Evaluate verifiable rewards
        rewards = []
        reward_breakdowns = []
        for comp in completions:
            res = self.evaluator.evaluate_completion(prompt, comp)
            rewards.append(res["total_reward"])
            reward_breakdowns.append(res)

        # 3. Compute group normalized advantages
        advantages = self.compute_group_advantages(rewards)

        # 4. Compute policy gradient loss across the sampled group
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_len = len(prompt_ids)

        total_loss = torch.tensor(0.0, device=self.device)

        self.optimizer.zero_grad()
        for i in range(self.group_size):
            tokens = comp_tokens[i]
            if len(tokens) == 0:
                continue

            # Full sequence input
            full_ids = torch.cat([
                torch.tensor(prompt_ids, device=self.device),
                tokens,
            ]).unsqueeze(0)

            out = self.policy_model(full_ids)
            logits = out["logits"][0, prompt_len - 1 : -1, :]  # Logits corresponding to generation steps
            new_logprobs = F.log_softmax(logits, dim=-1)
            target_tokens = tokens.unsqueeze(-1)
            selected_new_logprobs = torch.gather(new_logprobs, dim=-1, index=target_tokens).squeeze(-1)

            # Ratio: pi_theta / pi_old
            ratio = torch.exp(selected_new_logprobs - old_logprobs[i].detach())
            adv = advantages[i]

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # KL divergence penalty against ref model if provided
            kl_loss = torch.tensor(0.0, device=self.device)
            if self.ref_model is not None:
                with torch.no_grad():
                    ref_out = self.ref_model(full_ids)
                    ref_logits = ref_out["logits"][0, prompt_len - 1 : -1, :]
                    ref_logprobs = F.log_softmax(ref_logits, dim=-1)
                    ref_selected = torch.gather(ref_logprobs, dim=-1, index=target_tokens).squeeze(-1)
                kl = (torch.exp(ref_selected) * (ref_selected - selected_new_logprobs)).mean()
                kl_loss = self.kl_beta * kl

            sample_loss = (policy_loss + kl_loss) / self.group_size
            sample_loss.backward()
            total_loss += sample_loss.detach()

        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "mean_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "reward_breakdowns": reward_breakdowns,
        }


def run_grpo_alignment(
    checkpoint_path: Optional[str] = None,
    prompts_file: Optional[str] = None,
    output_dir: str = "./checkpoints/grpo_aligned",
    group_size: int = 8,
    steps: int = 100,
    lr: float = 5.0e-5,
    kl_beta: float = 0.04,
    clip_eps: float = 0.2,
    device: Optional[str] = None,
) -> str:
    """Executes the complete GRPO policy alignment workflow and saves aligned checkpoint."""
    import copy
    import glob
    from data.synthetic_pipeline import AgenticTraceGenerator, RLVRPromptGenerator
    from models.sub20m_model import AttoConfig, AttoModel

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Resolve starting checkpoint
    resolved_ckpt = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        resolved_ckpt = checkpoint_path
    else:
        # Check standard locations
        candidates = [
            "./checkpoints/pretrain_final.pt",
            "./checkpoints/ckpt_step_0009537.pt",
        ] + sorted(glob.glob("./checkpoints/ckpt_step_*.pt"), reverse=True)
        for c in candidates:
            if os.path.exists(c):
                resolved_ckpt = c
                break

    if resolved_ckpt:
        print(f"[*] Loading base model from checkpoint: '{resolved_ckpt}'")
        state = torch.load(resolved_ckpt, map_location="cpu")
        config = AttoConfig.from_dict(state["config"]) if "config" in state else AttoConfig()
        policy_model = AttoModel(config)
        policy_model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    else:
        print("[!] No pretrained checkpoint found. Initializing fresh model for GRPO demo.")
        config = AttoConfig()
        policy_model = AttoModel(config)

    # Clone reference model for KL constraint
    ref_model = copy.deepcopy(policy_model)

    trainer = GRPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        group_size=group_size,
        kl_beta=kl_beta,
        clip_eps=clip_eps,
        lr=lr,
        device=target_device,
    )

    # Resolve prompts dataset
    prompts: List[str] = []
    target_prompts_file = prompts_file or ("data/rlvr_prompts.json" if os.path.exists("data/rlvr_prompts.json") else None)
    if target_prompts_file and os.path.exists(target_prompts_file):
        try:
            with open(target_prompts_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list) and len(loaded) > 0:
                    prompts = loaded
                    print(f"[*] Loaded {len(prompts)} verifiable RLVR prompts from '{target_prompts_file}'")
        except Exception as e:
            print(f"[!] Warning: failed to parse '{target_prompts_file}': {e}")

    if not prompts:
        # Generate diverse verifiable prompts
        prompts = RLVRPromptGenerator.generate_rlvr_prompts(count=max(steps, 50))
        print(f"[*] Generated {len(prompts)} diverse verifiable RLVR prompts from RLVRPromptGenerator.")

    print("=" * 70)
    print(" ATTO-MODEL (18.5M) GRPO + RLVR POLICY ALIGNMENT")
    print(f" Device: {target_device} | Group Size: G = {group_size} | Steps: {steps} | LR: {lr:.2e}")
    print(f" Output Dir: {output_dir}")
    print("=" * 70)

    for step_idx in range(1, steps + 1):
        prompt = prompts[(step_idx - 1) % len(prompts)]
        res = trainer.step(prompt)

        if step_idx % 5 == 0 or step_idx == 1:
            print(
                f"Step {step_idx:03d}/{steps:03d} | "
                f"Loss: {res['loss']:.4f} | "
                f"Mean Reward: {res['mean_reward']:+.2f} | "
                f"Max Reward: {res['max_reward']:+.2f} | "
                f"Min Reward: {res['min_reward']:+.2f}"
            )

        if step_idx % 25 == 0:
            step_ckpt = os.path.join(output_dir, f"grpo_step_{step_idx:04d}.pt")
            torch.save({
                "step": step_idx,
                "model_state_dict": trainer.policy_model.state_dict(),
                "config": trainer.policy_model.config.to_dict(),
                "type": "grpo_aligned",
            }, step_ckpt)

    # Save final aligned model in both output_dir and root ./checkpoints
    final_path = os.path.join(output_dir, "grpo_final.pt")
    root_final_path = os.path.join("./checkpoints", "grpo_final.pt")
    save_payload = {
        "step": steps,
        "model_state_dict": trainer.policy_model.state_dict(),
        "config": trainer.policy_model.config.to_dict(),
        "type": "grpo_aligned",
    }
    torch.save(save_payload, final_path)
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(save_payload, root_final_path)

    # Update metadata
    try:
        with open(os.path.join(output_dir, "checkpoint_meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "latest_checkpoint": final_path,
                "type": "grpo_aligned",
                "steps": steps,
            }, f, indent=2)
    except Exception:
        pass

    print(f"[*] Successfully saved GRPO aligned model to:")
    print(f"    - {final_path}")
    print(f"    - {root_final_path}")
    return final_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AttoModel GRPO RLVR Alignment")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/pretrain_final.pt", help="Path to base pretrained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/grpo_aligned", help="Directory to save aligned checkpoints")
    parser.add_argument("--group_size", type=int, default=8, help="Number of completions per prompt (G)")
    parser.add_argument("--steps", type=int, default=50, help="Number of GRPO policy update steps")
    parser.add_argument("--lr", type=float, default=5.0e-5, help="Learning rate for policy optimizer")
    parser.add_argument("--prompts_file", type=str, default=None, help="Path to JSON file containing RLVR prompts")
    parser.add_argument("--kl_beta", type=float, default=0.04, help="KL penalty coefficient against reference model")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clipping epsilon")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    run_grpo_alignment(
        checkpoint_path=args.checkpoint,
        prompts_file=args.prompts_file,
        output_dir=args.output_dir,
        group_size=args.group_size,
        steps=args.steps,
        lr=args.lr,
        kl_beta=args.kl_beta,
        clip_eps=args.clip_eps,
        device=args.device,
    )


if __name__ == "__main__":
    main()
