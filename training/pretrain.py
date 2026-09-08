"""
Multi-Stage Curriculum Pre-Training Engine with Automated Rollbacks and High Throughput.

Designed for the 6-Hour Single RTX PRO 6000 Execution Sprint (5.0B tokens at ~231,500 tok/sec).
Implements:
- 3-phase WSD curriculum management (Syntax -> Logic -> Agentic).
- Micro-batching with gradient accumulation.
- Automated anomaly detection & rollback to healthy checkpoints on loss spikes or NaNs.
- Checkpointing, throughput monitoring, and metric logging.
"""

from __future__ import annotations

import collections
import glob
import json
import math
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs import *
from data.dataset import PackedSequenceDataset
from data.tokenizer import AttoTokenizer, get_tokenizer
from models.sub20m_model import AttoConfig, AttoModel
from optimizers.muon import CombinedMuonAdamW, create_optimizer
from optimizers.scheduler import WSDScheduler
from training.precision import PrecisionManager


class CheckpointManager:
    """
    Manages saving, loading, rolling pruning, and anomaly rollbacks for training checkpoints.
    """

    def __init__(self, output_dir: str = "./checkpoints", keep_last_k: int = 5):
        self.output_dir = output_dir
        self.keep_last_k = keep_last_k
        os.makedirs(self.output_dir, exist_ok=True)
        self.saved_checkpoints: List[str] = []
        self.last_healthy_checkpoint: Optional[str] = None

    def save_checkpoint(
        self,
        model: AttoModel,
        optimizer: CombinedMuonAdamW,
        scheduler: WSDScheduler,
        step: int,
        tokens_seen: int,
        loss: float,
        is_healthy: bool = True,
    ) -> str:
        """Saves a training checkpoint to disk."""
        ckpt_path = os.path.join(self.output_dir, f"ckpt_step_{step:07d}.pt")
        state = {
            "step": step,
            "tokens_seen": tokens_seen,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": model.config.to_dict(),
        }
        torch.save(state, ckpt_path)
        self.saved_checkpoints.append(ckpt_path)

        if is_healthy:
            self.last_healthy_checkpoint = ckpt_path

        # Update metadata JSON
        meta_path = os.path.join(self.output_dir, "checkpoint_meta.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "latest_checkpoint": ckpt_path,
                    "type": "pretrain",
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "loss": loss,
                }, f, indent=2)
        except Exception:
            pass

        # Prune old checkpoints
        if len(self.saved_checkpoints) > self.keep_last_k:
            oldest = self.saved_checkpoints.pop(0)
            if oldest != self.last_healthy_checkpoint and os.path.exists(oldest):
                try:
                    os.remove(oldest)
                except OSError:
                    pass

        return ckpt_path

    def load_checkpoint(
        self,
        ckpt_path: str,
        model: AttoModel,
        optimizer: Optional[CombinedMuonAdamW] = None,
        scheduler: Optional[WSDScheduler] = None,
    ) -> Dict[str, Any]:
        """Loads model and optimizer state from checkpoint."""
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        return state


class PretrainTrainer:
    """
    Curriculum Pre-Training Engine for AttoModel.
    """

    def __init__(
        self,
        model: AttoModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        total_steps: int = 9537,
        grad_accum_steps: int = 16,
        precision: str = "bfloat16",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        muon_lr: float = 0.02,
        adamw_lr: float = 1.2e-3,
        save_every_steps: int = 500,
        checkpoint_dir: str = "./checkpoints",
        spike_threshold: float = 3.5,
        enable_rollback: bool = True,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.total_steps = total_steps
        self.grad_accum_steps = grad_accum_steps
        self.save_every_steps = save_every_steps
        if self.total_steps < self.save_every_steps:
            self.save_every_steps = max(1, self.total_steps // 2)
        self.spike_threshold = spike_threshold
        self.enable_rollback = enable_rollback

        # Optimizers and Schedulers
        self.optimizer = create_optimizer(self.model, muon_lr=muon_lr, adamw_lr=adamw_lr)
        self.scheduler = WSDScheduler(self.optimizer, total_steps=total_steps)
        self.precision_mgr = PrecisionManager(precision=precision, device=device)
        self.ckpt_mgr = CheckpointManager(output_dir=checkpoint_dir)

        # Loss history for anomaly detection
        self.loss_window: collections.deque = collections.deque(maxlen=50)

    def _is_loss_anomalous(self, current_loss: float) -> bool:
        """Checks if current loss has experienced an explosive spike."""
        if math.isnan(current_loss) or math.isinf(current_loss):
            return True
        if len(self.loss_window) >= 10:
            mean = sum(self.loss_window) / len(self.loss_window)
            variance = sum((x - mean) ** 2 for x in self.loss_window) / len(self.loss_window)
            std = math.sqrt(variance)
            if current_loss > mean + self.spike_threshold * max(std, 0.1):
                return True
        return False

    def train(self) -> Dict[str, Any]:
        """
        Executes the main pretraining loop.
        """
        self.model.train()
        step = 0
        tokens_seen = 0
        accum_loss = 0.0
        start_time = time.time()
        step_start_time = time.time()
        interval_tokens = 0
        last_logged_step = 0

        data_iter = iter(self.train_dataloader)

        print(f"[*] Starting Pretraining Run on {self.device}")
        print(f"[*] Total Target Steps: {self.total_steps} | Accumulation: {self.grad_accum_steps}")

        while step < self.total_steps:
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            step_tokens = 0

            # Gradient Accumulation Inner Loop
            for micro_step in range(self.grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_dataloader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                B, T = input_ids.shape
                step_tokens += B * T

                with self.precision_mgr.autocast():
                    out = self.model(
                        input_ids=input_ids,
                        targets=labels,
                        return_aux_losses=True,
                    )
                    loss = out["loss"] / self.grad_accum_steps

                self.precision_mgr.backward(loss)
                accum_loss += loss.item() * self.grad_accum_steps

            # Check for loss anomaly / explosion
            if self._is_loss_anomalous(accum_loss):
                print(f"[!] Loss anomaly detected at step {step}: Loss = {accum_loss:.4f}")
                if self.enable_rollback and self.ckpt_mgr.last_healthy_checkpoint:
                    print(f"[*] Rolling back to checkpoint: {self.ckpt_mgr.last_healthy_checkpoint}")
                    self.ckpt_mgr.load_checkpoint(
                        self.ckpt_mgr.last_healthy_checkpoint,
                        self.model,
                        self.optimizer,
                        self.scheduler,
                    )
                    # Skip step advance after rollback
                    continue

            # Optimizer Step and Scheduler Update
            self.precision_mgr.step(self.optimizer)
            current_lrs = self.scheduler.step()
            self.loss_window.append(accum_loss)

            step += 1
            tokens_seen += step_tokens
            interval_tokens += step_tokens

            # Progress Logging
            if step % 10 == 0 or step == 1 or step == self.total_steps:
                elapsed = time.time() - step_start_time
                tok_sec = interval_tokens / max(elapsed, 1e-4)
                steps_remaining = self.total_steps - step
                steps_done_interval = max(1, step - last_logged_step)
                sec_per_step = elapsed / steps_done_interval
                eta_secs = steps_remaining * sec_per_step
                eta_str = f"{eta_secs / 60:.1f}m" if eta_secs < 3600 else f"{eta_secs / 3600:.1f}h"
                pct = (step / self.total_steps) * 100.0
                print(
                    f"Step {step:05d}/{self.total_steps:05d} ({pct:.1f}%) | "
                    f"Loss: {accum_loss:.4f} | "
                    f"LR: {current_lrs[0]:.2e} | "
                    f"Throughput: {tok_sec:,.0f} tok/s | "
                    f"Tokens: {tokens_seen / 1e6:.2f}M | "
                    f"ETA: {eta_str}"
                )
                step_start_time = time.time()
                interval_tokens = 0
                last_logged_step = step

            # Save Checkpoint
            if step % self.save_every_steps == 0 or step == self.total_steps:
                self.ckpt_mgr.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    step=step,
                    tokens_seen=tokens_seen,
                    loss=accum_loss,
                    is_healthy=True,
                )

        # Save final pretraining checkpoint
        final_ckpt = os.path.join(self.ckpt_mgr.output_dir, "pretrain_final.pt")
        torch.save({
            "step": step,
            "tokens_seen": tokens_seen,
            "loss": accum_loss,
            "model_state_dict": self.model.state_dict(),
            "config": self.model.config.to_dict(),
            "type": "pretrain",
        }, final_ckpt)
        print(f"[*] Saved final pretrain checkpoint to: {final_ckpt}")

        total_elapsed = time.time() - start_time
        avg_throughput = tokens_seen / max(total_elapsed, 1e-4)
        print(f"[*] Pretraining Complete in {total_elapsed / 3600:.2f} hours")
        print(f"[*] Total Tokens Processed: {tokens_seen / 1e9:.3f} Billion | Avg Throughput: {avg_throughput:,.0f} tok/sec")

        return {
            "total_steps": step,
            "tokens_seen": tokens_seen,
            "final_loss": accum_loss,
            "elapsed_seconds": total_elapsed,
            "avg_throughput": avg_throughput,
            "final_checkpoint": final_ckpt,
        }


def main():
    import argparse
    import yaml
    from data.synthetic_pipeline import ASTJSONTraceGenerator, AgenticTraceGenerator

    parser = argparse.ArgumentParser(description="AttoModel Pre-Training Engine")
    parser.add_argument("--model_config", type=str, default="configs/model_18m.yaml", help="Path to model config YAML")
    parser.add_argument("--train_config", type=str, default="configs/training_wsd.yaml", help="Path to training config YAML")
    parser.add_argument("--data_config", type=str, default="configs/data_mix.yaml", help="Path to data mix config YAML")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--data_bin", type=str, default=None, help="Path to uint16 binary token dataset")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs over data_bin (default: 1 if subset data_bin provided and max_steps not specified)")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--micro_batch_size", type=int, default=None, help="Override micro-batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=None, help="Override gradient accumulation steps")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    # Load YAML configs
    model_config = AttoConfig.from_yaml(args.model_config)
    with open(args.train_config, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    micro_batch = args.micro_batch_size or train_cfg.get("training", {}).get("micro_batch_size", 8)
    accum_steps = args.grad_accum_steps or train_cfg.get("training", {}).get("gradient_accumulation_steps", 16)
    precision = train_cfg.get("precision", {}).get("amp_dtype", "bfloat16")
    muon_lr = train_cfg.get("optimizer", {}).get("muon", {}).get("lr", 0.02)
    adamw_lr = train_cfg.get("optimizer", {}).get("adamw", {}).get("lr", 1.2e-3)
    save_steps = train_cfg.get("checkpointing", {}).get("save_every_steps", 500)

    # Dataset setup: binary memmap dataset or dynamic curriculum mix from data_mix.yaml
    resolved_data_bin = None
    if args.data_bin and os.path.exists(args.data_bin):
        resolved_data_bin = args.data_bin
    elif os.path.exists("curriculum_train.bin"):
        resolved_data_bin = "curriculum_train.bin"
    elif os.path.exists("data/curriculum_train.bin"):
        resolved_data_bin = "data/curriculum_train.bin"

    tokens_per_step = micro_batch * accum_steps * model_config.max_seq_len  # 8 * 16 * 4096 = 524,288

    if resolved_data_bin:
        file_bytes = os.path.getsize(resolved_data_bin)
        total_tokens_in_bin = file_bytes // 2
        steps_per_epoch = max(1, math.ceil(total_tokens_in_bin / tokens_per_step))

        if args.max_steps is not None:
            max_steps = args.max_steps
        elif args.epochs is not None:
            max_steps = steps_per_epoch * args.epochs
            print(f"[*] Binary dataset ({total_tokens_in_bin:,} tokens). Setting max_steps={max_steps} ({args.epochs} epoch(s) @ {steps_per_epoch} steps/epoch).")
        elif total_tokens_in_bin < 1_000_000_000:
            # Subset dataset (e.g. 10M, 50M, 100M tokens): default to 1 complete epoch
            max_steps = steps_per_epoch
            print(f"[*] Binary dataset ({total_tokens_in_bin:,} tokens). Setting max_steps={max_steps} (1 full epoch @ {steps_per_epoch} steps).")
        else:
            max_steps = train_cfg.get("training", {}).get("max_steps", 9537)
    else:
        max_steps = args.max_steps or train_cfg.get("training", {}).get("max_steps", 9537)

    print("=" * 70)
    print(" ATTO-MODEL (18.5M) CURRICULUM PRE-TRAINING")
    print(f" Device: {device} | Max Steps: {max_steps} | Micro-Batch: {micro_batch} | Accum: {accum_steps}")
    print(f" Tokens/Step: {tokens_per_step:,} | Total Run Tokens: {max_steps * tokens_per_step / 1e6:.2f}M")
    print(f" Checkpoint Dir: {args.checkpoint_dir}")
    print("=" * 70)

    # Initialize model
    model = AttoModel(model_config)
    param_info = model.count_parameters()
    print(f" Total Parameters:  {param_info['total_parameters_M']}M")
    print(f" Active Parameters: {param_info['active_parameters_M']}M ({param_info['active_reduction_percent']}% reduction)")

    if resolved_data_bin:
        from data.dataset import BinaryMemmapDataset
        print(f"[*] Streaming from binary memmap dataset: {resolved_data_bin}")
        train_dataset = BinaryMemmapDataset(resolved_data_bin, max_seq_len=model_config.max_seq_len, infinite=True)
        train_loader = DataLoader(train_dataset, batch_size=micro_batch)
    else:
        print(f"[*] No binary dataset specified or found. Generating curriculum directly from '{args.data_config}'...")
        from data.synthetic_pipeline import CurriculumDataMixer
        tokenizer = get_tokenizer()

        fim_rate = 0.50
        spm_prob = 0.50
        if os.path.exists(args.data_config):
            try:
                with open(args.data_config, "r", encoding="utf-8") as f:
                    d_cfg = yaml.safe_load(f)
                    fim_rate = d_cfg.get("fim", {}).get("rate", 0.50)
                    spm_prob = d_cfg.get("fim", {}).get("spm_prob", 0.50)
            except Exception:
                pass

        # Generate curriculum documents according to Phase 1 (50%), Phase 2 (34%), Phase 3 (16%)
        synthetic_samples = CurriculumDataMixer.generate_curriculum_documents(
            total_samples=300,
            config_path=args.data_config,
        )

        # Pack into sequence length
        train_dataset = PackedSequenceDataset(
            documents=synthetic_samples,
            tokenizer=tokenizer,
            max_seq_len=min(model_config.max_seq_len, 512 if "cpu" in device else model_config.max_seq_len),
            fim_rate=fim_rate,
            spm_prob=spm_prob,
        )
        train_loader = DataLoader(train_dataset, batch_size=micro_batch, shuffle=True)
        print(f"[*] Prepared {len(train_dataset)} packed sequence samples across Syntax, Logic, and Agentic phases.")

    trainer = PretrainTrainer(
        model=model,
        train_dataloader=train_loader,
        total_steps=max_steps,
        grad_accum_steps=accum_steps,
        precision=precision if "cuda" in device else "float32",
        device=device,
        muon_lr=muon_lr,
        adamw_lr=adamw_lr,
        save_every_steps=save_steps,
        checkpoint_dir=args.checkpoint_dir,
    )

    trainer.train()


if __name__ == "__main__":
    main()
