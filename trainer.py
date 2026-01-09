"""
Training loop with logging for scaling experiments.

Supports:
- AdamW optimizer with cosine LR schedule
- Mixed precision training
- Gradient accumulation
- CSV and TensorBoard logging
"""

import os
import csv
import math
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import Transformer
from config import ModelConfig, compute_flops


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0

    # Training schedule
    total_tokens: int = 60_000_000  # Chinchilla: 20 * params
    warmup_steps: int = 100
    batch_size: int = 32
    gradient_accumulation_steps: int = 1

    # Mixed precision
    use_amp: bool = False

    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    checkpoint_interval: int = 1000

    # Paths
    output_dir: str = "logs"
    run_name: str = "run"

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


class Trainer:
    """Trainer for language models."""

    def __init__(
        self,
        model: Transformer,
        model_config: ModelConfig,
        train_config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.model_config = model_config
        self.config = train_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Calculate training steps
        tokens_per_step = (
            train_config.effective_batch_size * model_config.max_seq_len
        )
        self.total_steps = train_config.total_tokens // tokens_per_step
        print(f"Training for {self.total_steps} steps ({train_config.total_tokens:,} tokens)")

        # Setup scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_steps,
            eta_min=train_config.learning_rate * 0.1,
        )

        # Mixed precision
        self.scaler = GradScaler('cuda') if (train_config.use_amp and device == "cuda") else None

        # Logging
        self.output_dir = Path(train_config.output_dir) / train_config.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tb_writer = SummaryWriter(self.output_dir / "tensorboard")
        self.csv_path = self.output_dir / "metrics.csv"
        self._init_csv()

        # Training state
        self.step = 0
        self.tokens_processed = 0
        self.best_val_loss = float("inf")

    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer with weight decay only on non-embedding params."""
        # Separate parameters that should have weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "emb" in name or "ln" in name or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return AdamW(
            optim_groups,
            lr=self.config.learning_rate,
            betas=self.config.betas,
        )

    def _init_csv(self):
        """Initialize CSV log file."""
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step",
                "tokens_processed",
                "train_loss",
                "val_loss",
                "learning_rate",
                "time_elapsed",
                "tokens_per_sec",
            ])

    def _log_metrics(
        self,
        train_loss: float,
        val_loss: Optional[float],
        time_elapsed: float,
    ):
        """Log metrics to CSV and TensorBoard."""
        lr = self.scheduler.get_last_lr()[0]
        tokens_per_sec = self.tokens_processed / time_elapsed if time_elapsed > 0 else 0

        # CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.step,
                self.tokens_processed,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}" if val_loss else "",
                f"{lr:.2e}",
                f"{time_elapsed:.1f}",
                f"{tokens_per_sec:.0f}",
            ])

        # TensorBoard
        self.tb_writer.add_scalar("train/loss", train_loss, self.step)
        self.tb_writer.add_scalar("train/learning_rate", lr, self.step)
        self.tb_writer.add_scalar("train/tokens_per_sec", tokens_per_sec, self.step)

        if val_loss is not None:
            self.tb_writer.add_scalar("val/loss", val_loss, self.step)

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            with autocast('cuda', enabled=self.config.use_amp):
                _, loss = self.model(input_ids, labels=labels)

            total_loss += loss.item()
            num_batches += 1

            # Limit eval batches for speed
            if num_batches >= 50:
                break

        self.model.train()
        return total_loss / num_batches

    def train_step(self, batch: dict) -> float:
        """Execute one training step."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        with autocast('cuda', enabled=self.config.use_amp):
            _, loss = self.model(input_ids, labels=labels)
            loss = loss / self.config.gradient_accumulation_steps

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item() * self.config.gradient_accumulation_steps

    def train(self) -> dict:
        """
        Main training loop.

        Returns:
            Dictionary with final metrics
        """
        self.model.train()
        start_time = time.time()

        # Training loop
        train_iter = iter(self.train_loader)
        accumulated_loss = 0.0
        pbar = tqdm(total=self.total_steps, desc="Training")

        while self.step < self.total_steps:
            # Accumulate gradients
            for _ in range(self.config.gradient_accumulation_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                loss = self.train_step(batch)
                accumulated_loss += loss

                # Update token count
                batch_tokens = batch["input_ids"].numel()
                self.tokens_processed += batch_tokens

            # Gradient step
            if self.config.grad_clip > 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)

            # Learning rate warmup
            if self.step < self.config.warmup_steps:
                lr_scale = (self.step + 1) / self.config.warmup_steps
                for pg in self.optimizer.param_groups:
                    pg["lr"] = self.config.learning_rate * lr_scale
            else:
                self.scheduler.step()

            self.step += 1
            avg_loss = accumulated_loss / self.config.gradient_accumulation_steps
            accumulated_loss = 0.0

            # Logging
            if self.step % self.config.log_interval == 0:
                time_elapsed = time.time() - start_time
                val_loss = None

                if self.step % self.config.eval_interval == 0:
                    val_loss = self.evaluate()
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint("best")

                self._log_metrics(avg_loss, val_loss, time_elapsed)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "val": f"{val_loss:.4f}" if val_loss else "N/A"})

            # Checkpointing
            if self.step % self.config.checkpoint_interval == 0:
                self._save_checkpoint(f"step_{self.step}")

            pbar.update(1)

        pbar.close()

        # Final evaluation
        final_val_loss = self.evaluate()
        time_elapsed = time.time() - start_time
        self._log_metrics(avg_loss, final_val_loss, time_elapsed)

        # Save final checkpoint
        self._save_checkpoint("final")

        # Compute final metrics
        params = self.model.count_parameters()
        flops = compute_flops(params, self.tokens_processed)

        final_metrics = {
            "params": params,
            "tokens": self.tokens_processed,
            "flops": flops,
            "final_val_loss": final_val_loss,
            "best_val_loss": self.best_val_loss,
            "training_time": time_elapsed,
        }

        # Save final metrics
        self._save_final_metrics(final_metrics)

        self.tb_writer.close()
        return final_metrics

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "step": self.step,
            "tokens_processed": self.tokens_processed,
            "model_config": self.model_config,
        }
        torch.save(checkpoint, self.output_dir / f"checkpoint_{name}.pt")

    def _save_final_metrics(self, metrics: dict):
        """Save final metrics to JSON-like format."""
        import json
        with open(self.output_dir / "final_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)


def train_model(
    model: Transformer,
    model_config: ModelConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    total_tokens: int,
    run_name: str,
    output_dir: str = "logs",
    batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 3e-4,
    use_amp: bool = False,
    device: str = None,
) -> dict:
    """
    Convenience function to train a model.

    Args:
        model: The transformer model
        model_config: Model configuration
        train_loader: Training dataloader
        val_loader: Validation dataloader
        total_tokens: Total tokens to train on
        run_name: Name for this run
        output_dir: Output directory for logs
        batch_size: Batch size
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        use_amp: Whether to use mixed precision
        device: Device to train on

    Returns:
        Dictionary with final metrics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_config = TrainingConfig(
        total_tokens=total_tokens,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        use_amp=use_amp,
        output_dir=output_dir,
        run_name=run_name,
    )

    trainer = Trainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    return trainer.train()
