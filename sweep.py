"""
Experiment runner CLI for scaling law experiments.

Run individual experiments with specified model size and activation function.
"""

import argparse
import json
from pathlib import Path

import torch

from config import (
    build_config,
    get_training_config,
    compute_optimal_tokens,
    compute_flops,
)
from model import build_model
from data import create_dataloaders
from trainer import train_model


def run_experiment(
    target_params: int,
    activation: str,
    output_dir: str = "logs",
    run_name: str = None,
    dry_run: bool = False,
    device: str = None,
) -> dict:
    """
    Run a single training experiment.

    Args:
        target_params: Target parameter count
        activation: Activation function ("gelu" or "swiglu")
        output_dir: Output directory for logs
        run_name: Name for this run (auto-generated if None)
        dry_run: If True, only print config without training
        device: Device to train on

    Returns:
        Dictionary with experiment results
    """
    # Build model config
    model_config = build_config(target_params, activation=activation)
    actual_params = model_config.count_parameters()

    # Get training hyperparameters
    train_hparams = get_training_config(actual_params)

    # Compute training budget
    total_tokens = compute_optimal_tokens(actual_params)
    total_flops = compute_flops(actual_params, total_tokens)

    # Generate run name
    if run_name is None:
        run_name = f"{activation}_{actual_params / 1_000_000:.1f}M"

    # Print experiment info
    print("\n" + "=" * 60)
    print(f"EXPERIMENT: {run_name}")
    print("=" * 60)
    print(f"\nModel Configuration:")
    print(f"  Target params:  {target_params:,}")
    print(f"  Actual params:  {actual_params:,}")
    print(f"  Activation:     {activation}")
    print(f"  Layers:         {model_config.n_layers}")
    print(f"  d_model:        {model_config.d_model}")
    print(f"  d_ff:           {model_config.d_ff}")
    print(f"  n_heads:        {model_config.n_heads}")
    print(f"  max_seq_len:    {model_config.max_seq_len}")

    print(f"\nTraining Configuration:")
    print(f"  Total tokens:   {total_tokens:,} ({total_tokens/1e6:.1f}M)")
    print(f"  Total FLOPs:    {total_flops:.2e}")
    print(f"  Batch size:     {train_hparams['batch_size']}")
    print(f"  Grad accum:     {train_hparams['gradient_accumulation_steps']}")
    print(f"  Learning rate:  {train_hparams['learning_rate']:.0e}")
    print(f"  Use AMP:        {train_hparams['use_amp']}")

    if dry_run:
        print("\n[DRY RUN] Skipping training")
        return {
            "run_name": run_name,
            "params": actual_params,
            "tokens": total_tokens,
            "flops": total_flops,
            "config": {
                "n_layers": model_config.n_layers,
                "d_model": model_config.d_model,
                "d_ff": model_config.d_ff,
                "n_heads": model_config.n_heads,
                "activation": activation,
            },
        }

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Build model
    print("\nBuilding model...")
    model = build_model(model_config)
    print(f"Model parameters: {model.count_parameters():,}")

    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader = create_dataloaders(
        batch_size=train_hparams["batch_size"],
        max_seq_len=model_config.max_seq_len,
        num_workers=2,  # Reduce for Colab
    )

    # Train
    print("\nStarting training...")
    metrics = train_model(
        model=model,
        model_config=model_config,
        train_loader=train_loader,
        val_loader=val_loader,
        total_tokens=total_tokens,
        run_name=run_name,
        output_dir=output_dir,
        batch_size=train_hparams["batch_size"],
        gradient_accumulation_steps=train_hparams["gradient_accumulation_steps"],
        learning_rate=train_hparams["learning_rate"],
        use_amp=train_hparams["use_amp"],
        device=device,
    )

    # Print final results
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Final validation loss: {metrics['final_val_loss']:.4f}")
    print(f"Best validation loss:  {metrics['best_val_loss']:.4f}")
    print(f"Training time:         {metrics['training_time']/60:.1f} minutes")
    print(f"Tokens processed:      {metrics['tokens']:,}")
    print(f"Total FLOPs:           {metrics['flops']:.2e}")

    # Add experiment metadata to metrics
    metrics["run_name"] = run_name
    metrics["activation"] = activation
    metrics["model_config"] = {
        "n_layers": model_config.n_layers,
        "d_model": model_config.d_model,
        "d_ff": model_config.d_ff,
        "n_heads": model_config.n_heads,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run a scaling law experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--params",
        type=int,
        required=True,
        help="Target parameter count (e.g., 3000000 for 3M)",
    )
    parser.add_argument(
        "--activation",
        type=str,
        choices=["gelu", "swiglu"],
        default="gelu",
        help="Activation function",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs",
        help="Output directory for logs",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this run (auto-generated if not provided)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config without training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (auto-detected if not provided)",
    )

    args = parser.parse_args()

    metrics = run_experiment(
        target_params=args.params,
        activation=args.activation,
        output_dir=args.output_dir,
        run_name=args.run_name,
        dry_run=args.dry_run,
        device=args.device,
    )

    # Save results to JSON
    if not args.dry_run:
        results_dir = Path(args.output_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / "all_results.json"

        # Load existing results if any
        all_results = {}
        if results_file.exists():
            with open(results_file) as f:
                all_results = json.load(f)

        # Add new result
        all_results[metrics["run_name"]] = metrics

        # Save
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
