"""
Master orchestrator for scaling law experiments.

Automates running the full experimental grid:
- 2 activations (GeLU, SwiGLU)
- 4 model sizes (3M, 10M, 30M anchors + 85M holdout)
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

from config import ANCHOR_PARAMS, HOLDOUT_PARAMS, ALL_PARAMS, build_config
from sweep import run_experiment


def run_full_sweep(
    output_dir: str = "logs",
    dry_run: bool = False,
    activations: list = None,
    model_sizes: list = None,
    skip_holdout: bool = False,
) -> dict:
    """
    Run the full experimental sweep.

    Args:
        output_dir: Output directory for logs
        dry_run: If True, only print configs without training
        activations: List of activations to test (default: ["gelu", "swiglu"])
        model_sizes: List of model sizes (default: all anchor + holdout)
        skip_holdout: If True, skip the 85M holdout models

    Returns:
        Dictionary with all experiment results
    """
    if activations is None:
        activations = ["gelu", "swiglu"]

    if model_sizes is None:
        if skip_holdout:
            model_sizes = ANCHOR_PARAMS
        else:
            model_sizes = ALL_PARAMS

    # Print experiment plan
    print("\n" + "=" * 70)
    print("SCALING LAW EXPERIMENT SWEEP")
    print("=" * 70)
    print(f"\nActivations: {activations}")
    print(f"Model sizes: {[f'{p/1e6:.0f}M' for p in model_sizes]}")
    print(f"Total experiments: {len(activations) * len(model_sizes)}")
    print(f"Output directory: {output_dir}")
    print(f"Dry run: {dry_run}")

    # Create timestamp for this sweep
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(output_dir) / f"sweep_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Track all results
    all_results = {}
    start_time = time.time()

    # Run experiments
    experiment_num = 0
    total_experiments = len(activations) * len(model_sizes)

    for activation in activations:
        for params in model_sizes:
            experiment_num += 1
            print(f"\n{'=' * 70}")
            print(f"EXPERIMENT {experiment_num}/{total_experiments}")
            print(f"{'=' * 70}")

            run_name = f"{activation}_{params // 1_000_000}M"

            try:
                metrics = run_experiment(
                    target_params=params,
                    activation=activation,
                    output_dir=str(sweep_dir),
                    run_name=run_name,
                    dry_run=dry_run,
                )
                all_results[run_name] = metrics
                all_results[run_name]["status"] = "success"

            except Exception as e:
                print(f"\nERROR in experiment {run_name}: {e}")
                all_results[run_name] = {
                    "status": "failed",
                    "error": str(e),
                    "params": params,
                    "activation": activation,
                }

            # Save intermediate results
            with open(sweep_dir / "all_results.json", "w") as f:
                json.dump(all_results, f, indent=2)

    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {sweep_dir}")

    # Print summary table
    print("\nResults Summary:")
    print("-" * 50)
    print(f"{'Run Name':<20} {'Params':>10} {'Val Loss':>10}")
    print("-" * 50)

    for run_name, result in all_results.items():
        if result.get("status") == "success":
            params = result.get("params", 0)
            val_loss = result.get("final_val_loss", 0)
            print(f"{run_name:<20} {params/1e6:>9.1f}M {val_loss:>10.4f}")
        else:
            print(f"{run_name:<20} {'FAILED':>10} {result.get('error', '')[:20]}")

    return all_results


def print_experiment_grid(dry_run: bool = True):
    """Print the experiment grid without running."""
    print("\n" + "=" * 70)
    print("EXPERIMENT GRID (DRY RUN)")
    print("=" * 70)

    for activation in ["gelu", "swiglu"]:
        print(f"\n{activation.upper()} Models:")
        print("-" * 60)
        print(f"{'Target':>10} {'Actual':>10} {'Layers':>8} {'d_model':>8} {'d_ff':>8}")
        print("-" * 60)

        for params in ALL_PARAMS:
            config = build_config(params, activation=activation)
            actual = config.count_parameters()
            print(f"{params/1e6:>9.0f}M {actual/1e6:>9.2f}M {config.n_layers:>8} "
                  f"{config.d_model:>8} {config.d_ff:>8}")


def main():
    parser = argparse.ArgumentParser(
        description="Run scaling law experiment sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs",
        help="Output directory for logs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print experiment grid without training",
    )
    parser.add_argument(
        "--activations",
        type=str,
        nargs="+",
        default=None,
        choices=["gelu", "swiglu"],
        help="Activations to test (default: both)",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        nargs="+",
        default=None,
        choices=["3M", "10M", "30M", "85M"],
        help="Model sizes to test (default: all)",
    )
    parser.add_argument(
        "--skip-holdout",
        action="store_true",
        help="Skip the 85M holdout models",
    )
    parser.add_argument(
        "--anchors-only",
        action="store_true",
        help="Only run anchor points (3M, 10M, 30M)",
    )

    args = parser.parse_args()

    # Parse model sizes
    model_sizes = None
    if args.sizes:
        size_map = {"3M": 3_000_000, "10M": 10_000_000, "30M": 30_000_000, "85M": 85_000_000}
        model_sizes = [size_map[s] for s in args.sizes]
    elif args.anchors_only:
        model_sizes = ANCHOR_PARAMS

    if args.dry_run:
        print_experiment_grid()
    else:
        run_full_sweep(
            output_dir=args.output_dir,
            dry_run=False,
            activations=args.activations,
            model_sizes=model_sizes,
            skip_holdout=args.skip_holdout,
        )


if __name__ == "__main__":
    main()
