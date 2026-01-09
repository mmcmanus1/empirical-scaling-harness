"""
Model configuration builder for scaling experiments.

Computes model dimensions (layers, width, heads) to hit a target parameter count.
"""

from dataclasses import dataclass
from typing import Literal
import math


@dataclass
class ModelConfig:
    """Configuration for a decoder-only transformer."""
    n_layers: int
    d_model: int
    n_heads: int
    d_ff: int
    vocab_size: int
    max_seq_len: int
    activation: Literal["gelu", "swiglu"]
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        # Embedding
        embed_params = self.vocab_size * self.d_model

        # Per layer
        # Attention: Q, K, V, O projections
        attn_params = 4 * self.d_model * self.d_model

        # FFN
        if self.activation == "swiglu":
            # SwiGLU: gate, up, down projections
            ffn_params = 3 * self.d_model * self.d_ff
        else:
            # GeLU: up, down projections
            ffn_params = 2 * self.d_model * self.d_ff

        # Layer norms: weight + bias (2 * d_model each), 2 per layer + 1 final
        ln_params = 4 * self.d_model * self.n_layers + 2 * self.d_model

        layer_params = (attn_params + ffn_params) * self.n_layers

        # Output projection (tied with embedding in our case, so not counted)
        total = embed_params + layer_params + ln_params

        return total


# Standard model dimensions for different scales
# These are approximate - the config builder will adjust to hit exact targets
MODEL_SCALES = {
    "3M": {"d_model": 256, "n_layers": 4},
    "10M": {"d_model": 384, "n_layers": 6},
    "30M": {"d_model": 512, "n_layers": 8},
    "85M": {"d_model": 768, "n_layers": 12},
}


def compute_ffn_dim(d_model: int, activation: str) -> int:
    """
    Compute FFN intermediate dimension.

    For GeLU: d_ff = 4 * d_model (standard)
    For SwiGLU: d_ff = 8/3 * d_model to match param count
    """
    if activation == "swiglu":
        # SwiGLU uses 3 matrices instead of 2, so reduce d_ff
        # to keep param count equivalent
        d_ff = int(8 * d_model / 3)
        # Round to multiple of 64 for efficiency
        d_ff = (d_ff // 64) * 64
        if d_ff == 0:
            d_ff = 64
    else:
        d_ff = 4 * d_model
    return d_ff


def build_config(
    target_params: int,
    activation: Literal["gelu", "swiglu"] = "gelu",
    vocab_size: int = 50257,
    max_seq_len: int = 512,
) -> ModelConfig:
    """
    Build a model config targeting a specific parameter count.

    Uses heuristics to find dimensions that approximate the target,
    then reports the actual param count.

    Args:
        target_params: Target number of parameters (e.g., 3_000_000)
        activation: Activation function ("gelu" or "swiglu")
        vocab_size: Vocabulary size (default: GPT-2's 50257)
        max_seq_len: Maximum sequence length

    Returns:
        ModelConfig with dimensions approximating target_params
    """
    # Find the closest scale
    scale_name = None
    for name, scale in MODEL_SCALES.items():
        scale_params = int(name.replace("M", "")) * 1_000_000
        if scale_params >= target_params * 0.8 and scale_params <= target_params * 1.2:
            scale_name = name
            break

    if scale_name:
        base_config = MODEL_SCALES[scale_name]
        d_model = base_config["d_model"]
        n_layers = base_config["n_layers"]
    else:
        # Estimate dimensions from target params
        # N ≈ vocab_size * d_model + 12 * n_layers * d_model^2
        # Assume n_layers = d_model / 64 as a rough heuristic
        # Solve for d_model
        d_model = int(math.sqrt(target_params / 20))  # rough estimate
        d_model = max(64, (d_model // 64) * 64)  # round to multiple of 64
        n_layers = max(2, d_model // 64)

    # Compute FFN dimension based on activation
    d_ff = compute_ffn_dim(d_model, activation)

    # Number of heads (one head per 64 dims)
    n_heads = max(1, d_model // 64)

    config = ModelConfig(
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        activation=activation,
    )

    # Fine-tune n_layers to get closer to target
    actual_params = config.count_parameters()

    # Adjust layers if we're off by more than 10%
    while actual_params < target_params * 0.9 and n_layers < 24:
        n_layers += 1
        config = ModelConfig(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            activation=activation,
        )
        actual_params = config.count_parameters()

    while actual_params > target_params * 1.1 and n_layers > 1:
        n_layers -= 1
        config = ModelConfig(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            activation=activation,
        )
        actual_params = config.count_parameters()

    return config


def get_training_config(params: int) -> dict:
    """
    Get training hyperparameters based on model size.

    Returns batch size, learning rate, and gradient accumulation steps
    optimized for Colab (~15GB VRAM).
    """
    if params <= 5_000_000:
        return {
            "batch_size": 32,
            "gradient_accumulation_steps": 1,
            "learning_rate": 3e-4,
            "use_amp": False,
        }
    elif params <= 15_000_000:
        return {
            "batch_size": 16,
            "gradient_accumulation_steps": 2,
            "learning_rate": 2e-4,
            "use_amp": False,
        }
    elif params <= 50_000_000:
        return {
            "batch_size": 8,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1e-4,
            "use_amp": True,
        }
    else:
        return {
            "batch_size": 4,
            "gradient_accumulation_steps": 8,
            "learning_rate": 6e-5,
            "use_amp": True,
        }


def compute_optimal_tokens(params: int) -> int:
    """
    Compute the optimal number of training tokens following Chinchilla scaling.

    Chinchilla ratio: D ≈ 20N for compute-optimal training.
    """
    return 20 * params


def compute_flops(params: int, tokens: int) -> float:
    """
    Estimate total training FLOPs.

    Following the approximation: C ≈ 6 * N * D
    (forward + backward pass ≈ 6x the params * tokens)
    """
    return 6 * params * tokens


# Experiment configurations
ANCHOR_PARAMS = [3_000_000, 10_000_000, 30_000_000]
HOLDOUT_PARAMS = [85_000_000]
ALL_PARAMS = ANCHOR_PARAMS + HOLDOUT_PARAMS


if __name__ == "__main__":
    # Test config generation
    print("Model Configurations for Scaling Experiment\n")
    print("=" * 70)

    for activation in ["gelu", "swiglu"]:
        print(f"\n{activation.upper()} Activation:")
        print("-" * 70)

        for target in ALL_PARAMS:
            config = build_config(target, activation=activation)
            actual = config.count_parameters()
            tokens = compute_optimal_tokens(actual)
            flops = compute_flops(actual, tokens)

            print(f"Target: {target/1e6:.1f}M | Actual: {actual/1e6:.2f}M")
            print(f"  Layers: {config.n_layers}, d_model: {config.d_model}, "
                  f"d_ff: {config.d_ff}, heads: {config.n_heads}")
            print(f"  Tokens: {tokens/1e6:.1f}M, FLOPs: {flops:.2e}")
            print()
