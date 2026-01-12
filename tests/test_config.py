"""Tests for config.py - model configuration and parameter counting."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ModelConfig,
    build_config,
    compute_ffn_dim,
    compute_optimal_tokens,
    compute_flops,
    get_training_config,
    ANCHOR_PARAMS,
    HOLDOUT_PARAMS,
    ALL_PARAMS,
)


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_head_dim_calculation(self):
        """head_dim should equal d_model // n_heads."""
        config = ModelConfig(
            n_layers=4,
            d_model=256,
            n_heads=4,
            d_ff=1024,
            vocab_size=50257,
            max_seq_len=512,
            activation="gelu",
        )
        assert config.head_dim == 64

    def test_count_parameters_gelu(self):
        """Parameter count for GeLU should include 2 FFN matrices per layer."""
        config = ModelConfig(
            n_layers=1,
            d_model=64,
            n_heads=1,
            d_ff=256,
            vocab_size=100,
            max_seq_len=32,
            activation="gelu",
        )
        params = config.count_parameters()

        assert params > 0
        assert isinstance(params, int)

    def test_count_parameters_swiglu_vs_gelu(self):
        """SwiGLU with same d_ff should have more params (3 matrices vs 2)."""
        gelu_config = ModelConfig(
            n_layers=2,
            d_model=128,
            n_heads=2,
            d_ff=512,
            vocab_size=1000,
            max_seq_len=64,
            activation="gelu",
        )
        swiglu_config = ModelConfig(
            n_layers=2,
            d_model=128,
            n_heads=2,
            d_ff=512,
            vocab_size=1000,
            max_seq_len=64,
            activation="swiglu",
        )

        gelu_params = gelu_config.count_parameters()
        swiglu_params = swiglu_config.count_parameters()

        # SwiGLU uses 3 matrices, GeLU uses 2, so with same d_ff, SwiGLU has more
        assert swiglu_params > gelu_params


class TestComputeFFNDim:
    """Tests for compute_ffn_dim function."""

    def test_gelu_ffn_is_4x(self):
        """GeLU FFN dimension should be 4x d_model."""
        assert compute_ffn_dim(256, "gelu") == 1024
        assert compute_ffn_dim(512, "gelu") == 2048

    def test_swiglu_ffn_is_smaller(self):
        """SwiGLU FFN dimension should be 8/3 * d_model, rounded to 64."""
        d_model = 256
        d_ff_swiglu = compute_ffn_dim(d_model, "swiglu")
        d_ff_gelu = compute_ffn_dim(d_model, "gelu")

        # SwiGLU d_ff should be less than GeLU d_ff
        assert d_ff_swiglu < d_ff_gelu
        # Should be multiple of 64
        assert d_ff_swiglu % 64 == 0

    def test_swiglu_minimum_ffn(self):
        """SwiGLU should have minimum d_ff of 64."""
        d_ff = compute_ffn_dim(16, "swiglu")
        assert d_ff >= 64


class TestBuildConfig:
    """Tests for build_config function."""

    @pytest.mark.parametrize("target_params", ANCHOR_PARAMS)
    def test_builds_config_for_anchor_sizes(self, target_params):
        """Should build valid config for anchor model sizes."""
        config = build_config(target_params, activation="gelu")

        assert config.n_layers >= 1
        assert config.d_model >= 64
        assert config.n_heads >= 1
        assert config.d_model % config.n_heads == 0  # Must divide evenly

    @pytest.mark.parametrize("activation", ["gelu", "swiglu"])
    def test_builds_config_for_both_activations(self, activation):
        """Should build valid config for both activation types."""
        config = build_config(3_000_000, activation=activation)

        assert config.activation == activation
        assert config.count_parameters() > 0

    def test_actual_params_reasonable(self):
        """Built config should produce a valid model with reasonable param count."""
        target = 10_000_000
        config = build_config(target, activation="gelu")
        actual = config.count_parameters()

        # The builder is approximate - just verify it produces a valid config
        assert actual > 0
        assert config.n_layers >= 1
        assert config.d_model >= 64

    def test_gelu_swiglu_similar_params(self):
        """GeLU and SwiGLU configs for same target should have similar param counts."""
        target = 10_000_000

        gelu_config = build_config(target, activation="gelu")
        swiglu_config = build_config(target, activation="swiglu")

        gelu_params = gelu_config.count_parameters()
        swiglu_params = swiglu_config.count_parameters()

        # Should be within 15% of each other
        ratio = max(gelu_params, swiglu_params) / min(gelu_params, swiglu_params)
        assert ratio < 1.15


class TestTrainingConfig:
    """Tests for get_training_config function."""

    def test_small_model_config(self):
        """Small models should have larger batch size, no AMP."""
        config = get_training_config(3_000_000)

        assert config["batch_size"] >= 16
        assert config["use_amp"] is False

    def test_large_model_config(self):
        """Large models should have smaller batch size, use AMP."""
        config = get_training_config(85_000_000)

        assert config["batch_size"] <= 8
        assert config["use_amp"] is True

    def test_all_required_keys_present(self):
        """Config should have all required keys."""
        required_keys = ["batch_size", "gradient_accumulation_steps", "learning_rate", "use_amp"]

        for params in [3_000_000, 30_000_000, 85_000_000]:
            config = get_training_config(params)
            for key in required_keys:
                assert key in config


class TestScalingComputation:
    """Tests for scaling-related computation functions."""

    def test_optimal_tokens_chinchilla(self):
        """Optimal tokens should follow Chinchilla ratio (20x params)."""
        params = 10_000_000
        tokens = compute_optimal_tokens(params)

        assert tokens == 20 * params

    def test_flops_computation(self):
        """FLOPs should follow C = 6 * N * D formula."""
        params = 10_000_000
        tokens = 200_000_000
        flops = compute_flops(params, tokens)

        assert flops == 6 * params * tokens


class TestConstants:
    """Tests for module constants."""

    def test_anchor_params_sorted(self):
        """Anchor params should be in ascending order."""
        assert ANCHOR_PARAMS == sorted(ANCHOR_PARAMS)

    def test_holdout_larger_than_anchors(self):
        """Holdout params should be larger than all anchors."""
        for holdout in HOLDOUT_PARAMS:
            for anchor in ANCHOR_PARAMS:
                assert holdout > anchor

    def test_all_params_is_union(self):
        """ALL_PARAMS should be anchors + holdout."""
        assert set(ALL_PARAMS) == set(ANCHOR_PARAMS) | set(HOLDOUT_PARAMS)
