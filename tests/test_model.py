"""Tests for model.py - transformer architecture components."""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ModelConfig
from model import (
    SwiGLU,
    GeLUFFN,
    CausalSelfAttention,
    TransformerBlock,
    Transformer,
    build_model,
)


class TestSwiGLU:
    """Tests for SwiGLU activation module."""

    def test_output_shape(self):
        """Output shape should match input shape."""
        d_model, d_ff = 64, 128
        swiglu = SwiGLU(d_model, d_ff)

        x = torch.randn(2, 16, d_model)
        out = swiglu(x)

        assert out.shape == x.shape

    def test_has_three_projections(self):
        """SwiGLU should have gate, up, and down projections."""
        swiglu = SwiGLU(64, 128)

        assert hasattr(swiglu, "w_gate")
        assert hasattr(swiglu, "w_up")
        assert hasattr(swiglu, "w_down")

    def test_no_bias_in_projections(self):
        """SwiGLU projections should not have bias."""
        swiglu = SwiGLU(64, 128)

        assert swiglu.w_gate.bias is None
        assert swiglu.w_up.bias is None
        assert swiglu.w_down.bias is None


class TestGeLUFFN:
    """Tests for GeLU FFN module."""

    def test_output_shape(self):
        """Output shape should match input shape."""
        d_model, d_ff = 64, 256
        ffn = GeLUFFN(d_model, d_ff)

        x = torch.randn(2, 16, d_model)
        out = ffn(x)

        assert out.shape == x.shape

    def test_has_two_projections(self):
        """GeLU FFN should have up and down projections."""
        ffn = GeLUFFN(64, 256)

        assert hasattr(ffn, "w_up")
        assert hasattr(ffn, "w_down")


class TestCausalSelfAttention:
    """Tests for causal self-attention module."""

    def test_output_shape(self, small_config_gelu):
        """Output shape should match input shape."""
        attn = CausalSelfAttention(small_config_gelu)

        batch_size = 2
        seq_len = small_config_gelu.max_seq_len
        x = torch.randn(batch_size, seq_len, small_config_gelu.d_model)
        out = attn(x)

        assert out.shape == x.shape

    def test_causal_mask_registered(self, small_config_gelu):
        """Causal mask should be registered as buffer."""
        attn = CausalSelfAttention(small_config_gelu)

        assert hasattr(attn, "causal_mask")
        assert attn.causal_mask.shape == (
            1,
            1,
            small_config_gelu.max_seq_len,
            small_config_gelu.max_seq_len,
        )

    def test_causal_mask_is_lower_triangular(self, small_config_gelu):
        """Causal mask should be lower triangular."""
        attn = CausalSelfAttention(small_config_gelu)

        mask = attn.causal_mask.squeeze()
        # Lower triangular means mask[i,j] = 0 for j > i
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if j > i:
                    assert mask[i, j] == 0


class TestTransformerBlock:
    """Tests for transformer block."""

    @pytest.mark.parametrize("activation", ["gelu", "swiglu"])
    def test_output_shape(self, activation):
        """Output shape should match input shape."""
        config = ModelConfig(
            n_layers=1,
            d_model=64,
            n_heads=2,
            d_ff=128,
            vocab_size=1000,
            max_seq_len=32,
            activation=activation,
        )
        block = TransformerBlock(config)

        x = torch.randn(2, 32, 64)
        out = block(x)

        assert out.shape == x.shape

    def test_uses_correct_ffn_type(self, small_config_gelu, small_config_swiglu):
        """Block should use correct FFN type based on activation."""
        gelu_block = TransformerBlock(small_config_gelu)
        swiglu_block = TransformerBlock(small_config_swiglu)

        assert isinstance(gelu_block.ffn, GeLUFFN)
        assert isinstance(swiglu_block.ffn, SwiGLU)

    def test_has_pre_norm_architecture(self, small_config_gelu):
        """Block should have LayerNorm before attention and FFN."""
        block = TransformerBlock(small_config_gelu)

        assert hasattr(block, "ln1")
        assert hasattr(block, "ln2")
        assert isinstance(block.ln1, nn.LayerNorm)
        assert isinstance(block.ln2, nn.LayerNorm)


class TestTransformer:
    """Tests for full transformer model."""

    def test_forward_pass(self, small_config_gelu, sample_batch):
        """Forward pass should return logits and loss."""
        model = Transformer(small_config_gelu)

        logits, loss = model(
            sample_batch["input_ids"],
            labels=sample_batch["labels"],
        )

        assert logits.shape == (
            sample_batch["input_ids"].shape[0],
            sample_batch["input_ids"].shape[1],
            small_config_gelu.vocab_size,
        )
        assert loss is not None
        assert loss.dim() == 0  # Scalar

    def test_forward_without_labels(self, small_config_gelu, sample_batch):
        """Forward pass without labels should return None for loss."""
        model = Transformer(small_config_gelu)

        logits, loss = model(sample_batch["input_ids"])

        assert logits is not None
        assert loss is None

    def test_weight_tying(self, small_config_gelu):
        """Token embeddings should be tied to output projection."""
        model = Transformer(small_config_gelu)

        # Check that weights are the same object
        assert model.lm_head.weight is model.token_emb.weight

    def test_count_parameters(self, small_config_gelu):
        """count_parameters should return positive integer."""
        model = Transformer(small_config_gelu)

        params = model.count_parameters()

        assert params > 0
        assert isinstance(params, int)

    @pytest.mark.parametrize("activation", ["gelu", "swiglu"])
    def test_builds_with_both_activations(self, activation):
        """Model should build successfully with both activation types."""
        config = ModelConfig(
            n_layers=2,
            d_model=64,
            n_heads=2,
            d_ff=128,
            vocab_size=1000,
            max_seq_len=32,
            activation=activation,
        )
        model = Transformer(config)

        assert model is not None
        assert len(model.blocks) == 2


class TestBuildModel:
    """Tests for build_model convenience function."""

    def test_returns_transformer(self, small_config_gelu):
        """build_model should return a Transformer instance."""
        model = build_model(small_config_gelu)

        assert isinstance(model, Transformer)

    def test_config_stored_on_model(self, small_config_gelu):
        """Config should be accessible on model."""
        model = build_model(small_config_gelu)

        assert model.config is small_config_gelu


class TestGeneration:
    """Tests for text generation."""

    @pytest.mark.slow
    def test_generate_produces_tokens(self, small_config_gelu):
        """generate should produce new tokens."""
        model = Transformer(small_config_gelu)
        model.eval()

        input_ids = torch.randint(0, small_config_gelu.vocab_size, (1, 10))
        max_new_tokens = 5

        output = model.generate(input_ids, max_new_tokens=max_new_tokens)

        assert output.shape[1] == input_ids.shape[1] + max_new_tokens
