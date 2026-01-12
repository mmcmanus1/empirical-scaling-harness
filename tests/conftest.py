"""Shared pytest fixtures for the scaling harness tests."""

import pytest
import torch

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ModelConfig


@pytest.fixture
def small_config_gelu() -> ModelConfig:
    """Create a minimal GeLU config for fast testing."""
    return ModelConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        d_ff=256,
        vocab_size=1000,
        max_seq_len=64,
        activation="gelu",
        dropout=0.0,
    )


@pytest.fixture
def small_config_swiglu() -> ModelConfig:
    """Create a minimal SwiGLU config for fast testing."""
    return ModelConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        d_ff=128,
        vocab_size=1000,
        max_seq_len=64,
        activation="swiglu",
        dropout=0.0,
    )


@pytest.fixture
def sample_batch(small_config_gelu) -> dict:
    """Create a sample batch for testing."""
    batch_size = 4
    seq_len = small_config_gelu.max_seq_len
    vocab_size = small_config_gelu.vocab_size

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
    }


@pytest.fixture
def device() -> str:
    """Get the appropriate device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"
