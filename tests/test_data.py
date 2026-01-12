"""Tests for data.py - data loading utilities.

These tests require the datasets and transformers packages.
They use mocking to avoid network requests to HuggingFace.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Skip all tests in this module if dependencies aren't available
datasets = pytest.importorskip("datasets")
transformers = pytest.importorskip("transformers")

from unittest.mock import patch, MagicMock
import torch


class TestGetTokenizer:
    """Tests for tokenizer loading."""

    @patch("data.GPT2Tokenizer.from_pretrained")
    def test_returns_tokenizer_with_pad_token(self, mock_from_pretrained):
        """Tokenizer should have pad_token set to eos_token."""
        from data import get_tokenizer

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token = "<|endoftext|>"
        mock_from_pretrained.return_value = mock_tokenizer

        tokenizer = get_tokenizer()

        assert tokenizer.pad_token == tokenizer.eos_token


class TestTinyStoriesDataset:
    """Tests for TinyStoriesDataset class."""

    @patch("data.load_dataset")
    @patch("data.GPT2Tokenizer.from_pretrained")
    def test_initialization(self, mock_tokenizer, mock_load_dataset):
        """Should initialize with correct parameters."""
        from data import TinyStoriesDataset

        mock_tok = MagicMock()
        mock_tok.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value = mock_tok

        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=1000)
        mock_ds.select = MagicMock(return_value=mock_ds)
        mock_load_dataset.return_value = mock_ds

        dataset = TinyStoriesDataset(split="train", max_samples=100)

        mock_load_dataset.assert_called_once()
        assert dataset.max_seq_len == 512  # default


class TestDataLoaderCreation:
    """Tests for create_dataloaders function."""

    @patch("data.TinyStoriesDataset")
    @patch("data.GPT2Tokenizer.from_pretrained")
    def test_returns_two_dataloaders(self, mock_tokenizer, mock_dataset_class):
        """Should return train and val dataloaders."""
        from data import create_dataloaders

        mock_tok = MagicMock()
        mock_tok.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value = mock_tok

        mock_ds = MagicMock()
        mock_ds.__len__ = MagicMock(return_value=100)
        mock_dataset_class.return_value = mock_ds

        train_loader, val_loader = create_dataloaders(
            batch_size=4,
            max_train_samples=100,
            max_val_samples=50,
            num_workers=0,
        )

        assert train_loader is not None
        assert val_loader is not None


class TestStreamingTinyStories:
    """Tests for StreamingTinyStories class."""

    @patch("data.load_dataset")
    @patch("data.GPT2Tokenizer.from_pretrained")
    def test_initialization(self, mock_tokenizer, mock_load_dataset):
        """Should initialize with correct parameters."""
        from data import StreamingTinyStories

        mock_tok = MagicMock()
        mock_tok.eos_token = "<|endoftext|>"
        mock_tokenizer.return_value = mock_tok

        mock_ds = MagicMock()
        mock_load_dataset.return_value = mock_ds

        streamer = StreamingTinyStories(
            split="train",
            max_seq_len=256,
            batch_size=16,
        )

        assert streamer.max_seq_len == 256
        assert streamer.batch_size == 16
        mock_load_dataset.assert_called_once()
