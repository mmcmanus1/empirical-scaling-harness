"""
Data loading for TinyStories dataset.

Uses HuggingFace datasets and GPT-2 tokenizer.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from typing import Iterator, Optional


class TinyStoriesDataset(Dataset):
    """TinyStories dataset for language modeling."""

    def __init__(
        self,
        split: str = "train",
        max_seq_len: int = 512,
        tokenizer: Optional[GPT2Tokenizer] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Initialize the dataset.

        Args:
            split: Dataset split ("train" or "validation")
            max_seq_len: Maximum sequence length
            tokenizer: Tokenizer to use (default: GPT-2)
            max_samples: Maximum number of samples to load (for testing)
        """
        self.max_seq_len = max_seq_len

        # Load tokenizer
        if tokenizer is None:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer

        # Load dataset
        print(f"Loading TinyStories {split} split...")
        self.dataset = load_dataset("roneneldan/TinyStories", split=split)

        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))

        print(f"Loaded {len(self.dataset)} samples")

        # Pre-tokenize the dataset for efficiency
        self._tokenized = None

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        """Get a tokenized sample."""
        text = self.dataset[idx]["text"]

        # Tokenize - return numpy arrays to avoid storage resize issues
        # with multiprocessing DataLoader workers. The default collate_fn
        # will convert these to tensors.
        tokens = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
        )

        input_ids = np.array(tokens["input_ids"], dtype=np.int64)

        return {
            "input_ids": input_ids,
            "labels": input_ids.copy(),  # Use copy() for numpy arrays
        }


class StreamingTinyStories:
    """
    Streaming data loader for TinyStories.

    More memory efficient than loading entire dataset.
    Yields batches of packed sequences for efficient training.
    """

    def __init__(
        self,
        split: str = "train",
        max_seq_len: int = 512,
        batch_size: int = 32,
        buffer_size: int = 10000,
    ):
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.split = split

        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load dataset in streaming mode
        self.dataset = load_dataset(
            "roneneldan/TinyStories",
            split=split,
            streaming=True,
        )

    def __iter__(self) -> Iterator[dict]:
        """Yield batches of packed sequences."""
        buffer = []

        for sample in self.dataset:
            text = sample["text"]

            # Tokenize
            tokens = self.tokenizer.encode(text, add_special_tokens=True)

            # Add to buffer
            buffer.extend(tokens)
            buffer.append(self.tokenizer.eos_token_id)

            # Yield complete sequences
            while len(buffer) >= self.max_seq_len * self.batch_size:
                batch_tokens = []
                for _ in range(self.batch_size):
                    seq = buffer[:self.max_seq_len]
                    buffer = buffer[self.max_seq_len:]
                    batch_tokens.append(seq)

                input_ids = torch.tensor(batch_tokens, dtype=torch.long)
                yield {
                    "input_ids": input_ids,
                    "labels": input_ids.clone(),
                }


def create_dataloaders(
    batch_size: int = 32,
    max_seq_len: int = 512,
    num_workers: int = 4,
    max_train_samples: int = None,
    max_val_samples: int = 10000,
) -> tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders.

    Args:
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        num_workers: Number of data loading workers
        max_train_samples: Maximum training samples (None for all)
        max_val_samples: Maximum validation samples

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create shared tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Create datasets
    train_dataset = TinyStoriesDataset(
        split="train",
        max_seq_len=max_seq_len,
        tokenizer=tokenizer,
        max_samples=max_train_samples,
    )

    val_dataset = TinyStoriesDataset(
        split="validation",
        max_seq_len=max_seq_len,
        tokenizer=tokenizer,
        max_samples=max_val_samples,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def get_tokenizer() -> GPT2Tokenizer:
    """Get the GPT-2 tokenizer."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


if __name__ == "__main__":
    # Test data loading
    print("Testing TinyStories data loading\n")

    # Test standard dataset
    dataset = TinyStoriesDataset(split="train", max_samples=100)
    sample = dataset[0]
    print(f"Sample input_ids shape: {sample['input_ids'].shape}")

    # Test dataloader
    train_loader, val_loader = create_dataloaders(
        batch_size=4,
        max_train_samples=100,
        max_val_samples=50,
        num_workers=0,
    )

    batch = next(iter(train_loader))
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")

    # Decode a sample
    tokenizer = get_tokenizer()
    text = tokenizer.decode(batch["input_ids"][0][:100])
    print(f"\nSample text (first 100 tokens):\n{text[:200]}...")
