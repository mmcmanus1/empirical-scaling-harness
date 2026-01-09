"""
Minimal decoder-only transformer with configurable activation.

Supports GeLU and SwiGLU activations for scaling law experiments.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig


class SwiGLU(nn.Module):
    """
    SwiGLU activation: (x @ W_gate * swish(x @ W_up)) @ W_down

    Uses 3 linear projections instead of 2, so d_ff should be reduced
    to keep parameter count equivalent to standard FFN.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))  # swish activation
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))


class GeLUFFN(nn.Module):
    """Standard FFN with GeLU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w_up = nn.Linear(d_model, d_ff)
        self.w_down = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.gelu(self.w_up(x))))


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.d_model = config.d_model

        # Combined QKV projection
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=-1)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)

        if config.activation == "swiglu":
            self.ffn = SwiGLU(config.d_model, config.d_ff, config.dropout)
        else:
            self.ffn = GeLUFFN(config.d_model, config.d_ff, config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class Transformer(nn.Module):
    """Decoder-only transformer for language modeling."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # Output projection (weight-tied with token embeddings)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        # Initialize weights
        self.apply(self._init_weights)

        # Scale residual projections
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, std=0.02 / math.sqrt(2 * config.n_layers))
            if hasattr(block.ffn, 'w_down'):
                nn.init.normal_(block.ffn.w_down.weight, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs, shape (batch, seq_len)
            labels: Target token IDs for loss computation

        Returns:
            logits: Output logits, shape (batch, seq_len, vocab_size)
            loss: Cross-entropy loss if labels provided, else None
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Get embeddings
        pos = torch.arange(0, T, device=device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift so we predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            idx_cond = input_ids[:, -self.config.max_seq_len:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


def build_model(config: ModelConfig) -> Transformer:
    """Build a transformer model from config."""
    return Transformer(config)


if __name__ == "__main__":
    from config import build_config

    # Test model creation
    print("Testing model creation\n")

    for activation in ["gelu", "swiglu"]:
        print(f"\n{activation.upper()} Model:")
        config = build_config(3_000_000, activation=activation)
        model = build_model(config)

        # Count parameters
        total_params = model.count_parameters()
        print(f"  Config params: {config.count_parameters():,}")
        print(f"  Model params:  {total_params:,}")

        # Test forward pass
        batch = torch.randint(0, config.vocab_size, (2, 128))
        logits, loss = model(batch, labels=batch)
        print(f"  Forward pass OK - logits shape: {logits.shape}")
        print(f"  Loss: {loss.item():.4f}")
