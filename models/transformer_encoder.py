"""Transformer branch encoder for MATC-Net.

Uses Multi-Head Self-Attention for fine-grained local semantic relationships.
Implemented with PyTorch's nn.TransformerEncoder for clean, maintainable code.
"""

import torch
import torch.nn as nn


class TransformerBranchEncoder(nn.Module):
    """Transformer encoder branch with [CLS]-token pooling.

    Architecture:
        - Stack of N TransformerEncoderLayers (pre-norm, GELU, 12 heads)
        - [CLS] token representation used as output
    """

    def __init__(self, config):
        """
        Args:
            config: namespace with d_model, num_heads, d_ff, dropout, num_layers
        """
        super().__init__()
        d_model = config.d_model
        nhead = config.num_heads
        d_ff = config.d_ff
        dropout = config.dropout
        num_layers = config.num_layers

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.final_norm = nn.LayerNorm(d_model)

        print(f"TransformerBranchEncoder: {num_layers} layers, {nhead} heads, d_model={d_model}")

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: (batch, seq_len, d_model) — shared embeddings
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding
        Returns:
            pooled: (batch, d_model) — [CLS] token representation
        """
        # Convert attention_mask to src_key_padding_mask
        # TransformerEncoder expects True for positions to IGNORE
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)  # True = pad
        else:
            src_key_padding_mask = None

        h = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        h = self.final_norm(h)

        # Use first token ([CLS]) as pooled representation
        pooled = h[:, 0, :]  # (B, d_model)

        return pooled
