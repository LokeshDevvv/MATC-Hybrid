"""Gated Cross-Fusion (GCF) module for MATC-Net.

Dynamically fuses Mamba and Transformer branch outputs using a learned gate.
When gate ~ 1: model relies on Mamba (global context).
When gate ~ 0: model relies on Transformer (local semantics).
"""

import torch
import torch.nn as nn


class GatedCrossFusion(nn.Module):
    """Gated fusion of two encoder branch representations.

    H_fused = g * H_mamba + (1 - g) * H_transformer
    where g = sigmoid(W_g * [H_mamba ; H_transformer] + b_g)
    """

    def __init__(self, d_model: int = 768):
        super().__init__()
        self.gate_linear = nn.Linear(2 * d_model, d_model)

        # Store gate values for visualization
        self._last_gate_values = None

    def forward(self, h_mamba, h_transformer):
        """
        Args:
            h_mamba: (batch, d_model) — Mamba branch output
            h_transformer: (batch, d_model) — Transformer branch output
        Returns:
            h_fused: (batch, d_model) — gated fusion
        """
        combined = torch.cat([h_mamba, h_transformer], dim=-1)  # (B, 2*d_model)
        gate = torch.sigmoid(self.gate_linear(combined))  # (B, d_model), values in [0, 1]

        h_fused = gate * h_mamba + (1.0 - gate) * h_transformer  # (B, d_model)

        # Store for later visualization
        self._last_gate_values = gate.detach()

        return h_fused

    def get_last_gate_values(self):
        """Return gate values from the last forward pass."""
        return self._last_gate_values
