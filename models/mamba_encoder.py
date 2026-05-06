"""Mamba (Selective State Space Model) encoder branch for MATC-Net.

Provides O(n) linear-time global context modeling. If the mamba-ssm library
is available and CUDA is present, uses optimized Mamba blocks. Otherwise,
falls back to a pure-PyTorch selective SSM implementation using a parallel
associative scan for training efficiency.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import the optimized Mamba implementation
try:
    from mamba_ssm import Mamba as MambaBlock

    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Custom autograd scan for SSM recurrence (avoids deep autograd graph)
# ---------------------------------------------------------------------------

class _ScanFn(torch.autograd.Function):
    """Custom autograd for linear recurrence h_t = A_t * h_{t-1} + BX_t.

    Forward: sequential scan in O(L) steps (no autograd graph built).
    Backward: reverse-time sequential scan for gradients in O(L) steps.

    This is dramatically faster than letting PyTorch build and traverse a
    graph of L sequential operations per layer.
    """

    @staticmethod
    def forward(ctx, A_bar, BX):
        """
        Args:
            A_bar: (B, L, D, N) — discretized transition matrices
            BX:    (B, L, D, N) — discretized input contributions
        Returns:
            H: (B, L, D, N) — hidden states at each time step
        """
        B, L, D, N = A_bar.shape
        H = torch.empty(B, L, D, N, device=A_bar.device, dtype=A_bar.dtype)
        h = torch.zeros(B, D, N, device=A_bar.device, dtype=A_bar.dtype)
        for t in range(L):
            h = A_bar[:, t] * h + BX[:, t]
            H[:, t] = h
        ctx.save_for_backward(A_bar, H)
        return H

    @staticmethod
    def backward(ctx, grad_H):
        """
        Given dL/dH (B, L, D, N), compute dL/dA_bar and dL/dBX.

        Recurrence:  h_t = A_t * h_{t-1} + BX_t
        Gradients:
            dL/dBX_t  = dL/dh_t  (where dL/dh_t includes contributions from future steps)
            dL/dA_t   = dL/dh_t * h_{t-1}
            dL/dh_{t-1} += dL/dh_t * A_t   (backward propagation through time)
        """
        A_bar, H = ctx.saved_tensors
        B, L, D, N = A_bar.shape

        grad_A = torch.empty_like(A_bar)
        grad_BX = torch.empty_like(A_bar)

        # dL/dh_t accumulates future gradient contributions
        grad_h = torch.zeros(B, D, N, device=A_bar.device, dtype=A_bar.dtype)

        for t in range(L - 1, -1, -1):
            grad_h = grad_h + grad_H[:, t]          # add direct gradient
            grad_BX[:, t] = grad_h                   # dL/dBX_t = dL/dh_t
            if t > 0:
                h_prev = H[:, t - 1]
            else:
                h_prev = torch.zeros(B, D, N, device=A_bar.device, dtype=A_bar.dtype)
            grad_A[:, t] = grad_h * h_prev           # dL/dA_t = dL/dh_t * h_{t-1}
            grad_h = grad_h * A_bar[:, t]            # propagate backward

        return grad_A, grad_BX


def parallel_scan(A_bar, BX):
    """Scan for linear recurrence h_t = A_t * h_{t-1} + BX_t.

    Uses a custom autograd function to avoid building a deep computation
    graph (L operations per layer). Forward and backward are both O(L)
    sequential scans with no autograd overhead.

    Args:
        A_bar: (batch, seq_len, d_inner, d_state) — transition matrices
        BX:    (batch, seq_len, d_inner, d_state) — input contributions
    Returns:
        H: (batch, seq_len, d_inner, d_state) — hidden states at each step
    """
    return _ScanFn.apply(A_bar, BX)


# ---------------------------------------------------------------------------
# Fallback: Pure-PyTorch Selective SSM Block
# ---------------------------------------------------------------------------

class SelectiveSSM(nn.Module):
    """Selective State Space Model — input-dependent discretization.

    Uses parallel associative scan for training efficiency.
    """

    def __init__(self, d_inner, d_state=16):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        # A is log-parameterized for stability (initialized as HiPPO-like)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)
                       .unsqueeze(0).expand(d_inner, -1).clone())
        )  # (d_inner, d_state)

        # D is a residual connection parameter
        self.D = nn.Parameter(torch.ones(d_inner))

        # Input-dependent projections for Δ, B, C
        self.proj_delta = nn.Linear(d_inner, d_inner, bias=True)
        self.proj_B = nn.Linear(d_inner, d_state, bias=False)
        self.proj_C = nn.Linear(d_inner, d_state, bias=False)

        # Initialize delta bias to small positive values
        with torch.no_grad():
            self.proj_delta.bias.uniform_(0.001, 0.1)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_inner)
        Returns:
            y: (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = x.shape

        # Compute input-dependent parameters
        delta = F.softplus(self.proj_delta(x))  # (B, L, d_inner), positive
        B = self.proj_B(x)                       # (B, L, d_state)
        C = self.proj_C(x)                       # (B, L, d_state)

        # Get A from log parameterization
        A = -torch.exp(self.A_log)  # (d_inner, d_state), negative

        # Discretize using Zero-Order Hold (ZOH)
        # Ā = exp(Δ * A)
        delta_A = delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # (B, L, d_inner, d_state)
        A_bar = torch.exp(delta_A)  # (B, L, d_inner, d_state)

        # B̄ * x = Δ * B * x  (simplified ZOH)
        BX = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)  # (B, L, d_inner, d_state)

        # Parallel scan: compute all h_t in O(L log L) parallel steps
        H = parallel_scan(A_bar, BX)  # (B, L, d_inner, d_state)

        # Output: y_t = C_t · h_t
        y = (H * C.unsqueeze(2)).sum(dim=-1)  # (B, L, d_inner)

        # Add residual (D parameter)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y


class CustomMambaBlock(nn.Module):
    """Pure-PyTorch Mamba block with selective scan, conv1d, and SiLU gating."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand

        self.norm = nn.LayerNorm(d_model)

        # Input projection: d_model -> 2 * d_inner (for x and z branches)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # 1D convolution for local context
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )

        # Selective SSM
        self.ssm = SelectiveSSM(self.d_inner, d_state)

        # Output projection: d_inner -> d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)

        # Split into x and z branches
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_branch, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Conv1d on x branch
        x_conv = self.conv1d(x_branch.transpose(1, 2))[:, :, :x_branch.shape[1]]
        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # Selective SSM
        x_ssm = self.ssm(x_conv)  # (B, L, d_inner)

        # SiLU gating with z branch
        y = x_ssm * F.silu(z)

        # Output projection
        y = self.out_proj(y)

        return y + residual


# ---------------------------------------------------------------------------
# Mamba Encoder (stacks multiple blocks)
# ---------------------------------------------------------------------------

class MambaEncoder(nn.Module):
    """Mamba branch encoder: stack of Mamba blocks with mean pooling.

    Uses optimized mamba-ssm blocks when available, otherwise falls
    back to the pure-PyTorch CustomMambaBlock.
    """

    def __init__(self, config):
        """
        Args:
            config: namespace with d_model, d_state, d_conv, expand_factor, num_layers
        """
        super().__init__()
        d_model = config.d_model
        d_state = config.d_state
        d_conv = config.d_conv
        expand = config.expand_factor
        num_layers = config.num_layers

        if MAMBA_AVAILABLE and torch.cuda.is_available():
            print(f"MambaEncoder: using optimized mamba-ssm blocks ({num_layers} layers)")
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    MambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand),
                )
                for _ in range(num_layers)
            ])
            self.use_residual_wrapper = True
        else:
            if not MAMBA_AVAILABLE:
                print(f"MambaEncoder: mamba-ssm not available, using pure-PyTorch fallback ({num_layers} layers)")
            else:
                print(f"MambaEncoder: CUDA not available, using pure-PyTorch fallback ({num_layers} layers)")
            self.blocks = nn.ModuleList([
                CustomMambaBlock(d_model, d_state, d_conv, expand)
                for _ in range(num_layers)
            ])
            self.use_residual_wrapper = False

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: (batch, seq_len, d_model) — shared embeddings
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding
        Returns:
            pooled: (batch, d_model) — mean-pooled representation
        """
        h = x

        for block in self.blocks:
            if self.use_residual_wrapper:
                # Optimized Mamba block doesn't include residual
                h = h + block(h)
            else:
                # CustomMambaBlock already includes residual
                h = block(h)

        h = self.final_norm(h)  # (B, L, d_model)

        # Mean pooling over non-padded tokens
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
            h = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        else:
            h = h.mean(dim=1)

        return h  # (B, d_model)
