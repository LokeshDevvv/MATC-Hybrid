"""Benchmark: custom autograd scan (forward + backward with gradients)."""

import time
import torch
from models.mamba_encoder import parallel_scan, MambaEncoder
from utils.helpers import load_config


def test_correctness():
    """Verify gradients by comparing with autograd sequential scan."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, L, D, N = 4, 32, 64, 8
    A_bar = (torch.rand(B, L, D, N, device=device) * 0.9).requires_grad_(True)
    BX = (torch.randn(B, L, D, N, device=device) * 0.1).requires_grad_(True)

    # Custom autograd scan
    H_custom = parallel_scan(A_bar, BX)
    loss_custom = H_custom.sum()
    loss_custom.backward()
    grad_A_custom = A_bar.grad.clone()
    grad_BX_custom = BX.grad.clone()

    # Reference: autograd sequential scan
    A_bar.grad = None
    BX.grad = None
    h = torch.zeros(B, D, N, device=device)
    outputs = []
    for t in range(L):
        h = A_bar[:, t] * h + BX[:, t]
        outputs.append(h)
    H_ref = torch.stack(outputs, dim=1)
    loss_ref = H_ref.sum()
    loss_ref.backward()

    fwd_diff = (H_custom - H_ref).abs().max().item()
    grad_A_diff = (grad_A_custom - A_bar.grad).abs().max().item()
    grad_BX_diff = (grad_BX_custom - BX.grad).abs().max().item()

    print(f"  Forward  max_diff: {fwd_diff:.2e}")
    print(f"  Grad A   max_diff: {grad_A_diff:.2e}")
    print(f"  Grad BX  max_diff: {grad_BX_diff:.2e}")
    ok = fwd_diff < 1e-5 and grad_A_diff < 1e-5 and grad_BX_diff < 1e-5
    print(f"  Status: {'PASS' if ok else 'FAIL'}")
    return ok


def benchmark_encoder_with_grad(batch_size, seq_len, device, n_warmup=2, n_runs=5):
    """Benchmark full Mamba encoder with forward + backward (training mode)."""
    config = load_config()
    encoder = MambaEncoder(config.mamba).to(device)
    encoder.train()

    x = torch.randn(batch_size, seq_len, config.mamba.d_model, device=device, requires_grad=True)
    mask = torch.ones(batch_size, seq_len, device=device)

    # Warmup
    for _ in range(n_warmup):
        out = encoder(x, mask)
        loss = out.sum()
        loss.backward()
        encoder.zero_grad()
        if x.grad is not None:
            x.grad = None
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Benchmark forward + backward
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        out = encoder(x, mask)
        loss = out.sum()
        loss.backward()
        encoder.zero_grad()
        if x.grad is not None:
            x.grad = None
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_runs


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # --- Correctness ---
    print("=" * 60)
    print("GRADIENT CORRECTNESS CHECK")
    print("=" * 60)
    test_correctness()

    # --- Encoder with gradients (training mode) ---
    print()
    print("=" * 60)
    print("MAMBA ENCODER: FORWARD + BACKWARD (training mode)")
    print("=" * 60)
    for bs, sl in [(32, 128), (16, 128), (8, 128)]:
        t = benchmark_encoder_with_grad(bs, sl, device)
        sps = bs / t
        print(f"  B={bs:>2} L={sl:>3} | {t*1000:>8.1f}ms (fwd+bwd) | {sps:>5.1f} samples/sec")

    # --- Training estimate ---
    print()
    print("=" * 60)
    print("TRAINING ESTIMATE (B=32, L=128)")
    print("=" * 60)
    t = benchmark_encoder_with_grad(32, 128, device, n_warmup=3, n_runs=5)
    batches = 3187
    mamba_epoch = t * batches
    # Full model: ~2x Mamba (add Transformer + classifier + overhead)
    full_epoch = mamba_epoch * 2.5
    print(f"  Mamba fwd+bwd:       {t*1000:.1f}ms/batch")
    print(f"  Mamba per epoch:     {mamba_epoch/60:.1f}min")
    print(f"  Full model estimate: ~{full_epoch/60:.0f}min/epoch")
    print(f"  30 epochs:           ~{full_epoch*30/3600:.1f}hrs")
