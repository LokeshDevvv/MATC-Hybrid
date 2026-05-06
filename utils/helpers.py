"""Utility functions for MATC-Net."""

import os
import random
import yaml
import torch
import numpy as np
from types import SimpleNamespace


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _coerce_numeric(v):
    """Try to convert string values to numeric types."""
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass
    return v


def _dict_to_namespace(d):
    """Recursively convert a dictionary to SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    return _coerce_numeric(d)


def load_config(config_path: str = "config/config.yaml") -> SimpleNamespace:
    """Load configuration from YAML file and return as nested namespace."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return _dict_to_namespace(config_dict)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save a training checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, path)


def load_checkpoint(model, optimizer, path, device):
    """Load a training checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint.get("metrics", {})
