"""Select the best compute device available to PyTorch."""

import torch


def get_device() -> torch.device:
    """Prefer CUDA, then MPS, and fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
