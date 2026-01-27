"""Random seed utilities for reproducibility."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_random_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility across NumPy, PyTorch, and Python's random module.

    Args:
        seed: Random seed value

    Example:
        >>> set_random_seeds(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Also set CUDA seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
