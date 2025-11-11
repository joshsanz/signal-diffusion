"""
Feature flags for LocalMamba model.

Controls optional features like gradient checkpointing, compilation, etc.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import threading
from typing import Optional


@dataclass
class LocalMambaFlags:
    """Configuration flags for LocalMamba model."""

    # Gradient checkpointing to save memory
    gradient_checkpointing: bool = False

    # PyTorch 2.0 compilation
    compile: bool = False

    # Mixed precision training
    use_fp16: bool = False
    use_bf16: bool = False

    # Scan directions (e.g., ["h", "v", "w7"])
    scan_directions: Optional[list[str]] = None

    # Use BiAttn for merging multi-scan results
    use_biattr: bool = True

    # Simple initialization (faster but potentially less stable)
    simple_init: bool = False

    def __post_init__(self):
        """Validate flags."""
        if self.scan_directions is None:
            # Default to h, v, w7
            self.scan_directions = ["h", "v", "w7"]

        if self.use_fp16 and self.use_bf16:
            raise ValueError("Cannot use both fp16 and bf16")


# Thread-local state so checkpointing can be toggled per forward pass.
_state = threading.local()
_state.checkpointing = False


@contextmanager
def checkpointing(enable: bool = True):
    """Temporarily enable gradient checkpointing (mirrors hourglass flags)."""
    try:
        prev, _state.checkpointing = getattr(_state, "checkpointing", False), enable
        yield
    finally:
        _state.checkpointing = prev


def get_checkpointing() -> bool:
    """Return current checkpointing flag."""
    return getattr(_state, "checkpointing", False)


# Default flags
DEFAULT_FLAGS = LocalMambaFlags()
