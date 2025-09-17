"""Top-level package for Signal Diffusion utilities."""
from __future__ import annotations

from importlib import metadata

__all__ = ["__version__"]


def _detect_version() -> str:
    try:
        return metadata.version("signal-diffusion")
    except metadata.PackageNotFoundError:
        # Fallback for editable / non-installed usage
        return "0.0.0"


__version__ = _detect_version()
