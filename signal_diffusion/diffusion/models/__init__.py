"""Model registry for diffusion training."""
from .base import DiffusionAdapter, DiffusionModules, registry
from . import stable_diffusion  # noqa: F401 - register adapter
from . import dit  # noqa: F401 - register adapter

__all__ = ["DiffusionAdapter", "DiffusionModules", "registry"]
