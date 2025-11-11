"""Model registry for diffusion training."""

from .base import DiffusionAdapter, DiffusionModules, registry
from . import dit
from . import hourglass
from . import localmamba
from . import stable_diffusion

__all__ = ["DiffusionAdapter", "DiffusionModules", "registry"]