"""Model registry for diffusion training."""

from .base import DiffusionAdapter, DiffusionModules, registry
from . import dit as _dit  # noqa: F401  # side-effect registration
from . import hourglass as _hourglass  # noqa: F401  # side-effect registration
from . import localmamba as _localmamba  # noqa: F401  # side-effect registration
from . import stable_diffusion as _stable_diffusion  # noqa: F401  # side-effect registration
from . import stable_diffusion_35 as _stable_diffusion_35  # noqa: F401  # side-effect registration

__all__ = ["DiffusionAdapter", "DiffusionModules", "registry"]
