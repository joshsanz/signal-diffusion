"""Base classes and registry for diffusion model factories."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Protocol

import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from transformers import PreTrainedTokenizerBase, CLIPTextModel

from signal_diffusion.diffusion.config import DiffusionConfig
from signal_diffusion.diffusion.data import DiffusionBatch


@dataclass(slots=True)
class DiffusionModules:
    """Container for model components participating in training."""

    denoiser: torch.nn.Module
    noise_scheduler: SchedulerMixin
    weight_dtype: torch.dtype
    vae: AutoencoderKL | None = None
    text_encoder: CLIPTextModel | None = None
    tokenizer: PreTrainedTokenizerBase | None = None
    extra_conditioning: Mapping[str, torch.Tensor] | None = None
    ema: torch.nn.Module | None = None
    parameters: Iterable[torch.nn.Parameter] = field(default_factory=list)
    clip_grad_norm_target: Iterable[torch.nn.Parameter] | None = None


class DiffusionAdapter(Protocol):
    """Interface for model-specific diffusion training hooks."""

    name: str

    def create_tokenizer(self, cfg: DiffusionConfig) -> PreTrainedTokenizerBase | None:
        """Load tokenizer used to prepare text conditioning, if any."""

    def build_modules(
        self,
        accelerator: Accelerator,
        cfg: DiffusionConfig,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> DiffusionModules:
        """Instantiate model components and return :class:`DiffusionModules`."""

    def training_step(
        self,
        accelerator: Accelerator,
        cfg: DiffusionConfig,
        modules: DiffusionModules,
        batch: DiffusionBatch,
    ) -> tuple[torch.Tensor, Mapping[str, float]]:
        """Compute loss for a single training batch."""

    def validation_step(
        self,
        accelerator: Accelerator,
        cfg: DiffusionConfig,
        modules: DiffusionModules,
        batch: DiffusionBatch,
    ) -> Mapping[str, float]:
        """Run validation for a batch and return scalar metrics."""

    def generate_samples(
        self,
        accelerator: Accelerator,
        cfg: DiffusionConfig,
        modules: DiffusionModules,
        num_images: int,
        *,
        denoising_steps: int,
        cfg_scale: float,
    ) -> torch.Tensor:
        """Generate unconditional samples in ``[-1, 1]`` of shape ``(N, C, H, W)``."""

    def save_checkpoint(
        self,
        accelerator: Accelerator,
        cfg: DiffusionConfig,
        modules: DiffusionModules,
        output_dir: str,
    ) -> None:
        """Persist model weights to ``output_dir``."""


class AdapterRegistry:
    """Simple registry mapping adapter names to implementations."""

    def __init__(self) -> None:
        self._registry: dict[str, DiffusionAdapter] = {}

    def register(self, adapter: DiffusionAdapter) -> None:
        if adapter.name in self._registry:
            raise KeyError(f"Diffusion adapter '{adapter.name}' already registered")
        self._registry[adapter.name] = adapter

    def get(self, name: str) -> DiffusionAdapter:
        try:
            return self._registry[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown diffusion model '{name}'. Available: {sorted(self._registry)}"
            ) from exc

    def names(self) -> list[str]:
        return sorted(self._registry)


registry = AdapterRegistry()

__all__ = ["DiffusionModules", "DiffusionAdapter", "registry"]
