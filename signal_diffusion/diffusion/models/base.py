"""Base classes and registry for diffusion model factories."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL
from diffusers.training_utils import EMAModel
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
    ema: EMAModel | None = None
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
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Generate unconditional samples in ``[-1, 1]`` of shape ``(N, C, H, W)``."""

    def generate_conditional_samples(
        self,
        accelerator: Accelerator,
        cfg: DiffusionConfig,
        modules: DiffusionModules,
        num_images: int,
        *,
        denoising_steps: int,
        cfg_scale: float,
        conditioning: torch.Tensor | str | Iterable[str] | None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Generate samples using classifier-free guidance.

        ``conditioning`` may be ``None`` for unconditional sampling, a tensor of integer
        class labels with shape ``(N,)``, or a caption / iterable of captions when the
        adapter supports text conditioning.
        """

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


# Common utility functions shared across adapters


def create_noise_tensor(
    num_samples: int,
    cfg: DiffusionConfig,
    modules: DiffusionModules,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Create noise tensor for sampling, handling time-series shapes."""
    channels = cfg.model.extras.get("in_channels", 3)

    if cfg.settings and getattr(cfg.settings, "data_type", "") == "timeseries":
        n_eeg_channels = cfg.dataset.extras.get("n_eeg_channels")
        sequence_length = cfg.dataset.extras.get("sequence_length")
        if n_eeg_channels is None or sequence_length is None:
            raise ValueError(
                "Time-series config missing required extras: "
                f"n_eeg_channels={n_eeg_channels}, sequence_length={sequence_length}"
            )

        return torch.randn(
            (num_samples, channels, n_eeg_channels, sequence_length),
            generator=generator,
            device=device,
            dtype=dtype,
        )

    sample_size = cfg.model.sample_size
    if sample_size is None:
        sample_size = cfg.dataset.resolution

    return torch.randn(
        (num_samples, channels, sample_size, sample_size),
        generator=generator,
        device=device,
        dtype=dtype,
    )


def prepare_class_labels(
    batch: DiffusionBatch,
    *,
    device: torch.device,
    num_dataset_classes: int,
    cfg_dropout: float,
) -> torch.Tensor | None:
    """Prepare class labels with CFG dropout."""
    if num_dataset_classes <= 0:
        return None
    if batch.class_labels is None:
        raise ValueError("Dataset must provide class_labels for class-conditioned runs")
    labels = batch.class_labels.to(device=device, dtype=torch.long)
    if labels.numel() == 0:
        raise ValueError("Received empty class label batch")
    if labels.min() < 0 or labels.max() >= num_dataset_classes:
        raise ValueError(
            f"Class labels must be in [0, {num_dataset_classes - 1}]"
        )
    if cfg_dropout > 0:
        # Randomly replace labels with the dropout token so the model
        # learns unconditional predictions for CFG at training time.
        dropout_token = torch.full_like(labels, num_dataset_classes)
        mask = torch.rand_like(labels.float()) < cfg_dropout
        labels = torch.where(mask, dropout_token, labels)
    return labels


def resolve_checkpoint_candidates(checkpoint_path: Path) -> list[Path]:
    """Return ordered candidate checkpoints for paths/files supplied via config."""
    if checkpoint_path.is_dir():
        preferred_names = [
            "localmamba_model.pt",
            "hourglass_model.pt",
            "pytorch_model.bin",
            "diffusion_pytorch_model.bin",
            "model.bin",
            "model.pt",
            "model.pth",
            "model.safetensors",
        ]
        candidate_files = [checkpoint_path / name for name in preferred_names if (checkpoint_path / name).is_file()]
        if not candidate_files:
            candidate_files = sorted(
                [path for path in checkpoint_path.iterdir() if path.suffix in {".pt", ".pth", ".bin", ".safetensors"}]
            )
    else:
        candidate_files = [checkpoint_path]
    return candidate_files


def load_pretrained_weights(
    model: torch.nn.Module,
    source: str | Path,
    logger: Any,
    model_name: str = "model",
) -> None:
    """Load pretrained weights from checkpoint."""
    checkpoint_path = Path(source).expanduser()
    candidate_files = resolve_checkpoint_candidates(checkpoint_path)

    if not candidate_files:
        raise FileNotFoundError(f"Could not locate checkpoint file under {checkpoint_path}")

    checkpoint_file = candidate_files[0]
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint file {checkpoint_file} does not exist")

    if checkpoint_file.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file as load_safetensors  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                f"Loading safetensors checkpoint {checkpoint_file} requires 'safetensors' to be installed."
            ) from exc
        state_dict = load_safetensors(str(checkpoint_file))
    else:
        state_dict = torch.load(checkpoint_file, map_location="cpu")

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.warning(
            "Loaded checkpoint from %s with missing=%s unexpected=%s",
            checkpoint_file,
            missing,
            unexpected,
        )
    else:
        logger.info("Loaded %s weights from %s", model_name, checkpoint_file)


def finalize_generated_sample(
    sample: torch.Tensor,
    *,
    device: torch.device,
    vae: Any | None,
    latent_space: bool,
) -> torch.Tensor:
    """Convert latent samples back to pixel space if required."""
    if latent_space:
        if vae is None:
            raise RuntimeError("VAE expected for latent-space sampling")
        scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
        shift_factor = getattr(vae.config, "shift_factor", 0.0)
        sample = sample / scaling_factor + shift_factor
        with torch.no_grad():
            sample = vae.decode(sample.to(device=device, dtype=vae.dtype)).sample

    return sample.to(dtype=torch.float32).detach()


__all__ = [
    "DiffusionModules",
    "DiffusionAdapter",
    "registry",
    "create_noise_tensor",
    "prepare_class_labels",
    "resolve_checkpoint_candidates",
    "load_pretrained_weights",
    "finalize_generated_sample",
]
