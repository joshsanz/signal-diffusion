"""Hourglass Diffusion Transformer (HDiT) adapter."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from accelerate import Accelerator
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm

from signal_diffusion.diffusion.config import DiffusionConfig
from signal_diffusion.diffusion.data import DiffusionBatch
from signal_diffusion.diffusion.models.base import DiffusionModules, registry
from signal_diffusion.log_setup import get_logger
from signal_diffusion.diffusion.train_utils import (
    apply_min_gamma_snr,
    get_sigmas_from_timesteps,
    get_snr,
    sample_timestep_logitnorm,
    verify_scheduler,
)
from .hourglass_model import flags
from .hourglass_model.hourglass_2d_transformer import (
    GlobalAttentionSpec,
    Hourglass2DModel,
    LevelSpec,
    MappingSpec,
    NeighborhoodAttentionSpec,
    NoAttentionSpec,
    ShiftedWindowAttentionSpec,
)


DEFAULT_NEIGHBORHOOD_ATTN: dict[str, Any] = {"type": "neighborhood", "d_head": 64, "kernel_size": 7}
DEFAULT_GLOBAL_ATTN: dict[str, Any] = {"type": "global", "d_head": 64}


@dataclass(slots=True)
class HourglassExtras:
    mapping_width: int = 256
    mapping_depth: int = 2
    mapping_d_ff: int | None = None
    mapping_cond_dim: int = 0
    mapping_dropout_rate: float = 0.0
    d_ffs: list[int] | None = None
    self_attns: list[Mapping[str, Any]] | None = None
    dropout_rate: float | list[float] | None = None
    cfg_dropout: float = 0.0
    # Apply Karras-style non-leaking augmentations, with conditioning
    augment_wrapper: bool = False
    augment_prob: float = 0.0 # Should be < ~0.8
    patch_size: list[int] = field(default_factory=lambda: [2, 2])
    depths: list[int] = field(default_factory=lambda: [2, 2, 4])
    widths: list[int] = field(default_factory=lambda: [128, 256, 512])
    in_channels: int = 3
    out_channels: int = 3
    latent_space: str | None = None


class HourglassAdapter:
    """Adapter instantiating the hourglass transformer denoiser."""

    name = "hourglass"

    def __init__(self) -> None:
        self._logger = get_logger(__name__)
        self._extras = HourglassExtras()
        self._num_dataset_classes = 0
        self._use_gradient_checkpointing = False

    def create_tokenizer(self, cfg: DiffusionConfig):  # noqa: D401 - protocol compliance
        return None

    def _parse_extras(self, cfg: DiffusionConfig) -> HourglassExtras:
        data = dict(cfg.model.extras)

        def _ensure_list(value: Any, *, length: int | None = None, item_type=float, name: str = "") -> list:
            if isinstance(value, (list, tuple)):
                result = [item_type(item) for item in value]
            elif value is None:
                if length is None:
                    raise ValueError(f"{name} requires an explicit value")
                result = [item_type(0) for _ in range(length)]
            else:
                result = [item_type(value) for _ in range(length or 1)]
            if length is not None and len(result) != length:
                raise ValueError(f"Expected {name} to have {length} entries, received {len(result)}")
            return result

        widths = _ensure_list(data.get("widths", [128, 256, 512]), item_type=int, name="widths")
        depths = _ensure_list(data.get("depths", [2, 2, 4]), item_type=int, length=len(widths), name="depths")

        mapping_width = int(data.get("mapping_width", 256))
        mapping_depth = int(data.get("mapping_depth", 2))
        mapping_d_ff_cfg = data.get("mapping_d_ff")
        mapping_d_ff = mapping_width * 3 if mapping_d_ff_cfg in (None, "", 0) else int(mapping_d_ff_cfg)
        mapping_cond_dim = int(data.get("mapping_cond_dim", 0))
        mapping_dropout_rate = float(data.get("mapping_dropout_rate", 0.0))

        d_ffs_cfg = data.get("d_ffs")
        if d_ffs_cfg is None:
            d_ffs = [mapping_d_ff] * len(widths)
        else:
            d_ffs = _ensure_list(d_ffs_cfg, length=len(widths), item_type=int, name="d_ffs")

        self_attns_cfg = data.get("self_attns")
        if self_attns_cfg is None:
            self_attns = []
            for idx in range(len(widths)):
                attn_cfg = DEFAULT_NEIGHBORHOOD_ATTN if idx < len(widths) - 1 else DEFAULT_GLOBAL_ATTN
                self_attns.append(attn_cfg)
        else:
            if len(self_attns_cfg) != len(widths):
                raise ValueError("self_attns length must match widths length")
            self_attns = [dict(spec) for spec in self_attns_cfg]

        dropout_cfg = data.get("dropout_rate")
        if isinstance(dropout_cfg, (int, float)):
            dropout_rate = [float(dropout_cfg)] * len(widths)
        elif dropout_cfg is None:
            dropout_rate = [0.0] * len(widths)
        else:
            dropout_rate = _ensure_list(dropout_cfg, length=len(widths), item_type=float, name="dropout_rate")

        patch_size_cfg = data.get("patch_size", [2, 2])
        if isinstance(patch_size_cfg, (int, float)):
            patch_size = [int(patch_size_cfg), int(patch_size_cfg)]
        else:
            patch_size = _ensure_list(patch_size_cfg, length=2, item_type=int, name="patch_size")

        augment_wrapper = bool(data.get("augment_wrapper", False))
        augment_prob = float(data.get("augment_prob", 0.0))
        latent_space = str(data.get("latent_space")) if "latent_space" in data else None

        extras = HourglassExtras(
            mapping_width=mapping_width,
            mapping_depth=mapping_depth,
            mapping_d_ff=mapping_d_ff,
            mapping_cond_dim=mapping_cond_dim,
            mapping_dropout_rate=mapping_dropout_rate,
            d_ffs=d_ffs,
            self_attns=self_attns,  # type: ignore
            dropout_rate=dropout_rate,
            augment_wrapper=augment_wrapper,
            augment_prob=augment_prob,
            patch_size=patch_size,
            depths=depths,
            widths=widths,
            in_channels=int(data.get("in_channels", 3)),
            out_channels=int(data.get("out_channels", 3)),
            latent_space=latent_space,
        )
        return extras

    def _create_noise_tensor(
        self,
        num_samples: int,
        cfg: DiffusionConfig,
        modules: DiffusionModules,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Create noise tensor for sampling, handling time-series shapes."""
        model_config = getattr(modules.denoiser, "config", None)
        channels = getattr(model_config, "in_channels", 3) if model_config is not None else 3

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

        sample_size = getattr(model_config, "sample_size", None)
        if sample_size is None:
            sample_size = int(cfg.model.sample_size or cfg.dataset.resolution)

        return torch.randn(
            (num_samples, channels, sample_size, sample_size),
            generator=generator,
            device=device,
            dtype=dtype,
        )

    def _build_attention_spec(self, spec: Mapping[str, Any]):
        attn_type = str(spec.get("type", "")).strip().lower()
        if attn_type == "global":
            return GlobalAttentionSpec(int(spec.get("d_head", 64)))
        if attn_type == "neighborhood":
            return NeighborhoodAttentionSpec(int(spec.get("d_head", 64)), int(spec.get("kernel_size", 7)))
        if attn_type == "shifted-window":
            if "window_size" not in spec:
                raise ValueError("shifted-window attention requires 'window_size'")
            return ShiftedWindowAttentionSpec(int(spec.get("d_head", 64)), int(spec["window_size"]))
        if attn_type == "none":
            return NoAttentionSpec()
        raise ValueError(f"Unsupported self attention type '{spec.get('type')}'")

    def _make_model(self, cfg: DiffusionConfig, extras: HourglassExtras) -> Hourglass2DModel:
        d_ffs = extras.d_ffs or [extras.mapping_width * 3] * len(extras.widths)
        attn_specs = extras.self_attns or []
        dropout_rates = (
            extras.dropout_rate
            if isinstance(extras.dropout_rate, list)
            else [float(extras.dropout_rate or 0.0)] * len(extras.widths)
        )

        levels = []
        for depth, width, d_ff, attn_spec, dropout in zip(
            extras.depths,
            extras.widths,
            d_ffs,
            attn_specs,
            dropout_rates,
        ):
            levels.append(
                LevelSpec(
                    depth=depth,
                    width=width,
                    d_ff=d_ff,
                    self_attn=self._build_attention_spec(attn_spec),
                    dropout=dropout,
                )
            )

        mapping = MappingSpec(
            depth=extras.mapping_depth,
            width=extras.mapping_width,
            d_ff=extras.mapping_d_ff or extras.mapping_width * 3,
            dropout=extras.mapping_dropout_rate,
        )

        num_classes = cfg.dataset.num_classes
        embedding_classes = num_classes + 1 if num_classes else 0
        mapping_cond_dim = extras.mapping_cond_dim + (9 if extras.augment_wrapper else 0)

        model = Hourglass2DModel(
            levels=levels,
            mapping=mapping,
            in_channels=extras.in_channels,
            out_channels=extras.out_channels,
            patch_size=tuple(extras.patch_size),
            num_classes=embedding_classes,
            mapping_cond_dim=mapping_cond_dim,
        )

        if extras.augment_wrapper:
            # model = augmentation.KarrasAugmentWrapper(model)
            raise ValueError("Karras augmentation not supported")

        if cfg.model.pretrained:
            self._load_pretrained_weights(model, cfg.model.pretrained)

        return model

    def _load_pretrained_weights(self, model: Hourglass2DModel, source: str | Path) -> None:
        checkpoint_path = Path(source).expanduser()
        candidate_files: list[Path]

        if checkpoint_path.is_dir():
            preferred_names = [
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
            self._logger.warning(
                "Loaded checkpoint from %s with missing=%s unexpected=%s",
                checkpoint_file,
                missing,
                unexpected,
            )
        else:
            self._logger.info("Loaded hourglass weights from %s", checkpoint_file)

    def build_modules(
        self,
        accelerator: Accelerator,
        cfg: DiffusionConfig,
        tokenizer=None,
    ) -> DiffusionModules:
        del tokenizer  # hourglass does not use text inputs
        self._extras = self._parse_extras(cfg)
        self._num_dataset_classes = cfg.dataset.num_classes
        self._use_gradient_checkpointing = bool(cfg.training.gradient_checkpointing)

        if accelerator.is_main_process:
            self._logger.info("Building Hourglass2DModel with extras=%s", self._extras)

        noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=cfg.objective.flow_match_timesteps)
        verify_scheduler(noise_scheduler)

        model = self._make_model(cfg, self._extras)

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        model.to(accelerator.device, dtype=weight_dtype)

        if self._use_gradient_checkpointing and accelerator.is_main_process:
            self._logger.info("Enabled gradient checkpointing for Hourglass denoiser")

        params = list(model.param_groups(base_lr=cfg.optimizer.learning_rate))
        modules = DiffusionModules(
            denoiser=model,
            noise_scheduler=noise_scheduler,
            weight_dtype=weight_dtype,
            parameters=params,
            clip_grad_norm_target=model.parameters(),
        )
        return modules

    def _prepare_class_labels(self, batch: DiffusionBatch, device: torch.device) -> torch.Tensor | None:
        if self._num_dataset_classes <= 0:
            return None
        if batch.class_labels is None:
            raise ValueError("Dataset must provide class_labels for class-conditioned hourglass runs")
        labels = batch.class_labels.to(device=device, dtype=torch.long)
        if labels.numel() == 0:
            raise ValueError("Received empty class label batch")
        if labels.min() < 0 or labels.max() >= self._num_dataset_classes:
            raise ValueError(
                f"Class labels must be in [0, {self._num_dataset_classes - 1}] for hourglass adapter"
            )
        if self._extras.cfg_dropout > 0:
            dropout_token = torch.full_like(labels, self._num_dataset_classes)
            mask = torch.rand_like(labels.float()) < self._extras.cfg_dropout
            labels = torch.where(mask, dropout_token, labels)
        return labels

    def _checkpoint_context(self):
        if self._use_gradient_checkpointing and torch.is_grad_enabled():
            return flags.checkpointing(True)
        return nullcontext()

    def training_step(
        self,
        accelerator: Accelerator,
        cfg: DiffusionConfig,
        modules: DiffusionModules,
        batch: DiffusionBatch,
    ) -> tuple[torch.Tensor, Mapping[str, float]]:
        model = modules.denoiser
        scheduler = modules.noise_scheduler
        extras = self._extras
        device = accelerator.device

        images = batch.pixel_values.to(device, dtype=modules.weight_dtype)
        if extras.latent_space:
            vae = modules.vae
            if vae is None:
                raise RuntimeError("VAE expected but not initialised")
            images = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor  # type: ignore

        class_labels = self._prepare_class_labels(batch, device=device)
        # TODO: text encoding

        noise = torch.randn_like(images)
        timesteps = sample_timestep_logitnorm(
            images.shape[0],
            num_train_timesteps=scheduler.config.num_train_timesteps,  # type: ignore
            timesteps=scheduler.timesteps,  # type: ignore
            device=device,
        )
        sigmas = get_sigmas_from_timesteps(scheduler, timesteps, device=device)
        z_t = scheduler.scale_noise(images, timesteps, noise)  # type: ignore
        snr = get_snr(scheduler, timesteps, device=device)

        if cfg.objective.prediction_type == "epsilon":
            target = noise
        elif cfg.objective.prediction_type == "vector_field":
            target = noise - images
        else:
            raise ValueError(f"Unsupported prediction type {cfg.objective.prediction_type} for DiT")

        # TODO: text cond
        with self._checkpoint_context():
            model_pred = model(z_t, sigma=sigmas, class_cond=class_labels)
        loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, loss.ndim)))
        weights = apply_min_gamma_snr(
            snr,
            timesteps=timesteps,
            gamma=cfg.training.snr_gamma,
            prediction_type=cfg.objective.prediction_type,
        )
        loss = (loss * weights).mean()
        metrics: Mapping[str, float] = {}
        return loss, metrics

    def validation_step(
        self,
        accelerator: Accelerator,
        cfg: DiffusionConfig,
        modules: DiffusionModules,
        batch: DiffusionBatch,
    ) -> Mapping[str, float]:
        loss, _ = self.training_step(accelerator, cfg, modules, batch)
        gathered = accelerator.gather_for_metrics(loss.detach())
        return {"loss": float(gathered.mean())}

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
        return self.generate_conditional_samples(
            accelerator,
            cfg,
            modules,
            num_images,
            denoising_steps=denoising_steps,
            cfg_scale=cfg_scale,
            conditioning=None,
            generator=generator,
        )

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
        scheduler: FlowMatchEulerDiscreteScheduler = FlowMatchEulerDiscreteScheduler.from_config(modules.noise_scheduler.config) # pyright: ignore[reportAssignmentType]
        device = accelerator.device
        dtype = modules.weight_dtype
        scheduler.set_timesteps(denoising_steps, device=device)

        model_config = getattr(modules.denoiser, "config", None)
        channels = getattr(model_config, "in_channels", 3) if model_config is not None else 3
        vae = modules.vae

        sample = self._create_noise_tensor(
            num_images,
            cfg,
            modules,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        dropout_label = self._num_dataset_classes

        if conditioning is None:
            class_labels = torch.full((num_images,), dropout_label, device=device, dtype=torch.long)
            unconditional_labels = None
            classifier_free = False
        elif isinstance(conditioning, torch.Tensor):
            classifier_free = True
            class_labels = conditioning.to(device=device, dtype=torch.long)
            if class_labels.ndim != 1:
                raise ValueError("Class-label conditioning tensor must be 1D with shape (num_images,)")
            if class_labels.shape[0] == 1 and num_images > 1:
                class_labels = class_labels.expand(num_images)
            if class_labels.shape[0] != num_images:
                raise ValueError("Number of class labels must match num_images")
            if class_labels.numel() == 0:
                raise ValueError("Class-label conditioning tensor must be non-empty")
            if class_labels.min() < 0 or class_labels.max() >= self._num_dataset_classes:
                raise ValueError(
                    f"Class labels must be in [0, {self._num_dataset_classes - 1}] for hourglass adapter"
                )
            unconditional_labels = torch.full_like(class_labels, dropout_label)
        elif isinstance(conditioning, str) or isinstance(conditioning, Iterable):
            raise NotImplementedError("Hourglass caption conditioning is not implemented yet")
        else:
            raise TypeError("Unsupported conditioning value for Hourglass sampling")

        with torch.no_grad():
            for timestep in tqdm(scheduler.timesteps, desc="Denoising", leave=False):
                if classifier_free and unconditional_labels is not None:
                    model_input = torch.cat([sample, sample], dim=0)
                else:
                    model_input = sample
                if hasattr(scheduler, "scale_model_input"):
                    model_input = scheduler.scale_model_input(model_input, timestep)

                timesteps = torch.full((model_input.size(0),), timestep, device=device)
                sigmas = get_sigmas_from_timesteps(scheduler, timesteps, device=device)

                if classifier_free and unconditional_labels is not None:
                    class_input = torch.cat([unconditional_labels, class_labels], dim=0)
                else:
                    class_input = class_labels

                with self._checkpoint_context():
                    model_output = modules.denoiser(
                        model_input, sigma=sigmas, class_cond=class_input
                    )

                if classifier_free and unconditional_labels is not None:
                    model_output_uncond, model_output_cond = model_output.chunk(2)
                    model_output = model_output_uncond + cfg_scale * (model_output_cond - model_output_uncond)

                step_output = scheduler.step(model_output, timestep, sample, return_dict=True)
                sample = step_output.prev_sample  # type: ignore

        if self._extras.latent_space:
            if vae is None:
                raise RuntimeError("VAE expected for latent-space DiT sampling")
            sample = sample / getattr(vae.config, "scaling_factor", 1.0)
            with torch.no_grad():
                sample = vae.decode(sample.to(device=device, dtype=vae.dtype)).sample

        return sample.to(dtype=torch.float32).detach()

    def save_checkpoint(
        self,
        accelerator: Accelerator,
        cfg: DiffusionConfig,
        modules: DiffusionModules,
        output_dir: str,
    ) -> None:
        del accelerator, cfg
        path = Path(output_dir) / "hourglass_model.pt"
        inner_model = getattr(modules.denoiser, "inner_model", modules.denoiser)
        torch.save(inner_model.state_dict(), path)
        self._logger.info("Saved hourglass checkpoint to %s", path)


registry.register(HourglassAdapter())
