"""
LocalMamba diffusion model adapter.

Bridges the LocalVMamba hierarchical selective-scan architecture with the
DiffusionAdapter protocol, providing DiT/Hourglass-compatible training hooks,
conditioning, and scheduler wiring.
"""
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
from signal_diffusion.diffusion.models.base import (
    DiffusionModules,
    create_noise_tensor,
    finalize_generated_sample,
    load_pretrained_weights,
    prepare_class_labels,
    registry,
)
from signal_diffusion.log_setup import get_logger
from signal_diffusion.diffusion.train_utils import (
    apply_min_gamma_snr,
    get_sigmas_from_timesteps,
    get_snr,
    sample_timestep_logitnorm,
    verify_scheduler,
)
from .localmamba_model import flags
from .localmamba_model.localmamba_2d import (
    LocalMamba2DModel,
    LevelSpec,
    MappingSpec,
)


@dataclass(slots=True)
class LocalMambaExtras:
    """Configuration for LocalMamba diffusion model."""
    depths: list[int] = field(default_factory=lambda: [2, 2, 9, 2])
    dims: list[int] = field(default_factory=lambda: [96, 192, 384, 768])
    d_state: int = 16
    ssm_ratio: float = 2.0
    ssm_dt_rank: str | int = "auto"
    mlp_ratio: float = 4.0
    scan_directions: list[str] = field(default_factory=lambda: ["h", "v", "w7"])
    mapping_width: int = 256
    mapping_depth: int = 2
    mapping_d_ff: int | None = None
    mapping_cond_dim: int = 0
    mapping_dropout_rate: float = 0.0
    patch_size: list[int] = field(default_factory=lambda: [2, 2])
    in_channels: int = 3
    out_channels: int = 3
    dropout_rate: float | list[float] | None = None
    drop_path_rate: float = 0.0
    cfg_dropout: float = 0.0
    augment_wrapper: bool = False
    augment_prob: float = 0.0
    latent_space: str | None = None


class LocalMambaAdapter:
    """Adapter instantiating the LocalMamba denoiser."""

    name = "localmamba"

    def __init__(self) -> None:
        self._logger = get_logger(__name__)
        self._extras = LocalMambaExtras()
        self._num_dataset_classes = 0
        self._use_gradient_checkpointing = False

    def create_tokenizer(self, cfg: DiffusionConfig):  # noqa: D401 - protocol compliance
        """Create tokenizer (None for LocalMamba, which uses class conditioning only)."""
        return None

    def _parse_extras(self, cfg: DiffusionConfig) -> LocalMambaExtras:
        """Parse extras from config."""
        data = dict(cfg.model.extras)

        def _ensure_list(value: Any, *, length: int | None = None, item_type=float, name: str = "") -> list:
            """Normalize scalar/sequence config entries into a list of fixed length."""
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

        # Parse hierarchical architecture
        dims = _ensure_list(data.get("dims", [96, 192, 384, 768]), item_type=int, name="dims")
        depths = _ensure_list(data.get("depths", [2, 2, 9, 2]), item_type=int, length=len(dims), name="depths")

        # Parse mapping network config
        mapping_width = int(data.get("mapping_width", 256))
        mapping_depth = int(data.get("mapping_depth", 2))
        mapping_d_ff_cfg = data.get("mapping_d_ff")
        mapping_d_ff = mapping_width * 3 if mapping_d_ff_cfg in (None, "", 0) else int(mapping_d_ff_cfg)
        mapping_cond_dim = int(data.get("mapping_cond_dim", 0))
        mapping_dropout_rate = float(data.get("mapping_dropout_rate", 0.0))

        # Parse SSM parameters
        d_state = int(data.get("d_state", 16))
        ssm_ratio = float(data.get("ssm_ratio", 2.0))
        ssm_dt_rank_cfg = data.get("ssm_dt_rank", "auto")
        ssm_dt_rank = ssm_dt_rank_cfg if isinstance(ssm_dt_rank_cfg, str) else int(ssm_dt_rank_cfg)
        mlp_ratio = float(data.get("mlp_ratio", 4.0))

        # Parse scan directions
        scan_directions_cfg = data.get("scan_directions", ["h", "v", "w7"])
        if isinstance(scan_directions_cfg, str):
            scan_directions = [scan_directions_cfg]
        else:
            scan_directions = list(scan_directions_cfg)

        # Parse dropout rates
        dropout_cfg = data.get("dropout_rate")
        if isinstance(dropout_cfg, (int, float)):
            dropout_rate = [float(dropout_cfg)] * len(dims)
        elif dropout_cfg is None:
            dropout_rate = [0.0] * len(dims)
        else:
            dropout_rate = _ensure_list(dropout_cfg, length=len(dims), item_type=float, name="dropout_rate")

        drop_path_rate = float(data.get("drop_path_rate", 0.0))

        # Parse patch size
        patch_size_cfg = data.get("patch_size", [2, 2])
        if isinstance(patch_size_cfg, (int, float)):
            patch_size = [int(patch_size_cfg), int(patch_size_cfg)]
        else:
            patch_size = _ensure_list(patch_size_cfg, length=2, item_type=int, name="patch_size")

        # Parse augmentation settings
        augment_wrapper = bool(data.get("augment_wrapper", False))
        augment_prob = float(data.get("augment_prob", 0.0))
        latent_space = str(data.get("latent_space")) if "latent_space" in data else None

        extras = LocalMambaExtras(
            depths=depths,
            dims=dims,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            ssm_dt_rank=ssm_dt_rank,
            mlp_ratio=mlp_ratio,
            scan_directions=scan_directions,
            mapping_width=mapping_width,
            mapping_depth=mapping_depth,
            mapping_d_ff=mapping_d_ff,
            mapping_cond_dim=mapping_cond_dim,
            mapping_dropout_rate=mapping_dropout_rate,
            patch_size=patch_size,
            in_channels=int(data.get("in_channels", 3)),
            out_channels=int(data.get("out_channels", 3)),
            dropout_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
            cfg_dropout=float(data.get("cfg_dropout", 0.0)),
            augment_wrapper=augment_wrapper,
            augment_prob=augment_prob,
            latent_space=latent_space,
        )
        return extras


    def _make_model(self, cfg: DiffusionConfig, extras: LocalMambaExtras) -> LocalMamba2DModel:
        """Construct LocalMamba2DModel from config and extras."""
        dropout_rates = (
            extras.dropout_rate
            if isinstance(extras.dropout_rate, list)
            else [float(extras.dropout_rate or 0.0)] * len(extras.dims)
        )

        # Compute stochastic depth rates (linearly increasing)
        num_levels = len(extras.depths)
        total_blocks = sum(extras.depths)
        dpr = [extras.drop_path_rate * i / (total_blocks - 1) if total_blocks > 1 else 0.0
               for i in range(total_blocks)]

        # Distribute drop_path rates across levels
        levels = []
        block_idx = 0
        for depth, dim, dropout in zip(extras.depths, extras.dims, dropout_rates):
            level_dprs = dpr[block_idx:block_idx + depth]
            # Use average drop_path rate for this level
            avg_dpr = sum(level_dprs) / len(level_dprs) if level_dprs else 0.0

            levels.append(
                LevelSpec(
                    depth=depth,
                    width=dim,
                    d_state=extras.d_state,
                    ssm_ratio=extras.ssm_ratio,
                    ssm_dt_rank=extras.ssm_dt_rank,
                    drop_path=avg_dpr,
                    dropout=dropout,
                    directions=extras.scan_directions,
                )
            )
            block_idx += depth

        mapping = MappingSpec(
            depth=extras.mapping_depth,
            width=extras.mapping_width,
            d_ff=extras.mapping_d_ff or extras.mapping_width * 3,
            dropout=extras.mapping_dropout_rate,
        )

        num_classes = cfg.dataset.num_classes
        embedding_classes = num_classes + 1 if num_classes else 0
        mapping_cond_dim = extras.mapping_cond_dim + (9 if extras.augment_wrapper else 0)

        model = LocalMamba2DModel(
            levels=levels,
            mapping=mapping,
            in_channels=extras.in_channels,
            out_channels=extras.out_channels,
            patch_size=tuple(extras.patch_size),
            num_classes=embedding_classes,
            mapping_cond_dim=mapping_cond_dim,
        )

        if extras.augment_wrapper:
            raise ValueError("Karras augmentation not supported yet for LocalMamba")

        if cfg.model.pretrained:
            load_pretrained_weights(model, cfg.model.pretrained, self._logger, "LocalMamba")

        return model


    def build_modules(
        self,
        accelerator: Accelerator,
        cfg: DiffusionConfig,
        tokenizer=None,
    ) -> DiffusionModules:
        """Build diffusion modules (model + scheduler)."""
        del tokenizer  # LocalMamba does not use text inputs
        self._extras = self._parse_extras(cfg)
        self._num_dataset_classes = cfg.dataset.num_classes
        self._use_gradient_checkpointing = bool(cfg.training.gradient_checkpointing)

        if accelerator.is_main_process:
            self._logger.info("Building LocalMamba2DModel with extras=%s", self._extras)

        noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=cfg.objective.flow_match_timesteps)
        verify_scheduler(noise_scheduler)

        model = self._make_model(cfg, self._extras)

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        if accelerator.is_main_process:
            self._logger.info(f"Moving model to device: {accelerator.device}, dtype: {weight_dtype}")

        model = model.to(accelerator.device, dtype=weight_dtype)

        if self._use_gradient_checkpointing and accelerator.is_main_process:
            self._logger.info("Enabled gradient checkpointing for LocalMamba denoiser")

        params = list(model.param_groups(base_lr=cfg.optimizer.learning_rate))
        modules = DiffusionModules(
            denoiser=model,
            noise_scheduler=noise_scheduler,
            weight_dtype=weight_dtype,
            parameters=params,
            clip_grad_norm_target=model.parameters(),
        )
        return modules


    def _checkpoint_context(self):
        """Context manager for gradient checkpointing."""
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
        """Compute training loss for a batch."""
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

        class_labels = prepare_class_labels(
            batch, device=device, num_dataset_classes=self._num_dataset_classes, cfg_dropout=self._extras.cfg_dropout
        )
        # TODO: text encoding for caption conditioning

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

        # Map scheduler target semantics to the requested prediction type.
        if cfg.objective.prediction_type == "epsilon":
            target = noise
        elif cfg.objective.prediction_type == "vector_field":
            target = noise - images
        else:
            raise ValueError(f"Unsupported prediction type {cfg.objective.prediction_type} for LocalMamba")

        # Forward pass with gradient checkpointing if enabled
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

        # Compute PSNR for timeseries datasets
        metrics: dict[str, float] = {}

        # Check if this is timeseries data (not latent space)
        is_timeseries = bool(
            cfg.settings
            and getattr(cfg.settings, "data_type", "") == "timeseries"
            and not extras.latent_space
        )

        if is_timeseries:
            from signal_diffusion.diffusion.train_utils import compute_training_psnr

            psnr_result = compute_training_psnr(
                images=images,
                z_t=z_t,
                model_pred=model_pred,
                sigmas=sigmas,
                prediction_type=cfg.objective.prediction_type,
                max_value=1.0,
            )

            if psnr_result is not None:
                mean_psnr, std_psnr = psnr_result
                metrics["psnr_mean"] = mean_psnr
                metrics["psnr_std"] = std_psnr

        return loss, metrics

    def validation_step(
        self,
        accelerator: Accelerator,
        cfg: DiffusionConfig,
        modules: DiffusionModules,
        batch: DiffusionBatch,
    ) -> Mapping[str, float]:
        """Compute validation loss."""
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
        """Generate unconditional samples."""
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
        """Generate conditional samples with classifier-free guidance."""
        scheduler: FlowMatchEulerDiscreteScheduler = FlowMatchEulerDiscreteScheduler.from_config(modules.noise_scheduler.config)  # pyright: ignore[reportAssignmentType]
        device = accelerator.device
        dtype = modules.weight_dtype
        scheduler.set_timesteps(denoising_steps, device=device)

        vae = modules.vae

        sample = create_noise_tensor(
            num_images,
            cfg,
            modules,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        dropout_label = self._num_dataset_classes

        class_labels, unconditional_labels, classifier_free = self._prepare_sampling_conditioning(
            conditioning=conditioning,
            num_images=num_images,
            dropout_label=dropout_label,
            device=device,
        )

        with torch.no_grad():
            for timestep in tqdm(scheduler.timesteps, desc="Denoising", leave=False):
                # Duplicate batch when classifier-free guidance is active so both
                # unconditional and conditional predictions share the same noise.
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

        return finalize_generated_sample(sample, device=device, vae=vae, latent_space=bool(self._extras.latent_space))

    def _prepare_sampling_conditioning(
        self,
        *,
        conditioning: torch.Tensor | str | Iterable[str] | None,
        num_images: int,
        dropout_label: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None, bool]:
        """Normalize sampling-time conditioning inputs.

        Returns:
            class_labels: labels fed to the denoiser
            unconditional_labels: dropout tokens for CFG (or None if unused)
            classifier_free: whether classifier-free guidance is active
        """
        if conditioning is None:
            class_labels = torch.full((num_images,), dropout_label, device=device, dtype=torch.long)
            return class_labels, None, False

        if isinstance(conditioning, torch.Tensor):
            class_labels = conditioning.to(device=device, dtype=torch.long)
            if class_labels.ndim != 1:
                raise ValueError("Class-label conditioning tensor must be 1D with shape (num_images,)")
            if class_labels.shape[0] == 1 and num_images > 1:
                class_labels = class_labels.expand(num_images)
            if class_labels.shape[0] != num_images:
                raise ValueError("Number of class labels must match num_images")
            if class_labels.numel() == 0:
                raise ValueError("Class-label conditioning must be non-empty")
            if class_labels.min() < 0 or class_labels.max() >= self._num_dataset_classes:
                raise ValueError(
                    f"Class labels must be in [0, {self._num_dataset_classes - 1}] for LocalMamba adapter"
                )
            unconditional_labels = torch.full_like(class_labels, dropout_label)
            return class_labels, unconditional_labels, True

        if isinstance(conditioning, (str, Iterable)):
            raise NotImplementedError("LocalMamba caption conditioning is not implemented yet")

        raise TypeError("Unsupported conditioning value for LocalMamba sampling")


    def save_checkpoint(
        self,
        accelerator: Accelerator,
        cfg: DiffusionConfig,
        modules: DiffusionModules,
        output_dir: str,
    ) -> None:
        """Save model checkpoint."""
        del accelerator, cfg
        path = Path(output_dir) / "localmamba_model.pt"
        inner_model = getattr(modules.denoiser, "inner_model", modules.denoiser)
        torch.save(inner_model.state_dict(), path)
        self._logger.info("Saved LocalMamba checkpoint to %s", path)


registry.register(LocalMambaAdapter())
