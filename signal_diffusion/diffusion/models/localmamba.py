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
from diffusers import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm

from signal_diffusion.diffusion.config import DiffusionConfig
from signal_diffusion.diffusion.data import DiffusionBatch
from signal_diffusion.diffusion.guidance import apply_cfg_guidance
from signal_diffusion.diffusion.models.base import (
    DiffusionModules,
    create_noise_tensor,
    finalize_generated_sample,
    load_pretrained_weights,
    prepare_class_labels,
    registry,
    compile_if_enabled,
    extract_state_dict,
)
from signal_diffusion.log_setup import get_logger
from signal_diffusion.diffusion.train_utils import (
    apply_soft_min_gamma_snr,
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
    latent_space: bool = False
    vae: str | None = None
    # Multi-attribute conditioning options (for gend_hlth_age)
    num_genders: int = 3   # M, F, dropout token
    num_health: int = 3    # H, PD, dropout token
    age_embedding_dim: int = 256  # Fourier features dimension for age


class LocalMambaAdapter:
    """Adapter for LocalMamba hierarchical selective-scan diffusion model with caption conditioning.

    LocalMamba provides an efficient State Space Model (SSM) based architecture for diffusion,
    featuring hierarchical Mamba blocks with fixed multi-directional scanning patterns. The
    adapter supports both class-based and caption-based conditioning, matching Hourglass's
    conditioning interface while using Mamba's efficient selective scan mechanism.

    Caption Conditioning:
        Caption conditioning in LocalMamba works identically to Hourglass, using DualCLIPTextEncoder
        to convert text descriptions into 2048-dimensional embeddings that condition the generation.
        The embeddings are injected via the mapping_cond parameter and processed through adaptive
        normalization layers in the Mamba blocks.

        Architecture Flow:
            1. Captions → DualCLIPTextEncoder → 2048D embeddings
            2. Embeddings → mapping_cond parameter in model forward pass
            3. Mamba blocks inject conditioning via AdaRMSNorm layers
            4. CFG dropout (10% default) enables classifier-free guidance during inference

        Configuration Example:
            ```toml
            [dataset]
            caption_column = "text"              # Column containing captions

            [model]
            name = "localmamba"
            conditioning = "caption"             # Enable caption conditioning

            [model.extras]
            cfg_dropout = 0.1                    # 10% CFG dropout for guidance training
            mapping_cond_dim = 2048              # DualCLIP output dimension
            mapping_width = 256                  # Mapping network width
            mapping_depth = 2                    # Mapping network depth
            depths = [2, 2, 9, 2]               # Mamba stage depths
            dims = [96, 192, 384, 768]          # Channel dimensions per stage
            scan_directions = ["h", "v", "w7"]  # Horizontal, vertical, window-7
            ```

        Training with Captions:
            ```bash
            uv run python -m signal_diffusion.training.diffusion config.toml
            ```

        Sampling with Prompts:
            ```python
            # Single prompt
            samples = adapter.generate_conditional_samples(
                accelerator=accelerator,
                cfg=cfg,
                modules=modules,
                conditioning="healthy EEG signal",
                guidance_scale=7.5,
                num_samples=4
            )

            # Multiple prompts
            samples = adapter.generate_conditional_samples(
                conditioning=["healthy EEG", "parkinsons pattern", "seizure activity"],
                guidance_scale=7.5
            )
            ```

        CFG Guidance Scale:
            - 1.0: No guidance (purely conditional)
            - 5.0-7.5: Recommended range (good balance)
            - 10.0+: Strong guidance (may reduce diversity)

    Mamba Architecture:
        LocalMamba uses hierarchical VSS (Visual State Space) blocks with:
        - Fixed multi-directional scanning (horizontal, vertical, window-based)
        - BiAttn fusion for combining scan directions
        - PatchMerging downsampling and TokenSplit upsampling
        - Efficient SS2D kernels from mamba-ssm v2.0

    Class Conditioning:
        Use `conditioning = "classes"` for standard discrete class labels, or
        `conditioning = "gend_hlth_age"` for multi-attribute combinations of
        gender, health status, and age.

    Latent Space Training:
        Optionally train in VAE latent space:
        ```toml
        [model.extras]
        latent_space = true
        vae = "stabilityai/stable-diffusion-3.5-medium"
        ```

    Note:
        Caption and class conditioning are mutually exclusive. Set `model.conditioning = "caption"`
        for text-based or `"classes"` for discrete labels. The caption conditioning implementation
        is identical to Hourglass for consistency across adapters.
    """

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
        latent_space = bool(data.get("latent_space", False))
        vae = data.get("vae")
        # Fallback to default stable diffusion model ID if VAE is unspecified
        if vae is None and cfg.settings:
            vae = cfg.settings.hf_models.get("stable_diffusion_model_id")
            if vae is not None:
                self._logger.info("VAE not specified in extras, using default from settings: %s", vae)

        # Multi-attribute conditioning options (for gend_hlth_age conditioning)
        num_genders = int(data.get("num_genders", 3))
        num_health = int(data.get("num_health", 3))
        age_embedding_dim = int(data.get("age_embedding_dim", mapping_width))

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
            vae=vae,
            num_genders=num_genders,
            num_health=num_health,
            age_embedding_dim=age_embedding_dim,
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
        del tokenizer  # LocalMamba does not use HF tokenizer directly
        self._extras = self._parse_extras(cfg)
        self._num_dataset_classes = cfg.dataset.num_classes
        self._use_gradient_checkpointing = bool(cfg.training.gradient_checkpointing)

        # Initialize text encoder and age embedding based on model.conditioning
        self._text_encoder = None
        self._age_embedding = None
        self._age_proj = None

        # Primary conditioning check
        conditioning = str(cfg.model.conditioning or "none").strip().lower()

        if conditioning == "caption":
            # Load dual CLIP text encoders for caption conditioning
            from signal_diffusion.diffusion.text_encoders import DualCLIPTextEncoder

            sd_model_id = "stabilityai/stable-diffusion-3.5-medium"
            if cfg.settings:
                sd_model_id = cfg.settings.hf_models.get("stable_diffusion_model_id", sd_model_id)

            if accelerator.is_main_process:
                self._logger.info("Loading dual CLIP text encoders from %s", sd_model_id)

            weight_dtype = torch.float32
            if accelerator.mixed_precision == "fp16":
                weight_dtype = torch.float16
            elif accelerator.mixed_precision == "bf16":
                weight_dtype = torch.bfloat16

            self._text_encoder = DualCLIPTextEncoder(
                sd_model_id=sd_model_id,
                device=accelerator.device,
                dtype=weight_dtype,
            )
            # Set mapping_cond_dim to match text encoder output (2048)
            self._extras.mapping_cond_dim = self._text_encoder.output_dim
            if accelerator.is_main_process:
                self._logger.info(
                    "Caption conditioning enabled with mapping_cond_dim=%d",
                    self._extras.mapping_cond_dim,
                )
        elif conditioning == "classes":
            # Standard class conditioning
            self._extras.mapping_cond_dim = 0
            if accelerator.is_main_process:
                self._logger.info("Standard class conditioning enabled, mapping_cond_dim set to 0")

        elif conditioning == "gend_hlth_age":
            # Multi-attribute: gender×health classes + age as mapping_cond
            from signal_diffusion.diffusion.conditioning import AgeEmbedding

            self._age_embedding = AgeEmbedding(
                out_features=self._extras.age_embedding_dim
            )
            # Project age embedding to mapping_cond_dim if needed
            if self._extras.age_embedding_dim != self._extras.mapping_width:
                self._age_proj = torch.nn.Linear(
                    self._extras.age_embedding_dim,
                    self._extras.mapping_width,
                    bias=False,
                )
            else:
                self._age_proj = torch.nn.Identity()

            # For gend_hlth_age mode, set mapping_cond_dim to mapping_width for age
            self._extras.mapping_cond_dim = self._extras.mapping_width
            if accelerator.is_main_process:
                self._logger.info(
                    "Multi-attribute (gender+health+age) conditioning enabled with "
                    "age_embedding_dim=%d, mapping_cond_dim=%d",
                    self._extras.age_embedding_dim,
                    self._extras.mapping_cond_dim,
                )

        if accelerator.is_main_process:
            self._logger.info("Building LocalMamba2DModel with extras=%s", self._extras)

        noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=cfg.objective.num_timesteps,
            # By default final timestep is 1 (no noise) so it's wasted during inference
            shift_terminal=max(1 / cfg.objective.num_timesteps, 1 / cfg.inference.denoising_steps / 2),
        )
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

        # Compile models if enabled (compiles all models: denoiser + VAE + text_encoder, not EMA)
        if cfg.training.compile_model:
            model = compile_if_enabled(
                model,
                enabled=True,
                mode=cfg.training.compile_mode,
                model_name="LocalMamba denoiser",
                logger=self._logger,
            )

        # Move age embedding modules to device if they exist
        if self._age_embedding is not None:
            self._age_embedding.to(accelerator.device, dtype=weight_dtype)
        if self._age_proj is not None and not isinstance(self._age_proj, torch.nn.Identity):
            self._age_proj.to(accelerator.device, dtype=weight_dtype)

        # Load VAE only when latent_space is enabled
        vae = None
        vae_latent_channels = None
        if self._extras.latent_space:
            if not self._extras.vae:
                raise ValueError("latent_space mode requires 'vae' path in model extras")

            if "vae" in self._extras.vae:
                vae = AutoencoderKL.from_pretrained(self._extras.vae)
            else:
                vae = AutoencoderKL.from_pretrained(self._extras.vae, subfolder="vae")

            # Optional: tile VAE decode to reduce peak memory during eval.
            if cfg.model.vae_tiling and hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
                if accelerator.is_main_process:
                    self._logger.info("Enabled VAE tiling for latent-space decode")

            vae.requires_grad_(False)
            vae.to(accelerator.device, dtype=weight_dtype)

            # Inspect and verify VAE latent channels
            vae_latent_channels = getattr(vae.config, "latent_channels", None)
            if vae_latent_channels is None:
                self._logger.warning(
                    "VAE config does not specify latent_channels, cannot verify dimensions"
                )
            else:
                # Verify model channels match VAE latent channels
                model_in_channels = self._extras.in_channels
                model_out_channels = self._extras.out_channels

                if model_in_channels != vae_latent_channels:
                    raise ValueError(
                        f"Model in_channels ({model_in_channels}) does not match "
                        f"VAE latent_channels ({vae_latent_channels}). "
                        f"Update model.extras.in_channels={vae_latent_channels} in your config."
                    )

                if model_out_channels != vae_latent_channels:
                    raise ValueError(
                        f"Model out_channels ({model_out_channels}) does not match "
                        f"VAE latent_channels ({vae_latent_channels}). "
                        f"Update model.extras.out_channels={vae_latent_channels} in your config."
                    )

                if accelerator.is_main_process:
                    self._logger.info(
                        "Verified: model channels (%d) match VAE latent_channels (%d)",
                        model_in_channels, vae_latent_channels
                    )

            if accelerator.is_main_process:
                self._logger.info("Loaded VAE from %s for latent space mode", self._extras.vae)

        # Compile VAE and text encoder if enabled
        if cfg.training.compile_model:
            if vae is not None:
                vae = compile_if_enabled(
                    vae,
                    enabled=True,
                    mode=cfg.training.compile_mode,
                    model_name="VAE",
                    logger=self._logger,
                )

            if self._text_encoder is not None:
                self._text_encoder = compile_if_enabled(
                    self._text_encoder,
                    enabled=True,
                    mode=cfg.training.compile_mode,
                    model_name="CLIP text encoder",
                    logger=self._logger,
                )

        if self._use_gradient_checkpointing and accelerator.is_main_process:
            self._logger.info("Enabled gradient checkpointing for LocalMamba denoiser")

        params = list(model.param_groups(base_lr=cfg.optimizer.learning_rate))
        modules = DiffusionModules(
            denoiser=model,
            noise_scheduler=noise_scheduler,
            weight_dtype=weight_dtype,
            vae=vae,
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
            scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
            shift_factor = getattr(vae.config, "shift_factor", 0.0)
            images = (vae.encode(images).latent_dist.sample() - shift_factor) * scaling_factor

        # Prepare conditioning based on model.conditioning
        class_labels: torch.Tensor | None = None
        mapping_cond: torch.Tensor | None = None

        # Primary conditioning check
        conditioning = str(cfg.model.conditioning or "none").strip().lower()

        if conditioning == "caption":
            # Caption conditioning: encode raw captions to mapping_cond
            if self._text_encoder is not None and batch.raw_captions is not None:
                with torch.no_grad():
                    mapping_cond = self._text_encoder.encode(batch.raw_captions)
                    mapping_cond = mapping_cond.to(device, dtype=modules.weight_dtype)
                # Apply CFG dropout by zeroing out some embeddings
                if extras.cfg_dropout > 0:
                    dropout_mask = torch.rand(mapping_cond.shape[0], device=device) < extras.cfg_dropout
                    mapping_cond = torch.where(
                        dropout_mask.unsqueeze(-1),
                        torch.zeros_like(mapping_cond),
                        mapping_cond,
                    )
        elif conditioning == "classes":
            # Standard class conditioning
            class_labels = prepare_class_labels(
                batch, device=device, num_dataset_classes=self._num_dataset_classes, cfg_dropout=extras.cfg_dropout
            )

        elif conditioning == "gend_hlth_age":
            # Multi-attribute: combine gender+health → class_labels, age → mapping_cond
            if batch.age_values is None:
                raise ValueError(
                    "gend_hlth_age conditioning requires age_values in the batch. "
                    "Ensure dataset.age_column is set and the column exists in your dataset."
                )
            if batch.gender_labels is None or batch.health_labels is None:
                raise ValueError(
                    "gend_hlth_age conditioning requires gender_labels and health_labels in the batch. "
                    "Ensure dataset.gender_column and dataset.health_column are set."
                )

            from signal_diffusion.diffusion.conditioning import (
                compute_combined_class,
                prepare_multi_attribute_labels,
            )

            gender_labels, health_labels, age_values = prepare_multi_attribute_labels(
                batch,
                device=device,
                cfg_dropout=extras.cfg_dropout,
                dropout_token=2,
            )

            # Compute combined class: gender * 2 + health, with dropout=num_dataset_classes
            class_labels = compute_combined_class(
                gender_labels,
                health_labels,
                num_health_classes=2,
                dropout_token=self._num_dataset_classes,  # Use num_dataset_classes as dropout
            )

            # Age embedding → mapping_cond
            if self._age_embedding is not None:
                age_emb = self._age_embedding(age_values)
                if self._age_proj is not None:
                    age_emb = self._age_proj(age_emb)
                mapping_cond = age_emb.to(dtype=modules.weight_dtype)
        # else: conditioning == "none", leave class_labels and mapping_cond as None

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
            model_pred = model(z_t, sigma=sigmas, class_cond=class_labels, mapping_cond=mapping_cond)

        loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, loss.ndim)))
        weights = apply_soft_min_gamma_snr(
            snr,
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
        conditioning: torch.Tensor | str | Iterable[str] | Mapping[str, torch.Tensor] | None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Generate conditional samples with classifier-free guidance."""
        scheduler: FlowMatchEulerDiscreteScheduler = FlowMatchEulerDiscreteScheduler.from_config(modules.noise_scheduler.config)  # pyright: ignore[reportAssignmentType]
        device = accelerator.device
        dtype = modules.weight_dtype
        scheduler.set_timesteps(denoising_steps, device=device)

        vae = modules.vae
        extras = self._extras

        sample = create_noise_tensor(
            num_images,
            cfg,
            modules,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        dropout_label = self._num_dataset_classes

        # Initialize conditioning variables
        class_labels: torch.Tensor | None = None
        unconditional_labels: torch.Tensor | None = None
        mapping_cond: torch.Tensor | None = None
        unconditional_mapping_cond: torch.Tensor | None = None
        classifier_free = False
        cfg_scale_value = float(cfg_scale)

        if conditioning is None:
            # Unconditional generation
            class_labels = torch.full((num_images,), dropout_label, device=device, dtype=torch.long)
            classifier_free = False
            # If model expects mapping_cond (e.g., in gend_hlth_age mode), provide zero tensor
            if extras.mapping_cond_dim > 0:
                mapping_cond = torch.zeros(num_images, extras.mapping_cond_dim, device=device, dtype=dtype)

        elif isinstance(conditioning, Mapping) and not isinstance(conditioning, torch.Tensor):
            # Multi-attribute conditioning: {"gender": Tensor, "health": Tensor, "age": Tensor}
            from signal_diffusion.diffusion.conditioning import compute_combined_class

            gender = conditioning.get("gender")
            health = conditioning.get("health")
            age = conditioning.get("age")

            if gender is not None and health is not None:
                gender_labels = gender.to(device=device, dtype=torch.long)
                health_labels = health.to(device=device, dtype=torch.long)

                # Expand if single value provided
                if gender_labels.shape[0] == 1 and num_images > 1:
                    gender_labels = gender_labels.expand(num_images)
                if health_labels.shape[0] == 1 and num_images > 1:
                    health_labels = health_labels.expand(num_images)

                # Compute combined class labels
                class_labels = compute_combined_class(
                    gender_labels,
                    health_labels,
                    num_health_classes=2,
                    dropout_token=dropout_label,
                )
                unconditional_labels = torch.full_like(class_labels, dropout_label)
                classifier_free = True

                # Age embedding → mapping_cond
                if age is not None and self._age_embedding is not None:
                    age_values = age.to(device=device, dtype=dtype)
                    if age_values.shape[0] == 1 and num_images > 1:
                        age_values = age_values.expand(num_images)
                    age_emb = self._age_embedding(age_values)
                    if self._age_proj is not None:
                        age_emb = self._age_proj(age_emb)
                    mapping_cond = age_emb.to(dtype=dtype)
                    # Unconditional: zero mapping_cond (NaN age → zero embedding)
                    unconditional_mapping_cond = torch.zeros_like(mapping_cond)
            else:
                # Fall back to unconditional
                class_labels = torch.full((num_images,), dropout_label, device=device, dtype=torch.long)

        elif isinstance(conditioning, str):
            # Single caption string
            conditioning = [conditioning] * num_images
            if self._text_encoder is None:
                raise RuntimeError("Text encoder not initialized for caption conditioning")
            with torch.no_grad():
                mapping_cond = self._text_encoder.encode(conditioning)
                mapping_cond = mapping_cond.to(device, dtype=dtype)
            unconditional_mapping_cond = torch.zeros_like(mapping_cond)
            classifier_free = True

        elif isinstance(conditioning, Iterable) and not isinstance(conditioning, torch.Tensor):
            # List of captions
            captions = list(conditioning)
            if len(captions) == 1 and num_images > 1:
                captions = captions * num_images
            if len(captions) != num_images:
                raise ValueError("Number of captions must match num_images")
            if self._text_encoder is None:
                raise RuntimeError("Text encoder not initialized for caption conditioning")
            with torch.no_grad():
                mapping_cond = self._text_encoder.encode(captions)
                mapping_cond = mapping_cond.to(device, dtype=dtype)
            unconditional_mapping_cond = torch.zeros_like(mapping_cond)
            classifier_free = True

        elif isinstance(conditioning, torch.Tensor):
            # Standard class-label conditioning tensor
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
                    f"Class labels must be in [0, {self._num_dataset_classes - 1}] for LocalMamba adapter"
                )
            unconditional_labels = torch.full_like(class_labels, dropout_label)
            # If model expects mapping_cond (e.g., in gend_hlth_age mode), provide zero tensor
            if extras.mapping_cond_dim > 0:
                mapping_cond = torch.zeros(num_images, extras.mapping_cond_dim, device=device, dtype=dtype)
                unconditional_mapping_cond = torch.zeros_like(mapping_cond)

        else:
            raise TypeError("Unsupported conditioning value for LocalMamba sampling")

        # Early exit optimization: skip CFG when cfg_scale is 1.0
        if classifier_free and cfg_scale_value == 1.0:
            classifier_free = False
            unconditional_labels = None
            unconditional_mapping_cond = None

        # Prepare conditioning vectors for CFG guidance
        if classifier_free:
            cond_dict = {}
            null_cond_dict = {}

            if class_labels is not None and unconditional_labels is not None:
                cond_dict["class"] = class_labels
                null_cond_dict["class"] = unconditional_labels

            if mapping_cond is not None:
                cond_dict["mapping"] = mapping_cond
                null_cond_dict["mapping"] = (
                    unconditional_mapping_cond
                    if unconditional_mapping_cond is not None
                    else torch.zeros_like(mapping_cond)
                )

            cond_vector = cond_dict
            null_cond_vector = null_cond_dict

        with torch.no_grad():
            for i, timestep in tqdm(enumerate(scheduler.timesteps), desc="Denoising", leave=False):
                model_input = sample
                if hasattr(scheduler, "scale_model_input"):
                    model_input = scheduler.scale_model_input(model_input, timestep)

                timesteps = torch.full((model_input.size(0),), timestep, device=device)
                sigmas = get_sigmas_from_timesteps(scheduler, timesteps, device=device)

                # Compute timestep delta for rectified-CFG++.
                schedule_timesteps = scheduler.timesteps
                schedule_sigmas = scheduler.sigmas
                if schedule_timesteps is None or schedule_sigmas is None:
                    raise RuntimeError("Scheduler timesteps/sigmas are not initialized")
                schedule_len = len(schedule_timesteps)
                if i < schedule_len - 1:
                    dt = schedule_sigmas[i + 1] - schedule_sigmas[i]
                    dT = schedule_timesteps[i + 1] - schedule_timesteps[i]
                else:
                    dt = -schedule_sigmas[i]
                    dT = -schedule_timesteps[i]

                if classifier_free:
                    # Define model evaluation callback for CFG guidance
                    def model_eval_fn(model_input_inner, timestep_inner, conditioning):
                        """Evaluate denoiser, routing conditioning dict to model parameters.

                        conditioning is a dict with keys like 'class' and 'mapping'.
                        apply_cfg_guidance has already concatenated null+cond for each dict entry.
                        """
                        # Extract timesteps for sigma computation
                        if isinstance(timestep_inner, torch.Tensor):
                            timesteps_inner = timestep_inner
                        else:
                            timesteps_inner = torch.full(
                                (model_input_inner.size(0),), timestep_inner, device=device
                            )
                        sigmas_inner = get_sigmas_from_timesteps(scheduler, timesteps_inner, device=device)

                        class_cond_inner = conditioning.get("class")  # May be None
                        mapping_cond_inner = conditioning.get("mapping")  # May be None

                        with self._checkpoint_context():
                            return modules.denoiser(
                                model_input_inner,
                                sigma=sigmas_inner,
                                class_cond=class_cond_inner,
                                mapping_cond=mapping_cond_inner,
                            )

                    # Apply CFG guidance (handles batching/concatenation internally)
                    model_output = apply_cfg_guidance(
                        x_t=model_input,
                        delta_t=dt,
                        delta_T=dT,
                        timestep=timestep,
                        model_eval_fn=model_eval_fn,
                        cond_vector=cond_vector,
                        null_cond_vector=null_cond_vector,
                        cfg_scale=cfg_scale_value,
                        prediction_type=cfg.objective.prediction_type,
                    )
                else:
                    # Unconditional sampling (no CFG)
                    with self._checkpoint_context():
                        model_output = modules.denoiser(
                            model_input, sigma=sigmas, class_cond=class_labels, mapping_cond=mapping_cond
                        )

                step_output = scheduler.step(model_output, timestep, sample, return_dict=True)
                sample = step_output.prev_sample  # type: ignore

        output = finalize_generated_sample(sample, device=device, vae=vae, latent_space=bool(extras.latent_space))
        return output


    def save_checkpoint(
        self,
        accelerator: Accelerator,
        cfg: DiffusionConfig,
        modules: DiffusionModules,
        output_dir: str,
    ) -> None:
        """Save model checkpoint."""
        del accelerator, cfg
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        path = output_path / "localmamba_model.pt"

        # Handle both inner_model wrapper AND torch.compile wrapper
        inner_model = getattr(modules.denoiser, "inner_model", modules.denoiser)
        state_dict = extract_state_dict(inner_model)

        torch.save(state_dict, path)
        self._logger.info("Saved LocalMamba checkpoint to %s", path)


registry.register(LocalMambaAdapter())
