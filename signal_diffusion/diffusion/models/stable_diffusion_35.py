"""Stable Diffusion 3.5 Medium adapter.

Uses flow matching with dual CLIP text encoders (CLIP-L + CLIP-G),
skipping T5XXL for memory efficiency.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from tqdm import tqdm

from signal_diffusion.diffusion.config import DiffusionConfig
from signal_diffusion.diffusion.data import DiffusionBatch
from signal_diffusion.diffusion.models.base import DiffusionModules, registry
from signal_diffusion.diffusion.text_encoders import DualCLIPTextEncoder
from signal_diffusion.diffusion.train_utils import (
    apply_min_gamma_snr,
    get_snr,
    sample_timestep_logitnorm,
    verify_scheduler,
)
from signal_diffusion.log_setup import get_logger


@dataclass(slots=True)
class SD35Extras:
    """Configuration for Stable Diffusion 3.5 Medium adapter."""
    skip_t5: bool = True  # Skip T5XXL encoder to save memory
    cfg_dropout: float = 0.1  # Caption dropout for classifier-free guidance
    train_text_encoder: bool = False  # Whether to train the CLIP encoders


class StableDiffusion35Adapter:
    """Adapter for Stable Diffusion 3.5 Medium with native caption conditioning and latent space training.

    This adapter provides production-ready diffusion training using Stable Diffusion 3.5 Medium's
    architecture, featuring native latent space operation, dual CLIP text encoders, and flow matching.
    Unlike Hourglass/LocalMamba which optionally support latent space, SD 3.5 always operates in
    latent space for memory efficiency.

    Architecture Components:
        - **Denoiser:** SD3Transformer2DModel (MMDiT architecture from SD 3.5)
        - **Scheduler:** FlowMatchEulerDiscreteScheduler for flow matching
        - **Text Encoder:** DualCLIPTextEncoder (CLIP-L 768D + CLIP-G 1280D = 2048D pooled)
        - **VAE:** AutoencoderKL with 16 latent channels and 8× spatial compression
        - **T5XXL:** Skipped by default (skip_t5=True) for memory efficiency

    Caption Conditioning:
        SD 3.5 has native caption conditioning built into the architecture via pooled_projections
        parameter in the transformer. Text embeddings are processed through the dual CLIP encoders
        and injected directly into the MMDiT blocks.

        Architecture Flow:
            1. Captions → DualCLIPTextEncoder → 2048D pooled embeddings
            2. Embeddings → pooled_projections parameter in SD3Transformer2DModel
            3. MMDiT blocks inject conditioning via joint attention mechanism
            4. CFG dropout (10% default) enables classifier-free guidance

        Configuration Example:
            ```toml
            [dataset]
            caption_column = "text"              # Column containing captions

            [model]
            name = "stable-diffusion-3.5-medium"
            conditioning = "caption"             # Enable caption conditioning
            pretrained = "stabilityai/stable-diffusion-3.5-medium"

            [model.extras]
            skip_t5 = true                       # Skip T5XXL encoder (recommended)
            cfg_dropout = 0.1                    # 10% CFG dropout
            train_text_encoder = false           # Keep CLIP encoders frozen
            latent_space = true                  # Always true for SD 3.5 (native)
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
                conditioning="healthy EEG signal from adult patient",
                guidance_scale=7.5,
                num_samples=4
            )

            # Multiple prompts
            samples = adapter.generate_conditional_samples(
                conditioning=[
                    "healthy EEG pattern",
                    "parkinsons tremor with alpha rhythm suppression",
                    "seizure activity in temporal lobe"
                ],
                guidance_scale=7.5
            )
            ```

        CFG Guidance Scale:
            - 1.0: No guidance (purely conditional)
            - 5.0-7.5: Recommended range (balanced quality and diversity)
            - 10.0+: Strong guidance (higher fidelity, lower diversity)

    Latent Space:
        SD 3.5 always operates in VAE latent space with:
        - **Compression:** 8× spatial (64×64 → 8×8)
        - **Channels:** 16 latent channels
        - **Scaling:** scaling_factor from VAE config (applied during encode/decode)
        - **Memory:** ~64× reduction compared to pixel space

        Images are automatically encoded to latents during training and decoded during sampling.
        No configuration needed - this is native to SD 3.5 architecture.

    Differences from SD v1.5:
        - **Text Encoders:** Dual CLIP (L+G) vs. single CLIP → 2048D vs. 768D embeddings
        - **Latent Channels:** 16 vs. 4 channels
        - **Architecture:** MMDiT (joint attention) vs. UNet (cross-attention)
        - **Scheduler:** Flow matching vs. DDPM
        - **Pooling:** Pooled embeddings (pooled_projections) vs. sequence (encoder_hidden_states)

    skip_t5 Parameter:
        By default, skip_t5=True to avoid loading the large T5XXL text encoder (~11GB). The dual
        CLIP encoders provide sufficient caption conditioning for EEG spectrogram generation tasks.
        Set skip_t5=False only if you need the full SD 3.5 text conditioning capacity.

    Training Text Encoders:
        By default, CLIP encoders are frozen (train_text_encoder=False) for stability and memory
        efficiency. Set train_text_encoder=True to fine-tune the encoders, but note this requires
        significantly more memory and may destabilize training.

    Note:
        This adapter requires `dataset.latent_space = true` in the config, though this is implicit
        for SD 3.5. Unconditional generation uses empty caption strings ("") processed through the
        same dual CLIP pipeline.
    """

    name = "stable-diffusion-3.5-medium"

    def __init__(self) -> None:
        self._extras = SD35Extras()
        self._logger = get_logger(__name__)
        self._text_encoder: DualCLIPTextEncoder | None = None

    def create_tokenizer(self, cfg: DiffusionConfig):
        """SD 3.5 uses DualCLIPTextEncoder which handles its own tokenization."""
        return None

    def _parse_extras(self, cfg: DiffusionConfig) -> SD35Extras:
        """Parse adapter-specific configuration."""
        extras = cfg.model.extras
        skip_t5 = bool(extras.get("skip_t5", True))
        cfg_dropout = float(extras.get("cfg_dropout", 0.1))
        train_text_encoder = bool(extras.get("train_text_encoder", False))
        return SD35Extras(
            skip_t5=skip_t5,
            cfg_dropout=cfg_dropout,
            train_text_encoder=train_text_encoder,
        )

    def build_modules(
        self,
        accelerator: Accelerator,
        cfg: DiffusionConfig,
        tokenizer=None,
    ) -> DiffusionModules:
        """Build SD 3.5 model components."""
        del tokenizer  # We use DualCLIPTextEncoder instead
        self._extras = self._parse_extras(cfg)

        # Get model path from config or settings
        model_id = cfg.model.pretrained or "stabilityai/stable-diffusion-3.5-medium"
        if cfg.settings:
            model_id = cfg.settings.hf_models.get("stable_diffusion_model_id", model_id)

        if accelerator.is_main_process:
            self._logger.info(
                "Building Stable Diffusion 3.5 Medium modules from %s (skip_t5=%s)",
                model_id,
                self._extras.skip_t5,
            )

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Load transformer (denoiser)
        transformer = SD3Transformer2DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=weight_dtype
        )

        # Load VAE
        vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=weight_dtype
        )

        if cfg.model.vae_tiling and hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
            if accelerator.is_main_process:
                self._logger.info("Enabled VAE tiling for latent-space decode")

        # Load dual CLIP text encoder
        self._text_encoder = DualCLIPTextEncoder(
            sd_model_id=model_id,
            device=accelerator.device,
            dtype=weight_dtype,
        )

        if accelerator.is_main_process:
            self._logger.info(
                "Loaded DualCLIPTextEncoder with output_dim=%d",
                self._text_encoder.output_dim,
            )

        # Create flow matching scheduler
        noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=cfg.objective.flow_match_timesteps
        )
        verify_scheduler(noise_scheduler)

        if accelerator.is_main_process:
            self._logger.info(
                "Using FlowMatchEulerDiscreteScheduler with %d timesteps",
                cfg.objective.flow_match_timesteps,
            )

        # Enable gradient checkpointing if requested
        if cfg.training.gradient_checkpointing:
            if hasattr(transformer, "enable_gradient_checkpointing"):
                transformer.enable_gradient_checkpointing()
            if accelerator.is_main_process:
                self._logger.info("Enabled gradient checkpointing on SD3 transformer")

        # Freeze components that shouldn't be trained
        vae.requires_grad_(False)
        if not self._extras.train_text_encoder:
            self._text_encoder.requires_grad_(False)

        # Move to device
        transformer.to(accelerator.device, dtype=weight_dtype)
        vae.to(accelerator.device, dtype=weight_dtype)

        if accelerator.is_main_process:
            self._logger.info(
                "SD 3.5 modules moved to %s with dtype=%s",
                accelerator.device,
                str(weight_dtype),
            )

        # Collect trainable parameters
        trainable_params = list(transformer.parameters())
        if self._extras.train_text_encoder:
            trainable_params.extend(self._text_encoder.parameters())

        modules = DiffusionModules(
            denoiser=transformer,
            noise_scheduler=noise_scheduler,
            vae=vae,
            text_encoder=self._text_encoder,
            weight_dtype=weight_dtype,
            parameters=trainable_params,
            clip_grad_norm_target=trainable_params,
        )
        return modules

    def training_step(
        self,
        accelerator: Accelerator,
        cfg: DiffusionConfig,
        modules: DiffusionModules,
        batch: DiffusionBatch,
    ) -> tuple[torch.Tensor, Mapping[str, float]]:
        """Compute training loss using flow matching."""
        transformer = modules.denoiser
        vae = modules.vae
        scheduler = modules.noise_scheduler
        device = accelerator.device

        if vae is None:
            raise RuntimeError("VAE required for SD 3.5 training")

        # Encode images to latent space
        images = batch.pixel_values.to(device, dtype=modules.weight_dtype)
        latents = vae.encode(images).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # Encode captions
        if batch.raw_captions is None:
            raise ValueError("SD 3.5 requires raw_captions for conditioning")

        if self._text_encoder is None:
            raise RuntimeError("Text encoder not initialized")

        with torch.no_grad():
            text_embeddings = self._text_encoder.encode(batch.raw_captions)
            text_embeddings = text_embeddings.to(device, dtype=modules.weight_dtype)

        # Apply CFG dropout by zeroing out some embeddings
        if self._extras.cfg_dropout > 0:
            dropout_mask = torch.rand(text_embeddings.shape[0], device=device) < self._extras.cfg_dropout
            text_embeddings = torch.where(
                dropout_mask.unsqueeze(-1),
                torch.zeros_like(text_embeddings),
                text_embeddings,
            )

        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        timesteps = sample_timestep_logitnorm(
            latents.shape[0],
            num_train_timesteps=scheduler.config.num_train_timesteps,
            timesteps=scheduler.timesteps,
            device=device,
        )
        z_t = scheduler.scale_noise(latents, timesteps, noise)
        snr = get_snr(scheduler, timesteps, device=device)

        # Compute target (flow matching uses velocity field)
        if cfg.objective.prediction_type == "epsilon":
            target = noise
        elif cfg.objective.prediction_type == "vector_field":
            target = noise - latents
        else:
            raise ValueError(f"Unsupported prediction type {cfg.objective.prediction_type}")

        # Forward pass through transformer
        # Since we skip T5 (text_encoder_3), pass zero tensors for encoder_hidden_states
        # to match behavior of official SD 3.5 pipeline when text_encoder_3=None
        joint_attention_dim = transformer.config.joint_attention_dim
        encoder_hidden_states = torch.zeros(
            (z_t.shape[0], 77, joint_attention_dim),
            device=device,
            dtype=modules.weight_dtype,
        )

        model_pred = transformer(
            hidden_states=z_t,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=text_embeddings,
            return_dict=False,
        )[0]

        # Compute loss with SNR weighting
        loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, loss.ndim)))
        weights = apply_min_gamma_snr(
            snr,
            timesteps=timesteps,
            gamma=cfg.training.snr_gamma,
            prediction_type=cfg.objective.prediction_type,
        )
        loss = (loss * weights).mean()

        metrics: dict[str, float] = {}
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
        """Generate unconditional samples (uses empty prompts)."""
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
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(modules.noise_scheduler.config)
        device = accelerator.device
        dtype = modules.weight_dtype
        scheduler.set_timesteps(denoising_steps, device=device)

        vae = modules.vae
        if vae is None:
            raise RuntimeError("VAE required for SD 3.5 sampling")
        if self._text_encoder is None:
            raise RuntimeError("Text encoder not initialized")

        # Determine latent shape
        height = int(cfg.model.sample_size or cfg.dataset.resolution)
        width = height
        latent_channels = getattr(modules.denoiser.config, "in_channels", 16)
        scaling = vae.config.scaling_factor

        # Initialize latents
        latents = torch.randn(
            (num_images, latent_channels, height // 8, width // 8),
            generator=generator,
            device=device,
            dtype=dtype,
        )

        # Prepare text conditioning
        if conditioning is None:
            prompts = [""] * num_images
        elif isinstance(conditioning, torch.Tensor):
            raise TypeError("SD 3.5 adapter does not accept tensor conditioning during sampling")
        elif isinstance(conditioning, str):
            prompts = [conditioning] * num_images
        elif isinstance(conditioning, Iterable):
            prompts = list(conditioning)
            if len(prompts) == 1 and num_images > 1:
                prompts = prompts * num_images
            if len(prompts) != num_images:
                raise ValueError("Number of prompts must match num_images")
        else:
            raise TypeError("Unsupported conditioning type")

        # Encode prompts
        with torch.no_grad():
            text_embeddings = self._text_encoder.encode(prompts)
            text_embeddings = text_embeddings.to(device, dtype=dtype)
            # Unconditional embeddings (empty prompts)
            uncond_embeddings = self._text_encoder.encode([""] * num_images)
            uncond_embeddings = uncond_embeddings.to(device, dtype=dtype)

        # Denoising loop with CFG
        with torch.no_grad():
            for timestep in tqdm(scheduler.timesteps, desc="Denoising", leave=False):
                # Duplicate for CFG
                latent_input = torch.cat([latents, latents], dim=0)
                if hasattr(scheduler, "scale_model_input"):
                    latent_input = scheduler.scale_model_input(latent_input, timestep)

                # Prepare conditioning
                pooled_projections = torch.cat([uncond_embeddings, text_embeddings], dim=0)
                timesteps_input = timestep.expand(latent_input.shape[0])

                # Since we skip T5 (text_encoder_3), pass zero tensors for encoder_hidden_states
                # to match behavior of official SD 3.5 pipeline when text_encoder_3=None
                joint_attention_dim = modules.denoiser.config.joint_attention_dim
                encoder_hidden_states = torch.zeros(
                    (latent_input.shape[0], 77, joint_attention_dim),
                    device=device,
                    dtype=dtype,
                )

                # Forward pass
                model_output = modules.denoiser(
                    hidden_states=latent_input,
                    timestep=timesteps_input,
                    encoder_hidden_states=encoder_hidden_states,
                    pooled_projections=pooled_projections,
                    return_dict=False,
                )[0]

                # Apply classifier-free guidance
                model_output_uncond, model_output_cond = model_output.chunk(2)
                model_output = model_output_uncond + cfg_scale * (model_output_cond - model_output_uncond)

                # Scheduler step
                step_output = scheduler.step(model_output, timestep, latents, return_dict=True)
                latents = step_output.prev_sample

            # Decode latents
            latents = latents / scaling
            images = vae.decode(latents.to(dtype=vae.dtype)).sample

        return images.to(dtype=torch.float32).detach()

    def save_checkpoint(
        self,
        accelerator: Accelerator,
        cfg: DiffusionConfig,
        modules: DiffusionModules,
        output_dir: str,
    ) -> None:
        """Save model checkpoint."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        modules.denoiser.save_pretrained(str(output_path / "transformer"))
        self._logger.info("Saved SD 3.5 checkpoint to %s", output_path)


registry.register(StableDiffusion35Adapter())
