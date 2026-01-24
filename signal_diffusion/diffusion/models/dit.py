"""DiT adapter supporting noise prediction and flow matching objectives."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Mapping

import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL, DiTTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm
from transformers import AutoTokenizer

from signal_diffusion.diffusion.config import DiffusionConfig
from signal_diffusion.diffusion.data import DiffusionBatch
from signal_diffusion.diffusion.guidance import apply_cfg_guidance
from signal_diffusion.diffusion.models.base import (
    DiffusionModules,
    create_noise_tensor,
    finalize_generated_sample,
    registry,
    compile_if_enabled,
    extract_state_dict,
)
from signal_diffusion.diffusion.train_utils import (
    apply_soft_min_gamma_snr,
    get_sigmas_from_timesteps,
    get_snr,
    sample_timestep_logitnorm,
    verify_scheduler,
)
from signal_diffusion.log_setup import get_logger


@dataclass(slots=True)
class DiTExtras:
    latent_space: bool = False
    vae: str | None = None
    text_encoder: str | None = None
    num_classes: int = 0
    cfg_dropout: float = 0.0
    timestep_embeddings: int = 1000
    in_channels: int = 3
    out_channels: int = 3


class DiTAdapter:
    name = "dit"

    def __init__(self) -> None:
        self._extras = DiTExtras()
        self._logger = get_logger(__name__)

    def create_tokenizer(self, cfg: DiffusionConfig):
        conditioning = str(cfg.model.conditioning or "none").strip().lower()
        if not conditioning or conditioning == "none" or conditioning == "classes":
            return None
        if conditioning != "caption":
            raise ValueError(f"Unsupported conditioning type '{conditioning}' for DiT models")
        text_encoder_id = cfg.model.extras.get("text_encoder")
        if not text_encoder_id:
            raise ValueError("Caption conditioning requires 'model.extras.text_encoder' to be set")
        return AutoTokenizer.from_pretrained(text_encoder_id)

    def _parse_extras(self, cfg: DiffusionConfig) -> DiTExtras:
        extras = cfg.model.extras
        conditioning = str(cfg.model.conditioning or "none").strip().lower()
        if conditioning not in {"none", "caption", "classes"}:
            raise ValueError(f"Unsupported conditioning type '{conditioning}' for DiT models")
        latent_space = bool(extras.get("latent_space", False))
        vae = extras.get("vae")
        # Fallback to default stable diffusion model ID if VAE is unspecified
        if vae is None and cfg.settings:
            vae = cfg.settings.hf_models.get("stable_diffusion_model_id")
            if vae is not None:
                self._logger.info("VAE not specified in extras, using default from settings: %s", vae)
        num_classes_value = extras.get("num_classes", cfg.dataset.num_classes)
        num_classes = int(num_classes_value or 0)
        cfg_dropout = float(extras.get("cfg_dropout", 0.0))
        timestep_embeddings = int(cfg.objective.num_timesteps or 1000)
        text_encoder = extras.get("text_encoder")
        in_channels = int(extras.get("in_channels", 4 if latent_space else 3))
        out_channels = int(extras.get("out_channels", 4 if latent_space else 3))

        return DiTExtras(
            latent_space=latent_space,
            vae=vae,
            text_encoder=text_encoder,
            num_classes=num_classes,
            cfg_dropout=cfg_dropout,
            timestep_embeddings=timestep_embeddings,
            in_channels=in_channels,
            out_channels=out_channels,
        )

    def _dit_prepare_class_labels(
        self,
        images: torch.Tensor,
        batch: DiffusionBatch,
        extras: DiTExtras,
        conditioning: str,
    ) -> torch.Tensor:
        """Prepare class labels for DiT training with CFG dropout."""
        if conditioning == "classes":
            # Standard class conditioning
            if batch.class_labels is None:
                raise ValueError("Class conditioning requires 'batch.class_labels'")
            class_labels_out = batch.class_labels.to(device=images.device, dtype=torch.long)
            if extras.cfg_dropout > 0:
                mask = torch.rand(class_labels_out.shape, device=class_labels_out.device) < extras.cfg_dropout
                class_labels_out = class_labels_out.masked_fill(mask, extras.num_classes)
        else:
            # Diffusers' DiT uses AdaLayerNormZero under the hood, which always expects
            # a label embedding. Provide a constant dummy tensor so unconditional runs
            # (conditioning="none" or "caption") can skip classifier guidance while satisfying the API.
            class_labels_out = torch.zeros(images.shape[0], device=images.device, dtype=torch.long)
        return class_labels_out

    def build_modules(
        self,
        accelerator: Accelerator,
        cfg: DiffusionConfig,
        tokenizer=None,
    ) -> DiffusionModules:
        self._extras = self._parse_extras(cfg)
        conditioning = str(cfg.model.conditioning or "none").strip().lower()

        if accelerator.is_main_process:
            self._logger.info(
                "Building DiT modules pretrained=%s conditioning=%s latent_space=%s",
                bool(cfg.model.pretrained),
                conditioning,
                self._extras.latent_space,
            )
            self._logger.info("DiT extras=%s", asdict(self._extras))

        if self._extras.latent_space and not self._extras.vae:
            raise ValueError("DiT latent_space requires 'vae' path in model extras")

        if cfg.model.pretrained:
            model = DiTTransformer2DModel.from_pretrained(cfg.model.pretrained)
        else:
            if cfg.model.sample_size and cfg.dataset.resolution and cfg.model.sample_size != cfg.dataset.resolution:
                raise ValueError(
                    f"Model sample size ({cfg.model.sample_size}) and dataset resolution ({cfg.dataset.resolution}) must be the same."
                )
            kwargs = dict(
                num_attention_heads=int(cfg.model.extras.get("num_attention_heads", 16)),
                attention_head_dim=int(cfg.model.extras.get("attention_head_dim", 72)),
                in_channels=int(cfg.model.extras.get("in_channels", 4 if self._extras.latent_space else 3)),
                out_channels=int(cfg.model.extras.get("out_channels", 4 if self._extras.latent_space else 3)),
                num_layers=int(cfg.model.extras.get("num_layers", 28)),
                dropout=float(cfg.model.extras.get("dropout", 0.0)),
                norm_num_groups=int(cfg.model.extras.get("norm_num_groups", 32)),
                attention_bias=bool(cfg.model.extras.get("attention_bias", True)),
                sample_size=int(cfg.model.sample_size or cfg.dataset.resolution),
                patch_size=int(cfg.model.extras.get("patch_size", 2)),
                activation_fn=str(cfg.model.extras.get("activation_fn", "gelu-approximate")),
                upcast_attention=bool(cfg.model.extras.get("upcast_attention", False)),
                norm_elementwise_affine=bool(cfg.model.extras.get("norm_elementwise_affine", False)),
                norm_eps=float(cfg.model.extras.get("norm_eps", 1e-5)),
            )
            # Diffusers' DiT implementation only supports `ada_norm_zero` when a patch_size is used.
            # To keep unconditional runs working we stick with `ada_norm_zero` and feed dummy
            # class ids during the forward pass when no conditioning is required.
            norm_type = "ada_norm_zero"
            if conditioning == "classes":
                if self._extras.num_classes <= 0:
                    raise ValueError("Class conditioning requires 'model.extras.num_classes' to be greater than 0")
                # Allocate num_classes + 1 embeddings to include the dropout token for CFG
                # E.g., 5 classes (0-4) need 6 embeddings (0-5), where 5 is the unconditional token
                num_embeds_ada_norm = self._extras.num_classes + 1
            else:
                # unconditional / caption runs fall back to timestep embedding count
                num_embeds_ada_norm = self._extras.timestep_embeddings
            kwargs["norm_type"] = norm_type
            kwargs["num_embeds_ada_norm"] = num_embeds_ada_norm
            model = DiTTransformer2DModel(**kwargs)

        if cfg.training.gradient_checkpointing:
            if hasattr(model, "enable_gradient_checkpointing"):
                model.enable_gradient_checkpointing()
            elif hasattr(model, "set_gradient_checkpointing"):
                model.set_gradient_checkpointing(True)  # type: ignore[attr-defined]
            else:
                self._logger.warning("Gradient checkpointing requested but unsupported by %s", type(model).__name__)
            if accelerator.is_main_process:
                self._logger.info("Enabled gradient checkpointing on DiT denoiser")
        if cfg.model.lora.enabled:
            raise NotImplementedError("LoRA for DiT models is not yet implemented")

        noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=cfg.objective.num_timesteps,
            # By default final timestep is 1 (no noise) so it's wasted during inference
            shift_terminal=max(1 / cfg.objective.num_timesteps, 1 / cfg.inference.denoising_steps / 2),
        )
        # noise_scheduler.register_to_config(prediction_type=cfg.objective.prediction_type)
        verify_scheduler(noise_scheduler)

        if accelerator.is_main_process:
            self._logger.info(
                "Noise scheduler %s with %d timesteps and prediction_type=%s",
                type(noise_scheduler).__name__,
                cfg.objective.num_timesteps,
                cfg.objective.prediction_type,
            )

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        model.to(accelerator.device, dtype=weight_dtype)

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

        if accelerator.is_main_process:
            self._logger.info(
                "DiT model moved to %s with dtype=%s", accelerator.device, str(weight_dtype)
            )

        # Compile models if enabled (compiles all models: denoiser + VAE, not EMA)
        if cfg.training.compile_model:
            model = compile_if_enabled(
                model,
                enabled=True,
                mode=cfg.training.compile_mode,
                model_name="DiT denoiser",
                logger=self._logger,
            )

            if vae is not None:
                vae = compile_if_enabled(
                    vae,
                    enabled=True,
                    mode=cfg.training.compile_mode,
                    model_name="VAE",
                    logger=self._logger,
                )

        params = list(model.parameters())
        modules = DiffusionModules(
            denoiser=model,
            noise_scheduler=noise_scheduler,
            vae=vae,
            weight_dtype=weight_dtype,
            parameters=params,
            clip_grad_norm_target=params,
        )
        return modules

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
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(modules.noise_scheduler.config)
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
        extras = self._extras
        dropout_label = extras.num_classes

        # Initialize conditioning variables
        class_labels: torch.Tensor | None = None
        unconditional_labels: torch.Tensor | None = None
        classifier_free = False

        if conditioning is None:
            # Unconditional generation
            class_labels = torch.zeros(num_images, device=device, dtype=torch.long)
            classifier_free = False

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
            num_embeds = int(getattr(getattr(modules.denoiser, "config", None), "num_embeds_ada_norm", dropout_label + 1))
            if dropout_label >= num_embeds:
                raise ValueError(
                    "DiT classifier-free guidance expects an extra class embedding for dropout; "
                    "increase 'model.extras.num_classes' to include the dropout token."
                )
            unconditional_labels = torch.full_like(class_labels, dropout_label)

        else:
            raise TypeError("DiT adapter only supports tensor class-label conditioning during sampling")

        # Prepare conditioning vectors for CFG guidance
        if classifier_free:
            # DiT only uses class conditioning (no mapping_cond)
            cond_vector = {"class": class_labels}
            null_cond_vector = {"class": unconditional_labels}

        with torch.no_grad():
            for i, timestep in tqdm(enumerate(scheduler.timesteps), desc="Denoising", leave=False):
                model_input = sample
                if hasattr(scheduler, "scale_model_input"):
                    model_input = scheduler.scale_model_input(model_input, timestep)

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

                        conditioning is a dict with key 'class'.
                        apply_cfg_guidance has already concatenated null+cond for the class labels.
                        """
                        # Expand timestep to match batch size (CFG doubles the batch)
                        if isinstance(timestep_inner, torch.Tensor):
                            denoiser_timestep_inner = timestep_inner
                        else:
                            denoiser_timestep_inner = torch.full(
                                (model_input_inner.shape[0],), timestep_inner, device=device
                            )

                        class_labels_inner = conditioning.get("class")

                        # Call model and extract .sample attribute
                        return modules.denoiser(
                            model_input_inner, timestep=denoiser_timestep_inner, class_labels=class_labels_inner
                        ).sample

                    # Apply CFG guidance (handles batching/concatenation internally)
                    model_output = apply_cfg_guidance(
                        x_t=model_input,
                        delta_t=dt,
                        delta_T=dT,
                        timestep=timestep,
                        model_eval_fn=model_eval_fn,
                        cond_vector=cond_vector,
                        null_cond_vector=null_cond_vector,
                        cfg_scale=cfg_scale,
                        prediction_type=cfg.objective.prediction_type,
                    )
                else:
                    # Unconditional sampling (no CFG)
                    denoiser_timestep = timestep.expand(model_input.shape[0])
                    model_output = modules.denoiser(
                        model_input, timestep=denoiser_timestep, class_labels=class_labels
                    ).sample

                step_output = scheduler.step(model_output, timestep, sample)
                sample = getattr(step_output, "prev_sample", step_output[0] if isinstance(step_output, tuple) else step_output)

        return finalize_generated_sample(
            sample,
            device=device,
            vae=vae,
            latent_space=self._extras.latent_space
        )

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
        conditioning = str(cfg.model.conditioning or "none").strip().lower()

        images = batch.pixel_values.to(accelerator.device, dtype=modules.weight_dtype)
        if extras.latent_space:
            vae = modules.vae
            if vae is None:
                raise RuntimeError("VAE expected but not initialised")
            scaling_factor = getattr(vae.config, "scaling_factor", 1.0)
            shift_factor = getattr(vae.config, "shift_factor", 0.0)
            images = (vae.encode(images).latent_dist.sample() - shift_factor) * scaling_factor

        noise = torch.randn_like(images)
        timesteps = sample_timestep_logitnorm(
            images.shape[0],
            num_train_timesteps=scheduler.config.num_train_timesteps,
            timesteps=scheduler.timesteps,
            device=images.device,
        )
        z_t = scheduler.scale_noise(images, timesteps, noise)
        snr = get_snr(scheduler, timesteps, device=images.device)

        class_labels = self._dit_prepare_class_labels(images, batch, extras, conditioning)

        if cfg.objective.prediction_type == "epsilon":
            target = noise
        elif cfg.objective.prediction_type == "vector_field":
            target = noise - images
        else:
            raise ValueError(f"Unsupported prediction type {cfg.objective.prediction_type} for DiT")

        model_pred = model(z_t, timestep=timesteps, class_labels=class_labels).sample
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
                sigmas=get_sigmas_from_timesteps(scheduler, timesteps, device=images.device),
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
        loss, _ = self.training_step(accelerator, cfg, modules, batch)
        gathered = accelerator.gather_for_metrics(loss.detach())
        return {"loss": float(gathered.mean())}

    def save_checkpoint(
        self,
        accelerator: Accelerator,
        cfg: DiffusionConfig,
        modules: DiffusionModules,
        output_dir: str,
    ) -> None:
        modules.denoiser.save_pretrained(output_dir)


registry.register(DiTAdapter())
