"""Stable Diffusion v1.5 adapter."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from signal_diffusion.diffusion.config import DiffusionConfig
from signal_diffusion.diffusion.data import DiffusionBatch
from signal_diffusion.diffusion.guidance import apply_cfg_guidance
from signal_diffusion.diffusion.models.base import DiffusionModules, registry
from signal_diffusion.log_setup import get_logger


@dataclass(slots=True)
class StableDiffusionExtras:
    train_text_encoder: bool = False
    text_encoder_lr_scale: float = 1.0


class StableDiffusionAdapterV15:
    """Adapter targeting Stable Diffusion v1.5-style checkpoints."""

    name = "stable-diffusion-v1-5"

    def __init__(self) -> None:
        self._extras = StableDiffusionExtras()
        self._logger = get_logger(__name__)

    def create_tokenizer(self, cfg: DiffusionConfig) -> CLIPTokenizer:
        pretrained = cfg.model.pretrained or "runwayml/stable-diffusion-v1-5"
        revision = cfg.model.revision
        tokenizer = CLIPTokenizer.from_pretrained(pretrained, subfolder="tokenizer", revision=revision)
        return tokenizer

    def _parse_extras(self, cfg: DiffusionConfig) -> StableDiffusionExtras:
        extras = cfg.model.extras
        train_text_encoder = bool(extras.get("train_text_encoder", False))
        text_encoder_lr_scale = float(extras.get("text_encoder_lr_scale", 1.0))
        return StableDiffusionExtras(
            train_text_encoder=train_text_encoder,
            text_encoder_lr_scale=text_encoder_lr_scale,
        )

    def build_modules(
        self,
        accelerator: Accelerator,
        cfg: DiffusionConfig,
        tokenizer: CLIPTokenizer | None = None,
    ) -> DiffusionModules:
        pretrained = cfg.model.pretrained or "runwayml/stable-diffusion-v1-5"
        revision = cfg.model.revision
        self._extras = self._parse_extras(cfg)

        if accelerator.is_main_process:
            self._logger.info(
                "Building Stable Diffusion v1.5 modules pretrained=%s revision=%s train_text_encoder=%s lora_enabled=%s",
                pretrained,
                revision,
                self._extras.train_text_encoder,
                cfg.model.lora.enabled,
            )

        text_encoder = CLIPTextModel.from_pretrained(pretrained, subfolder="text_encoder", revision=revision)
        vae = AutoencoderKL.from_pretrained(pretrained, subfolder="vae", revision=revision)
        unet = UNet2DConditionModel.from_pretrained(pretrained, subfolder="unet", revision=revision)

        if cfg.objective.prediction_type != "epsilon":
            raise ValueError(
                "Stable Diffusion adapter only supports noise prediction ('epsilon') objective"
            )

        noise_scheduler = DDPMScheduler.from_pretrained(pretrained, subfolder="scheduler", revision=revision)
        noise_scheduler.register_to_config(prediction_type="epsilon")
        if accelerator.is_main_process:
            self._logger.info("Using %s scheduler with prediction_type=epsilon", type(noise_scheduler).__name__)

        if cfg.training.gradient_checkpointing:
            if hasattr(unet, "enable_gradient_checkpointing"):
                unet.enable_gradient_checkpointing()
            elif hasattr(unet, "set_gradient_checkpointing"):
                unet.set_gradient_checkpointing(True)  # type: ignore[attr-defined]
            else:
                self._logger.warning("Gradient checkpointing requested but unsupported by %s", type(unet).__name__)
            if self._extras.train_text_encoder and hasattr(text_encoder, "gradient_checkpointing_enable"):
                text_encoder.gradient_checkpointing_enable()
            if accelerator.is_main_process:
                self._logger.info("Enabled gradient checkpointing on Stable Diffusion modules")

        if cfg.training.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            if accelerator.is_main_process:
                self._logger.info("Enabled TF32 matmul optimizations")

        unet.requires_grad_(not cfg.model.lora.enabled)
        text_encoder.requires_grad_(self._extras.train_text_encoder)
        vae.requires_grad_(False)

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        unet.to(accelerator.device, dtype=weight_dtype)
        vae.to(accelerator.device, dtype=weight_dtype)
        text_encoder.to(accelerator.device, dtype=weight_dtype)

        if accelerator.is_main_process:
            self._logger.info(
                "Stable Diffusion modules moved to %s with dtype=%s", accelerator.device, str(weight_dtype)
            )

        trainable_params: list[torch.nn.Parameter] = []
        clip_grad_params = None

        if cfg.model.lora.enabled:
            try:
                from peft import LoraConfig
            except ImportError as exc:  # pragma: no cover - helpful guidance when PEFT is missing
                raise ImportError(
                    "LoRA training now depends on PEFT. Install it with 'pip install peft' to enable LoRA adapters."
                ) from exc

            lora_config = LoraConfig(
                r=int(cfg.model.lora.rank),
                lora_alpha=float(cfg.model.lora.alpha),
                target_modules=list(cfg.model.lora.target_modules),
                lora_dropout=float(cfg.model.lora.dropout),
                bias=str(cfg.model.lora.bias),
            )

            unet.add_adapter(lora_config)
            unet.enable_adapters()

            trainable_params = [param for param in unet.parameters() if param.requires_grad]
            clip_grad_params = trainable_params
            if accelerator.is_main_process:
                self._logger.info(
                    "Configured LoRA adapters rank=%d alpha=%.2f targets=%s",
                    cfg.model.lora.rank,
                    cfg.model.lora.alpha,
                    list(cfg.model.lora.target_modules),
                )
        else:
            unet_params = list(unet.parameters())
            trainable_params.extend(unet_params)
            clip_grad_params = unet_params

        if self._extras.train_text_encoder:
            text_encoder.requires_grad_(True)
            text_params = list(text_encoder.parameters())
            trainable_params.extend(text_params)
            if clip_grad_params is not None:
                clip_grad_params = list(clip_grad_params) + text_params
            else:
                clip_grad_params = text_params
            if accelerator.is_main_process:
                self._logger.info(
                    "Text encoder training enabled with lr_scale=%.3f", self._extras.text_encoder_lr_scale
                )

        modules = DiffusionModules(
            denoiser=unet,
            noise_scheduler=noise_scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer or self.create_tokenizer(cfg),
            weight_dtype=weight_dtype,
            parameters=list(trainable_params),
            clip_grad_norm_target=list(clip_grad_params) if clip_grad_params is not None else None,
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
        scheduler_cls = type(modules.noise_scheduler)
        scheduler = scheduler_cls.from_config(modules.noise_scheduler.config)
        device = accelerator.device
        dtype = modules.weight_dtype
        scheduler.set_timesteps(denoising_steps, device=device)

        vae = modules.vae
        text_encoder = modules.text_encoder
        tokenizer = modules.tokenizer
        if vae is None or text_encoder is None or tokenizer is None:
            raise RuntimeError("Stable Diffusion sampling requires VAE, text encoder, and tokenizer")

        if cfg.model.sample_size and cfg.dataset.resolution and cfg.model.sample_size != cfg.dataset.resolution:
            raise ValueError(
                f"Model sample size ({cfg.model.sample_size}) and dataset resolution ({cfg.dataset.resolution}) must be the same."
            )

        height = int(cfg.model.sample_size or cfg.dataset.resolution)
        width = height
        latent_channels = getattr(modules.denoiser.config, 'in_channels', 4)
        latents = torch.randn((num_images, latent_channels, height // 8, width // 8), generator=generator, device=device, dtype=dtype)
        if hasattr(scheduler, 'init_noise_sigma'):
            latents = latents * scheduler.init_noise_sigma

        if conditioning is None:
            prompts = [""] * num_images
        elif isinstance(conditioning, torch.Tensor):
            raise TypeError("Stable Diffusion adapter does not accept tensor class-label conditioning during sampling")
        elif isinstance(conditioning, str):
            prompts = [conditioning] * num_images
        elif isinstance(conditioning, Iterable):
            prompts_list = [str(item) for item in conditioning]
            if len(prompts_list) == 1 and num_images > 1:
                prompts = prompts_list * num_images
            elif len(prompts_list) != num_images:
                raise ValueError("Number of text prompts must match num_images")
            else:
                prompts = prompts_list
        else:
            raise TypeError("Unsupported conditioning value for Stable Diffusion sampling")

        # Tokenize and encode prompts
        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_inputs = text_inputs.input_ids.to(device)

        with torch.no_grad():
            text_embeddings = text_encoder(text_inputs)[0]
            uncond_inputs = tokenizer(
                [""] * num_images,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            uncond_embeddings = text_encoder(uncond_inputs.input_ids.to(device))[0]

        # Prepare conditioning vectors for CFG guidance
        cond_vector = text_embeddings
        null_cond_vector = uncond_embeddings

        with torch.no_grad():
            for timestep in tqdm(scheduler.timesteps, desc="Denoising", leave=False):
                latent_model_input = latents
                if hasattr(scheduler, 'scale_model_input'):
                    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

                # Define model evaluation callback for CFG guidance
                def model_eval_fn(latents_inner, timestep_inner, encoder_hidden_states):
                    """Evaluate denoiser, routing encoder_hidden_states to model.

                    encoder_hidden_states is the sequence embeddings from CLIP.
                    apply_cfg_guidance has already concatenated null+cond embeddings.
                    """
                    # Call model and extract .sample attribute
                    return modules.denoiser(
                        latents_inner, timestep_inner, encoder_hidden_states=encoder_hidden_states
                    ).sample

                # Apply CFG guidance (handles batching/concatenation internally)
                # SD v1.5 uses epsilon prediction type
                guidance = apply_cfg_guidance(
                    x_t=latent_model_input,
                    timestep=timestep,
                    model_eval_fn=model_eval_fn,
                    cond_vector=cond_vector,
                    null_cond_vector=null_cond_vector,
                    cfg_scale=cfg_scale,
                    prediction_type="epsilon",
                )

                latents = scheduler.step(guidance, timestep, latents).prev_sample

            latents = latents / getattr(vae.config, 'scaling_factor', 1.0)
            images = vae.decode(latents.to(dtype=vae.dtype)).sample

        return images.to(dtype=torch.float32).detach()


    def training_step(
        self,
        accelerator: Accelerator,
        cfg: DiffusionConfig,
        modules: DiffusionModules,
        batch: DiffusionBatch,
    ) -> tuple[torch.Tensor, Mapping[str, float]]:
        unet = modules.denoiser
        vae = modules.vae
        text_encoder = modules.text_encoder
        tokenizer = modules.tokenizer
        scheduler = modules.noise_scheduler

        latents = vae.encode(batch.pixel_values.to(device=accelerator.device, dtype=modules.weight_dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        if tokenizer is None or batch.captions is None:
            raise ValueError("Stable Diffusion requires caption tokens for conditioning")

        captions = batch.captions.to(accelerator.device)
        encoder_hidden_states = text_encoder(captions)[0]

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
        target = noise

        loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
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
        return {"loss": float(accelerator.gather_for_metrics(loss.detach()).mean())}

    def save_checkpoint(
        self,
        accelerator: Accelerator,
        cfg: DiffusionConfig,
        modules: DiffusionModules,
        output_dir: str,
    ) -> None:
        project_dir = getattr(getattr(accelerator, "project_configuration", None), "project_dir", None)
        if project_dir is not None:
            output_path = Path(project_dir) / output_dir
        else:
            output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        accelerator.print(f"Saving Stable Diffusion weights to {output_path}")
        if cfg.model.lora.enabled:
            modules.denoiser.save_lora_adapter(str(output_path))
        else:
            modules.denoiser.save_pretrained(str(output_path))


registry.register(StableDiffusionAdapterV15())
