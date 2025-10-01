"""Stable Diffusion v1.5 adapter."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from signal_diffusion.diffusion.config import DiffusionConfig
from signal_diffusion.diffusion.data import DiffusionBatch
from signal_diffusion.diffusion.models.base import DiffusionAdapter, DiffusionModules, registry
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
