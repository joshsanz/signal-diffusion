"""Stable Diffusion v1.5 adapter."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers

try:  # pragma: no cover - depends on diffusers version
    from diffusers.models.cross_attention import LoRACrossAttnProcessor
except ImportError as exc:  # pragma: no cover - explicit guidance for incompatible versions
    raise ImportError(
        "LoRA training requires diffusers>=0.14 with 'LoRACrossAttnProcessor' available"
    ) from exc
from transformers import CLIPTextModel, CLIPTokenizer

from signal_diffusion.diffusion.config import DiffusionConfig
from signal_diffusion.diffusion.data import DiffusionBatch
from signal_diffusion.diffusion.models.base import DiffusionAdapter, DiffusionModules, registry


@dataclass(slots=True)
class StableDiffusionExtras:
    train_text_encoder: bool = False
    text_encoder_lr_scale: float = 1.0


class StableDiffusionAdapterV15:
    """Adapter targeting Stable Diffusion v1.5-style checkpoints."""

    name = "stable-diffusion-v1-5"

    def __init__(self) -> None:
        self._extras = StableDiffusionExtras()

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

        text_encoder = CLIPTextModel.from_pretrained(pretrained, subfolder="text_encoder", revision=revision)
        vae = AutoencoderKL.from_pretrained(pretrained, subfolder="vae", revision=revision)
        unet = UNet2DConditionModel.from_pretrained(pretrained, subfolder="unet", revision=revision)

        if cfg.objective.prediction_type != "epsilon":
            raise ValueError(
                "Stable Diffusion adapter only supports noise prediction ('epsilon') objective"
            )

        noise_scheduler = DDPMScheduler.from_pretrained(pretrained, subfolder="scheduler", revision=revision)
        noise_scheduler.register_to_config(prediction_type="epsilon")

        if cfg.training.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

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

        trainable_params: list[torch.nn.Parameter] = []
        clip_grad_params = None

        if cfg.model.lora.enabled:
            lora_attn_procs = {}
            for name in unet.attn_processors.keys():
                cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = unet.config.block_out_channels[block_id]
                else:
                    continue

            lora_attn_procs[name] = LoRACrossAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=cfg.model.lora.rank,
                network_alpha=cfg.model.lora.alpha,
                dropout=cfg.model.lora.dropout,
            )
            unet.set_attn_processor(lora_attn_procs)
            attn_layers = AttnProcsLayers(unet.attn_processors)
            lora_params = list(attn_layers.parameters())
            trainable_params.extend(lora_params)
            clip_grad_params = lora_params
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
        output_path = accelerator.project_configuration.project_dir / output_dir if accelerator.project_configuration else output_dir
        accelerator.print(f"Saving Stable Diffusion weights to {output_path}")
        if cfg.model.lora.enabled:
            modules.denoiser.save_attn_procs(output_dir)
        else:
            modules.denoiser.save_pretrained(output_dir)


registry.register(StableDiffusionAdapterV15())
