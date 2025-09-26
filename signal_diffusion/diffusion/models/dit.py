"""DiT adapter supporting noise prediction and flow matching objectives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL
from diffusers import DiTTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from signal_diffusion.diffusion.config import DiffusionConfig
from signal_diffusion.diffusion.data import DiffusionBatch
from signal_diffusion.diffusion.models.base import DiffusionAdapter, DiffusionModules, registry
from signal_diffusion.diffusion.objectives import (
    apply_min_gamma_snr,
    get_snr,
    sample_timestep_logitnorm,
    verify_scheduler,
)


@dataclass(slots=True)
class DiTExtras:
    latent_space: bool = False
    vae: str | None = None
    num_classes: int = 0
    cfg_dropout: float = 0.0


class DiTAdapter:
    name = "dit"

    def __init__(self) -> None:
        self._extras = DiTExtras()

    def create_tokenizer(self, cfg: DiffusionConfig):  # noqa: D401 - DiT has no tokenizer
        return None

    def _parse_extras(self, cfg: DiffusionConfig) -> DiTExtras:
        extras = cfg.model.extras
        latent_space = bool(extras.get("latent_space", False))
        vae = extras.get("vae")
        num_classes = int(extras.get("num_classes", 0))
        cfg_dropout = float(extras.get("cfg_dropout", 0.0))
        return DiTExtras(
            latent_space=latent_space,
            vae=vae,
            num_classes=num_classes,
            cfg_dropout=cfg_dropout,
        )

    def build_modules(
        self,
        accelerator: Accelerator,
        cfg: DiffusionConfig,
        tokenizer=None,
    ) -> DiffusionModules:
        self._extras = self._parse_extras(cfg)

        if self._extras.latent_space and not self._extras.vae:
            raise ValueError("DiT latent_space requires 'vae' path in model extras")

        if cfg.model.pretrained:
            model = DiTTransformer2DModel.from_pretrained(cfg.model.pretrained)
        else:
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
                num_embeds_ada_norm=int(cfg.model.extras.get("num_embeds_ada_norm", 1000)),
                upcast_attention=bool(cfg.model.extras.get("upcast_attention", False)),
                norm_type=str(cfg.model.extras.get("norm_type", "ada_norm_zero")),
                norm_elementwise_affine=bool(cfg.model.extras.get("norm_elementwise_affine", False)),
                norm_eps=float(cfg.model.extras.get("norm_eps", 1e-5)),
            )
            model = DiTTransformer2DModel(**kwargs)

        if cfg.model.lora.enabled:
            raise NotImplementedError("LoRA for DiT models is not yet implemented")

        noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=cfg.objective.flow_match_timesteps)
        noise_scheduler.register_to_config(prediction_type=cfg.objective.prediction_type)
        verify_scheduler(noise_scheduler)

        vae = None
        if self._extras.latent_space and self._extras.vae:
            vae = AutoencoderKL.from_pretrained(self._extras.vae)

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        model.to(accelerator.device, dtype=weight_dtype)
        if vae is not None:
            vae.requires_grad_(False)
            vae.to(accelerator.device, dtype=weight_dtype)

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

        images = batch.pixel_values.to(accelerator.device, dtype=modules.weight_dtype)
        if extras.latent_space:
            vae = modules.vae
            if vae is None:
                raise RuntimeError("VAE expected but not initialised")
            images = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor

        noise = torch.randn_like(images)
        timesteps = sample_timestep_logitnorm(
            images.shape[0],
            num_train_timesteps=scheduler.config.num_train_timesteps,
            timesteps=scheduler.timesteps,
            device=images.device,
        )
        z_t = scheduler.scale_noise(images, timesteps, noise)
        snr = get_snr(scheduler, timesteps, device=images.device)

        class_labels = None
        if batch.class_labels is not None and extras.num_classes > 0:
            class_labels = batch.class_labels.to(images.device)
            if extras.cfg_dropout > 0:
                mask = torch.rand_like(class_labels, dtype=torch.float32) < extras.cfg_dropout
                class_labels = class_labels.masked_fill(mask, extras.num_classes)
        else:
            class_labels = torch.zeros(images.shape[0], device=images.device, dtype=torch.long)

        if cfg.objective.prediction_type == "epsilon":
            target = noise
        elif cfg.objective.prediction_type == "vector_field":
            target = noise - images
        else:
            raise ValueError(f"Unsupported prediction type {cfg.objective.prediction_type} for DiT")

        model_pred = model(z_t, timesteps, y=class_labels).sample
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

    def save_checkpoint(
        self,
        accelerator: Accelerator,
        cfg: DiffusionConfig,
        modules: DiffusionModules,
        output_dir: str,
    ) -> None:
        modules.denoiser.save_pretrained(output_dir)


registry.register(DiTAdapter())
