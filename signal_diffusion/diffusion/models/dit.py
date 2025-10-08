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
from signal_diffusion.diffusion.models.base import DiffusionAdapter, DiffusionModules, registry
from signal_diffusion.diffusion.train_utils import (
    apply_min_gamma_snr,
    get_snr,
    sample_timestep_logitnorm,
    verify_scheduler,
)
from signal_diffusion.log_setup import get_logger


@dataclass(slots=True)
class DiTExtras:
    conditioning: str = "none"
    latent_space: bool = False
    vae: str | None = None
    text_encoder: str | None = None
    num_classes: int = 0
    cfg_dropout: float = 0.0
    timestep_embeddings: int = 1000


class DiTAdapter:
    name = "dit"

    def __init__(self) -> None:
        self._extras = DiTExtras()
        self._logger = get_logger(__name__)

    def create_tokenizer(self, cfg: DiffusionConfig):
        conditioning_value = cfg.model.conditioning
        if conditioning_value is None:
            conditioning_value = cfg.model.extras.get("conditioning", "none")
        conditioning = str(conditioning_value).strip().lower()
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
        conditioning_value = cfg.model.conditioning
        if conditioning_value is None:
            conditioning_value = extras.get("conditioning", "none")
        conditioning = str(conditioning_value).strip().lower() or "none"
        if conditioning not in {"none", "caption", "classes"}:
            raise ValueError(f"Unsupported conditioning type '{conditioning}' for DiT models")
        latent_space = bool(extras.get("latent_space", False))
        vae = extras.get("vae")
        num_classes_value = extras.get("num_classes", cfg.dataset.num_classes)
        num_classes = int(num_classes_value or 0)
        cfg_dropout = float(extras.get("cfg_dropout", 0.0))
        timestep_embeddings = int(cfg.objective.flow_match_timesteps or 1000)
        text_encoder = extras.get("text_encoder")
        return DiTExtras(
            conditioning=conditioning,
            latent_space=latent_space,
            vae=vae,
            text_encoder=text_encoder,
            num_classes=num_classes,
            cfg_dropout=cfg_dropout,
            timestep_embeddings=timestep_embeddings,
        )

    def build_modules(
        self,
        accelerator: Accelerator,
        cfg: DiffusionConfig,
        tokenizer=None,
    ) -> DiffusionModules:
        self._extras = self._parse_extras(cfg)

        if accelerator.is_main_process:
            self._logger.info(
                "Building DiT modules pretrained=%s conditioning=%s latent_space=%s",
                bool(cfg.model.pretrained),
                self._extras.conditioning,
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
            conditioning = self._extras.conditioning
            # Diffusers' DiT implementation only supports `ada_norm_zero` when a patch_size is used.
            # To keep unconditional runs working we stick with `ada_norm_zero` and feed dummy
            # class ids during the forward pass when no conditioning is required.
            norm_type = "ada_norm_zero"
            if conditioning == "classes":
                if self._extras.num_classes <= 0:
                    raise ValueError("Class conditioning requires 'model.extras.num_classes' to be greater than 0")
                num_embeds_ada_norm = self._extras.num_classes
            else:
                # unconditional / caption runs fall back to timestep embedding count
                num_embeds_ada_norm = self._extras.timestep_embeddings
            kwargs["norm_type"] = norm_type
            kwargs["num_embeds_ada_norm"] = num_embeds_ada_norm
            model = DiTTransformer2DModel(**kwargs)

        if cfg.model.lora.enabled:
            raise NotImplementedError("LoRA for DiT models is not yet implemented")

        noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=cfg.objective.flow_match_timesteps)
        # noise_scheduler.register_to_config(prediction_type=cfg.objective.prediction_type)
        verify_scheduler(noise_scheduler)

        if accelerator.is_main_process:
            self._logger.info(
                "Noise scheduler %s with %d timesteps and prediction_type=%s",
                type(noise_scheduler).__name__,
                cfg.objective.flow_match_timesteps,
                cfg.objective.prediction_type,
            )

        vae = None
        if self._extras.latent_space and self._extras.vae:
            vae = AutoencoderKL.from_pretrained(self._extras.vae)
            if accelerator.is_main_process:
                self._logger.info("Loaded VAE from %s", self._extras.vae)

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        model.to(accelerator.device, dtype=weight_dtype)
        if vae is not None:
            vae.requires_grad_(False)
            vae.to(accelerator.device, dtype=weight_dtype)

        if accelerator.is_main_process:
            self._logger.info(
                "DiT model moved to %s with dtype=%s", accelerator.device, str(weight_dtype)
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

        model_config = getattr(modules.denoiser, "config", None)
        channels = getattr(model_config, "in_channels", 3) if model_config is not None else 3
        sample_size = getattr(model_config, "sample_size", None)
        if sample_size is None:
            sample_size = int(cfg.model.sample_size or cfg.dataset.resolution)
        vae = modules.vae

        sample = torch.randn((num_images, channels, sample_size, sample_size), generator=generator, device=device, dtype=dtype)
        extras = self._extras

        if conditioning is not None and not isinstance(conditioning, torch.Tensor):
            raise ValueError("DiT adapter only supports tensor class-label conditioning during sampling")
        if conditioning is not None and extras.conditioning != "classes":
            raise ValueError("Class label conditioning is only enabled when 'conditioning' is set to 'classes'")
        if conditioning is None and extras.conditioning not in {"none", "classes"}:
            raise ValueError(f"Unsupported conditioning mode '{extras.conditioning}' for DiT sampling")

        classifier_free = conditioning is not None
        if conditioning is not None:
            class_labels = conditioning.to(device=device, dtype=torch.long)
            if class_labels.ndim != 1:
                raise ValueError("Class-label conditioning tensor must be 1D with shape (num_images,)")
            if class_labels.shape[0] == 1 and num_images > 1:
                class_labels = class_labels.expand(num_images)
            if class_labels.shape[0] != num_images:
                raise ValueError("Number of class labels must match num_images")
            dropout_label = extras.num_classes
            num_embeds = int(getattr(getattr(modules.denoiser, "config", None), "num_embeds_ada_norm", dropout_label + 1))
            if dropout_label >= num_embeds:
                raise ValueError(
                    "DiT classifier-free guidance expects an extra class embedding for dropout; "
                    "increase 'model.extras.num_classes' to include the dropout token."
                )
            unconditional_labels = torch.full_like(class_labels, dropout_label)
        else:
            class_labels = torch.zeros(num_images, device=device, dtype=torch.long)
            unconditional_labels = None

        with torch.no_grad():
            for timestep in tqdm(scheduler.timesteps, desc="Denoising", leave=False):
                if classifier_free:
                    model_input = torch.cat([sample, sample], dim=0)
                else:
                    model_input = sample
                if hasattr(scheduler, "scale_model_input"):
                    model_input = scheduler.scale_model_input(model_input, timestep)

                denoiser_timestep = timestep.expand(model_input.shape[0])
                if classifier_free and unconditional_labels is not None:
                    class_input = torch.cat([unconditional_labels, class_labels], dim=0)
                else:
                    class_input = class_labels

                model_output = modules.denoiser(
                    model_input, timestep=denoiser_timestep, class_labels=class_input
                ).sample

                if classifier_free and unconditional_labels is not None:
                    model_output_uncond, model_output_cond = model_output.chunk(2)
                    model_output = model_output_uncond + cfg_scale * (model_output_cond - model_output_uncond)

                step_output = scheduler.step(model_output, timestep, sample)
                sample = getattr(step_output, "prev_sample", step_output[0] if isinstance(step_output, tuple) else step_output)

        if self._extras.latent_space:
            if vae is None:
                raise RuntimeError("VAE expected for latent-space DiT sampling")
            sample = sample / getattr(vae.config, "scaling_factor", 1.0)
            with torch.no_grad():
                sample = vae.decode(sample.to(device=device, dtype=vae.dtype)).sample

        return sample.to(dtype=torch.float32).detach()


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

        class_labels: torch.Tensor | None
        if extras.conditioning == "classes":
            if batch.class_labels is None:
                raise ValueError("Class conditioning requires 'batch.class_labels'")
            class_labels = batch.class_labels.to(device=images.device, dtype=torch.long)
            if extras.cfg_dropout > 0:
                mask = torch.rand(class_labels.shape, device=class_labels.device) < extras.cfg_dropout
                class_labels = class_labels.masked_fill(mask, extras.num_classes)
        else:
            # Diffusers' DiT uses AdaLayerNormZero under the hood, which always expects
            # a label embedding. Provide a constant dummy tensor so unconditional runs
            # (conditioning="none") can skip classifier guidance while satisfying the API.
            class_labels = torch.zeros(images.shape[0], device=images.device, dtype=torch.long)

        if cfg.objective.prediction_type == "epsilon":
            target = noise
        elif cfg.objective.prediction_type == "vector_field":
            target = noise - images
        else:
            raise ValueError(f"Unsupported prediction type {cfg.objective.prediction_type} for DiT")

        model_pred = model(z_t, timestep=timesteps, class_labels=class_labels).sample
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
