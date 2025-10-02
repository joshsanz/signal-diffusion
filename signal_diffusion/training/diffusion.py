"""Unified diffusion training entrypoint."""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import torch
import typer
from accelerate import Accelerator
from accelerate.utils import LoggerType
from accelerate.utils import ProjectConfiguration
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

from signal_diffusion.diffusion import load_diffusion_config
from signal_diffusion.diffusion.data import build_dataloaders
from signal_diffusion.diffusion.eval_utils import compute_kid_score, save_image_grid
from signal_diffusion.diffusion.models import registry
from signal_diffusion.log_setup import get_logger


app = typer.Typer(add_completion=False, no_args_is_help=True)


LOGGER = get_logger(__name__)





def _resolve_output_dir(cfg, config_path: Path, override: Optional[Path]) -> Path:
    if override is not None:
        return override
    if cfg.training.output_dir is not None:
        return Path(cfg.training.output_dir)
    runs_root = Path("runs") / "diffusion"
    runs_root.mkdir(parents=True, exist_ok=True)
    return (runs_root / config_path.stem)


def _build_optimizer(parameters: Iterable[torch.nn.Parameter], cfg) -> torch.optim.Optimizer:
    params = list(parameters)
    if not params:
        raise ValueError("No trainable parameters found for optimizer")
    return torch.optim.AdamW(
        params,
        lr=cfg.optimizer.learning_rate,
        betas=cfg.optimizer.betas,
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay,
    )


def _run_evaluation(
    accelerator: Accelerator,
    adapter,
    cfg,
    modules,
    train_loader,
    val_loader,
    run_dir: Path,
    global_step: int,
):
    eval_examples = getattr(cfg.training, "eval_num_examples", 0) or 0
    eval_mmd_samples = getattr(cfg.training, "eval_mmd_samples", 0) or 0
    if eval_examples <= 0 and eval_mmd_samples <= 0:
        return {}

    num_generate = max(eval_examples, eval_mmd_samples)
    num_generate = max(1, num_generate)

    eval_gen_seed = getattr(cfg.training, "eval_gen_seed", None)
    generator = None
    if eval_gen_seed is not None:
        generator = torch.Generator(device=accelerator.device).manual_seed(eval_gen_seed)

    eval_batch_size = cfg.training.eval_batch_size or cfg.dataset.batch_size
    num_batches = math.ceil(num_generate / eval_batch_size)

    was_training = modules.denoiser.training
    modules.denoiser.eval()
    generated_samples = []
    try:
        with torch.no_grad():
            pbar = tqdm(
                range(num_batches),
                desc="Generating samples",
                disable=not accelerator.is_main_process,
                dynamic_ncols=True,
                leave=True,
            )
            for i in pbar:
                batch_size = min(eval_batch_size, num_generate - i * eval_batch_size)
                if batch_size <= 0:
                    continue

                generated_batch = adapter.generate_samples(
                    accelerator,
                    cfg,
                    modules,
                    batch_size,
                    denoising_steps=cfg.inference.denoising_steps,
                    cfg_scale=cfg.inference.cfg_scale,
                    generator=generator,
                )
                generated_samples.append(generated_batch.cpu())
        generated = torch.cat(generated_samples, dim=0)
    finally:
        if was_training:
            modules.denoiser.train()
        else:
            modules.denoiser.eval()

    if generated.ndim != 4:
        raise ValueError("Adapter.generate_samples must return a tensor shaped (N, C, H, W)")

    metrics: dict[str, float] = {}

    if eval_examples > 0:
        grid_images = generated[:eval_examples].detach().cpu()
        grid_path = run_dir / f"{global_step:06d}.jpg"
        save_image_grid(grid_images, grid_path, cols=4)
        accelerator.log({"eval/generated_samples": grid_images}, step=global_step)
        tb_tracker = accelerator.get_tracker("tensorboard")
        print(tb_tracker)
        if tb_tracker:
            tb_tracker.log_images({"eval/generated_samples": grid_images}, step=global_step) # type: ignore

    if eval_mmd_samples > 0:
        gen_for_kid = generated[:eval_mmd_samples]
        if gen_for_kid.shape[0] == 0:
            raise ValueError("Not enough generated samples to compute KID score")

        if val_loader is not None:
            ref_dataset = val_loader.dataset
        else:
            ref_dataset = train_loader.dataset

        kid_mean, kid_std = compute_kid_score(gen_for_kid, ref_dataset)
        metrics["eval/kid_mean"] = kid_mean
        metrics["eval/kid_std"] = kid_std

    return metrics


def _flatten_and_sanitize_hparams(config_dict: Any, prefix: str = "") -> dict:
    hparams = {}
    if not isinstance(config_dict, dict):
        return hparams

    for k, v in config_dict.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            hparams.update(_flatten_and_sanitize_hparams(v, prefix=key))
        elif isinstance(v, (str, int, float, bool)):
            hparams[key] = v
        elif isinstance(v, (list, tuple)):
            # Convert lists/tuples of simple types to a string
            if all(isinstance(item, (str, int, float, bool)) for item in v):
                hparams[key] = str(v)
    return hparams


def _should_evaluate(
    global_step: int,
    step_in_epoch: int,
    steps_per_epoch: int,
    training_cfg,
) -> bool:
    strategy = getattr(training_cfg, "eval_strategy", "epoch") or "epoch"
    strategy = strategy.strip().lower()
    if strategy == "epoch":
        return step_in_epoch == (steps_per_epoch - 1)
    if strategy == "steps":
        interval = getattr(training_cfg, "eval_num_steps", 0) or 0
        return interval > 0 and global_step > 0 and global_step % interval == 0
    return False


@app.command()
def train(
    config_path: Path = typer.Argument(..., help="Path to diffusion training TOML"),
    output_dir: Optional[Path] = typer.Option(None, help="Override the configured output directory"),
) -> None:
    torch.multiprocessing.set_start_method('spawn', force=True)
    cfg = load_diffusion_config(config_path)
    run_dir = _resolve_output_dir(cfg, config_path, output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.training.output_dir = run_dir

    adapter = registry.get(cfg.model.name)

    default_conditioning = "caption" if cfg.model.name.startswith("stable-diffusion") else "none"
    conditioning_value = cfg.model.conditioning
    if conditioning_value is None:
        conditioning_value = cfg.model.extras.get("conditioning", default_conditioning)
    conditioning = str(conditioning_value).strip().lower()
    if not conditioning:
        conditioning = default_conditioning
    if conditioning not in {"none", "caption", "classes"}:
        raise ValueError(f"Unsupported conditioning type '{conditioning}'")

    if conditioning == "caption":
        if not cfg.dataset.caption_column:
            raise ValueError("Caption conditioning requires 'dataset.caption_column' to be set")
    if conditioning == "classes":
        if cfg.dataset.num_classes <= 1:
            raise ValueError("Class conditioning requires 'dataset.num_classes' to be greater than 1")
        if not cfg.dataset.class_column:
            raise ValueError("Class conditioning requires 'dataset.class_column' to be set")

    tokenizer = adapter.create_tokenizer(cfg) if conditioning == "caption" else None

    log_with = []
    if getattr(cfg.logging, "tensorboard", False):
        log_with.append(LoggerType.TENSORBOARD)
    if getattr(cfg.logging, "wandb_project", None):
        log_with.append("wandb")

    # Initialize accelerator and logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.logging.run_name}-{timestamp}" if cfg.logging.run_name else f"signal_diffusion-{timestamp}"
    log_dir = str(cfg.logging.log_dir / run_name
                  if cfg.logging.log_dir
                  else run_dir / "tensorboard")
    LOGGER.info(f"Tracking run name {run_name}")
    LOGGER.info(f"Tracking log dir {log_dir}")
    project_config = ProjectConfiguration(project_dir=str(run_dir),
                                          logging_dir=log_dir,
                                          automatic_checkpoint_naming=True,
                                          total_limit=cfg.training.checkpoint_total_limit)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        project_config=project_config,
        log_with=log_with if log_with else None,
    )

    if accelerator.is_main_process and log_with:
        # A simple dict of hyperparameters for logging
        project_name = cfg.logging.wandb_project or "signal_diffusion"
        hps = _flatten_and_sanitize_hparams(asdict(cfg))
        accelerator.init_trackers(
            project_name,
            config=hps,
            init_kwargs={
                # "tensorboard": {"logging_dir": log_dir},
                "wandb": {"run_name": run_name},
            },
        )
    set_seed(cfg.training.seed)

    if accelerator.is_main_process:
        LOGGER.info("Launching diffusion training run")
        LOGGER.info("config=%s output_dir=%s", config_path, run_dir)
        LOGGER.info(
            "model=%s conditioning=%s mixed_precision=%s",
            cfg.model.name,
            conditioning,
            cfg.training.mixed_precision,
        )
        LOGGER.info(
            "epochs=%d grad_accum=%d max_train_steps=%s batch_size=%d eval_batch_size=%d",
            cfg.training.epochs,
            cfg.training.gradient_accumulation_steps,
            cfg.training.max_train_steps or "auto",
            cfg.dataset.batch_size,
            cfg.training.eval_batch_size or cfg.dataset.batch_size,
        )

    train_loader, val_loader = build_dataloaders(cfg.dataset, tokenizer=tokenizer, settings_path=cfg.settings_config)
    modules = adapter.build_modules(accelerator, cfg, tokenizer=tokenizer)

    optimizer = _build_optimizer(modules.parameters, cfg)

    num_update_steps_per_epoch = math.ceil(len(train_loader) / cfg.training.gradient_accumulation_steps)
    max_train_steps = cfg.training.max_train_steps or (cfg.training.epochs * num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        cfg.scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=cfg.scheduler.warmup_steps,
        num_training_steps=max_train_steps,
    )

    objects_to_prepare: list[Any] = [modules.denoiser, optimizer, train_loader, lr_scheduler]
    prepare_text_encoder = modules.text_encoder is not None
    if prepare_text_encoder:
        objects_to_prepare.insert(1, modules.text_encoder)
    if val_loader is not None:
        objects_to_prepare.append(val_loader)

    prepared = list(accelerator.prepare(*objects_to_prepare))

    idx = 0
    modules.denoiser = prepared[idx]
    idx += 1
    if prepare_text_encoder:
        modules.text_encoder = prepared[idx]
        idx += 1
    optimizer = prepared[idx]
    idx += 1
    train_loader = prepared[idx]
    idx += 1
    lr_scheduler = prepared[idx]
    idx += 1
    if val_loader is not None:
        val_loader = prepared[idx]

    modules.parameters = [param for group in optimizer.param_groups for param in group["params"]]


    if accelerator.is_main_process:
        with (run_dir / "config.json").open("w", encoding="utf-8") as fp:
            json.dump(asdict(cfg), fp, default=str, indent=2)
        train_example_count = None
        train_dataset = getattr(train_loader, "dataset", None)
        if train_dataset is not None:
            try:
                train_example_count = len(train_dataset)
            except TypeError:
                train_example_count = None
        val_example_count = None
        if val_loader is not None:
            val_dataset = getattr(val_loader, "dataset", None)
            if val_dataset is not None:
                try:
                    val_example_count = len(val_dataset)
                except TypeError:
                    val_example_count = None
        LOGGER.info(
            "train_batches=%d val_batches=%s train_samples=%s val_samples=%s",
            len(train_loader),
            len(val_loader) if val_loader is not None else 0,
            train_example_count if train_example_count is not None else "unknown",
            val_example_count if val_example_count is not None else "unknown",
        )

    if accelerator.is_main_process:
        LOGGER.info("Running initial evaluation...")
        eval_metrics = _run_evaluation(
            accelerator=accelerator,
            adapter=adapter,
            cfg=cfg,
            modules=modules,
            train_loader=train_loader,
            val_loader=val_loader,
            run_dir=run_dir,
            global_step=0,
        )
        if eval_metrics:
            accelerator.log(eval_metrics, step=0)
            LOGGER.info("Initial evaluation metrics: %s", eval_metrics)
    accelerator.wait_for_everyone()

    progress_bar = None
    if accelerator.is_main_process:
        progress_bar = tqdm(
            total=max_train_steps,
            desc=f"Epoch 1/{cfg.training.epochs}",
            dynamic_ncols=True,
        )

    global_step = 0
    for epoch in range(cfg.training.epochs):
        modules.denoiser.train()
        if accelerator.is_main_process:
            LOGGER.info("Epoch %d/%d", epoch + 1, cfg.training.epochs)
            if progress_bar is not None:
                progress_bar.set_description(f"Epoch {epoch + 1}/{cfg.training.epochs}")
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(modules.denoiser):
                loss, metrics = adapter.training_step(accelerator, cfg, modules, batch)
                accelerator.backward(loss)
                if cfg.training.gradient_clip_norm is not None:
                    clip_params = [param for group in optimizer.param_groups for param in group["params"]]
                    accelerator.clip_grad_norm_(clip_params, cfg.training.gradient_clip_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                log_payload = {"train/loss": accelerator.gather_for_metrics(loss.detach()).mean().item()}
                if metrics:
                    log_payload.update({f"train/{k}": float(v) for k, v in metrics.items()})
                accelerator.log(log_payload, step=global_step)

                postfix_payload: dict[str, str] = {}
                train_loss = log_payload.get("train/loss")
                if train_loss is not None:
                    postfix_payload["train_loss"] = f"{float(train_loss):.4f}"

                if cfg.training.checkpoint_interval and global_step % cfg.training.checkpoint_interval == 0:
                    save_dir = run_dir / f"checkpoint-{global_step}"
                    accelerator.save_state(str(save_dir))
                    if accelerator.is_main_process:
                        LOGGER.info("Saved checkpoint at step %d to %s", global_step, save_dir)

                should_evaluate = _should_evaluate(
                    global_step, step, len(train_loader), cfg.training
                )

                if should_evaluate and val_loader is not None:
                    modules.denoiser.eval()
                    val_losses: list[float] = []
                    with torch.no_grad():
                        for val_batch in val_loader:
                            val_metrics = adapter.validation_step(accelerator, cfg, modules, val_batch)
                            val_losses.append(val_metrics.get("loss", 0.0))
                    mean_val = sum(val_losses) / max(1, len(val_losses))
                    metrics_map = {"val/loss": mean_val}
                    accelerator.log(metrics_map, step=global_step)
                    if accelerator.is_main_process:
                        LOGGER.info("Validation step %d metrics=%s", global_step, metrics_map)
                    modules.denoiser.train()
                    postfix_payload["val_loss"] = f"{mean_val:.4f}"

                eval_metrics: dict[str, float] = {}
                if should_evaluate:
                    if accelerator.is_main_process:
                        eval_metrics = _run_evaluation(
                            accelerator=accelerator,
                            adapter=adapter,
                            cfg=cfg,
                            modules=modules,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            run_dir=run_dir,
                            global_step=global_step,
                        )
                    accelerator.wait_for_everyone()
                    if eval_metrics and accelerator.is_main_process:
                        accelerator.log(eval_metrics, step=global_step)
                        kid_score = eval_metrics.get("eval/kid_mean")
                        if kid_score is not None:
                            postfix_payload["kid"] = f"{kid_score:.4f}"
                if accelerator.is_main_process and progress_bar is not None:
                    progress_bar.update(1)
                    if postfix_payload:
                        progress_bar.set_postfix(**postfix_payload)

            if global_step >= max_train_steps:
                break
        if global_step >= max_train_steps:
            break

    if accelerator.is_main_process:
        LOGGER.info("Running final evaluation...")
        eval_metrics = _run_evaluation(
            accelerator=accelerator,
            adapter=adapter,
            cfg=cfg,
            modules=modules,
            train_loader=train_loader,
            val_loader=val_loader,
            run_dir=run_dir,
            global_step=global_step,
        )
        if eval_metrics:
            accelerator.log(eval_metrics, step=global_step)
            LOGGER.info("Final evaluation metrics: %s", eval_metrics)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        final_dir = run_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        adapter.save_checkpoint(accelerator, cfg, modules, str(final_dir))
        LOGGER.info("Saved final checkpoint to %s", final_dir)
        LOGGER.info("Completed training after %d steps", global_step)
        if progress_bar is not None:
            progress_bar.close()


if __name__ == "__main__":  # pragma: no cover
    app()
