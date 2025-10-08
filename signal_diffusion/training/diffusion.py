"""Unified diffusion training entrypoint."""
from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

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
from signal_diffusion.diffusion.models import registry
from signal_diffusion.log_setup import get_logger

from signal_diffusion.training.diffusion_utils import (
    build_optimizer,
    flatten_and_sanitize_hparams,
    resolve_output_dir,
    run_evaluation,
    should_evaluate,
)


app = typer.Typer(add_completion=False, no_args_is_help=True)


LOGGER = get_logger(__name__)


@app.command()
def train(
    config_path: Path = typer.Argument(..., help="Path to diffusion training TOML"),
    output_dir: Optional[Path] = typer.Option(None, help="Override the configured output directory"),
    resume_from_checkpoint: Optional[Path] = typer.Option(None, help="Path to a checkpoint directory to resume training from"),
) -> None:
    torch.multiprocessing.set_start_method('spawn', force=True)
    cfg = load_diffusion_config(config_path)
    run_dir = resolve_output_dir(cfg, config_path, output_dir)
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
                                          automatic_checkpoint_naming=False,
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
        hps = flatten_and_sanitize_hparams(asdict(cfg))
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

    optimizer = build_optimizer(modules.parameters, cfg)

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
        torch.cuda.empty_cache()
        eval_metrics = run_evaluation(
            accelerator=accelerator,
            adapter=adapter,
            cfg=cfg,
            modules=modules,
            train_loader=train_loader,
            val_loader=val_loader,
            run_dir=run_dir,
            global_step=0,
        )
        torch.cuda.empty_cache()
        if eval_metrics:
            accelerator.log(eval_metrics, step=0)
            LOGGER.info("Initial evaluation metrics: %s", eval_metrics)
    accelerator.wait_for_everyone()

    first_epoch = 0
    global_step = 0

    if resume_from_checkpoint:
        if not resume_from_checkpoint.is_dir():
            raise ValueError(f"Checkpoint path {resume_from_checkpoint} is not a directory.")

        accelerator.load_state(resume_from_checkpoint)
        LOGGER.info(f"Resumed from checkpoint: {resume_from_checkpoint}")

        # Extract global_step from the checkpoint path
        resume_step_str = resume_from_checkpoint.name.replace("checkpoint-", "")
        try:
            global_step = int(resume_step_str)
        except ValueError:
            raise ValueError(f"Could not parse global step from checkpoint path: {resume_from_checkpoint}")

        # Calculate first_epoch
        num_update_steps_per_epoch = math.ceil(len(train_loader) / cfg.training.gradient_accumulation_steps)
        first_epoch = global_step // num_update_steps_per_epoch

        LOGGER.info(f"Resuming training from global step {global_step}, epoch {first_epoch}")

    progress_bar = None
    if accelerator.is_main_process:
        progress_bar = tqdm(
            total=max_train_steps,
            initial=global_step,
            desc=f"Epoch {first_epoch + 1}/{cfg.training.epochs}",
            dynamic_ncols=True,
        )

    for epoch in range(first_epoch, cfg.training.epochs):
        modules.denoiser.train()
        if accelerator.is_main_process:
            LOGGER.info("Epoch %d/%d", epoch + 1, cfg.training.epochs)
            if progress_bar is not None:
                progress_bar.set_description(f"Epoch {epoch + 1}/{cfg.training.epochs}")
        grad_norm = 0
        grad_norm_steps = 0
        for batch in train_loader:
            with accelerator.accumulate(modules.denoiser):
                loss, metrics = adapter.training_step(accelerator, cfg, modules, batch)
                accelerator.backward(loss)
                if cfg.training.gradient_clip_norm is not None:
                    clip_params = [param for group in optimizer.param_groups for param in group["params"]]
                    clip_norm = accelerator.clip_grad_norm_(clip_params, cfg.training.gradient_clip_norm)
                    if clip_norm:
                        grad_norm += clip_norm.item()
                        grad_norm_steps += 1
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                log_payload = {
                    "train/loss": accelerator.gather_for_metrics(loss.detach()).mean().item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                }
                if grad_norm > 0:
                    log_payload.update({"train/grad_norm": grad_norm / grad_norm_steps})
                    grad_norm = 0
                    grad_norm_steps = 0
                if metrics:
                    log_payload.update({f"train/{k}": float(v) for k, v in metrics.items()})
                accelerator.log(log_payload, step=global_step)

                postfix_payload: dict[str, str] = {}
                train_loss = log_payload.get("train/loss")
                if train_loss is not None:
                    postfix_payload["train_loss"] = f"{float(train_loss):.4f}"

                if cfg.training.checkpoint_interval and global_step % cfg.training.checkpoint_interval == 0:
                    checkpoints_dir = run_dir / "checkpoints"
                    save_dir = checkpoints_dir / f"checkpoint-{global_step}"
                    accelerator.save_state(str(save_dir))
                    if accelerator.is_main_process:
                        LOGGER.info("Saved checkpoint at step %d to %s", global_step, save_dir)

                        if cfg.training.checkpoint_total_limit is not None and cfg.training.checkpoint_total_limit > 0:
                            # Get all checkpoint directories
                            saved_checkpoints = sorted(
                                [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
                                key=lambda x: int(x.name.split("-")[-1]),
                            )

                            # Remove oldest checkpoints if limit is exceeded
                            while len(saved_checkpoints) > cfg.training.checkpoint_total_limit:
                                checkpoint_to_delete = saved_checkpoints.pop(0)
                                LOGGER.info(f"Deleting old checkpoint: {checkpoint_to_delete}")
                                shutil.rmtree(checkpoint_to_delete)

                run_eval_now = should_evaluate(
                    global_step, len(train_loader), cfg.training
                )

                if run_eval_now and val_loader is not None:
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
                if run_eval_now:
                    if accelerator.is_main_process:
                        torch.cuda.empty_cache()
                        eval_metrics = run_evaluation(
                            accelerator=accelerator,
                            adapter=adapter,
                            cfg=cfg,
                            modules=modules,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            run_dir=run_dir,
                            global_step=global_step,
                        )
                        torch.cuda.empty_cache()
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

    # Training finished
    if accelerator.is_main_process:
        LOGGER.info("Running final evaluation...")
        torch.cuda.empty_cache()
        eval_metrics = run_evaluation(
            accelerator=accelerator,
            adapter=adapter,
            cfg=cfg,
            modules=modules,
            train_loader=train_loader,
            val_loader=val_loader,
            run_dir=run_dir,
            global_step=global_step,
        )
        torch.cuda.empty_cache()
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
