"""Unified diffusion training entrypoint."""
from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager, nullcontext
from typing import Any, Optional

import torch
import typer
from accelerate import Accelerator
from accelerate.utils import LoggerType
from accelerate.utils import ProjectConfiguration
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
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


app = typer.Typer(add_completion=False, no_args_is_help=True,
                  pretty_exceptions_show_locals=False)


LOGGER = get_logger(__name__)


def ema_weights_context(accelerator: Accelerator, modules: Any):
    if getattr(modules, "ema", None) is None:
        return nullcontext()

    unwrapped = accelerator.unwrap_model(modules.denoiser)

    @contextmanager
    def _apply_ema():
        modules.ema.store(unwrapped.parameters())
        modules.ema.copy_to(unwrapped.parameters())
        try:
            yield
        finally:
            modules.ema.restore(unwrapped.parameters())

    return _apply_ema()


@app.command()
def train(
    config_path: Path = typer.Argument(..., help="Path to diffusion training TOML"),
    output_dir: Optional[Path] = typer.Option(None, help="Override the configured output directory"),
    resume_from_checkpoint: Optional[Path] = typer.Option(None, help="Path to a checkpoint directory to resume training from"),
    max_train_steps: Optional[int] = typer.Option(
        None,
        "--max-train-steps",
        "--max_train_steps",
        help="Override training.max_train_steps from the config.",
    ),
) -> None:
    torch.multiprocessing.set_start_method('spawn', force=True)
    cfg = load_diffusion_config(config_path)
    if max_train_steps is not None:
        cfg.training.max_train_steps = max_train_steps
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
    if conditioning not in {"none", "caption", "classes", "gend_hlth_age"}:
        raise ValueError(
            f"Unsupported conditioning type '{conditioning}'. "
            f"Must be one of: 'none', 'classes', 'caption', 'gend_hlth_age'"
        )

    if conditioning == "caption":
        if not cfg.dataset.caption_column:
            raise ValueError("caption conditioning requires 'dataset.caption_column' to be set")

    elif conditioning == "classes":
        if cfg.dataset.num_classes <= 1:
            raise ValueError("classes conditioning requires 'dataset.num_classes' > 1")
        if not cfg.dataset.class_column:
            raise ValueError("classes conditioning requires 'dataset.class_column' to be set")

    elif conditioning == "gend_hlth_age":
        if not cfg.dataset.gender_column:
            raise ValueError("gend_hlth_age conditioning requires 'dataset.gender_column' to be set")
        if not cfg.dataset.health_column:
            raise ValueError("gend_hlth_age conditioning requires 'dataset.health_column' to be set")
        if not cfg.dataset.age_column:
            raise ValueError("gend_hlth_age conditioning requires 'dataset.age_column' to be set")

    tokenizer = adapter.create_tokenizer(cfg) if conditioning == "caption" else None

    is_timeseries = bool(cfg.settings and getattr(cfg.settings, "data_type", "") == "timeseries")

    if is_timeseries:
        # ------------------------------------------------------------------
        # Auto-populate n_eeg_channels from normalization stats (optional)
        # ------------------------------------------------------------------
        # For time-series training, we need to know the number of EEG channels.
        # If using a configured dataset identifier (from default.toml), we can
        # auto-load this from the {dataset}_normalization_stats.json file.
        # Otherwise, users must set n_eeg_channels manually in [dataset.extras].
        #
        # NOTE: This is separate from data normalization itself, which happens
        # in the dataset classes (e.g., SEEDTimeseriesDataset). This is just
        # for getting the channel count dimension.
        extras = cfg.dataset.extras
        if extras is None:
            raise ValueError(
                "Time-series diffusion requires [dataset.extras] section in config with "
                "'n_eeg_channels' and 'sequence_length'. Run preprocessing first to get n_eeg_channels."
            )

        settings = cfg.settings
        dataset_key = cfg.dataset.identifier
        dataset_base = dataset_key.replace("_timeseries", "")
        lookup_key = dataset_key if dataset_key in settings.datasets else dataset_base
        LOGGER.debug(
            "Attempting to auto-load n_eeg_channels for time-series training. "
            "Dataset identifier: '%s', lookup key: '%s'",
            dataset_key,
            lookup_key,
        )
        dataset_settings = None
        if cfg.settings_config:
            try:
                dataset_settings = settings.dataset(lookup_key)
            except KeyError:
                LOGGER.info(
                    "Dataset identifier '%s' not found in settings. "
                    "Unable to auto-load n_eeg_channels from normalization stats.",
                    lookup_key,
                )
                LOGGER.info(
                    "To resolve: (1) Set 'n_eeg_channels' manually in [dataset.extras] "
                    "(e.g., n_eeg_channels = 20 for SEED), or "
                    "(2) Use a configured dataset identifier from config/default.toml "
                    "(e.g., 'seed_timeseries' instead of absolute path)."
                )

        if dataset_settings is not None:
            stats_file = f"{dataset_base}_normalization_stats.json"
            stats_path = dataset_settings.output / stats_file
            if stats_path.exists():
                try:
                    with stats_path.open("r", encoding="utf-8") as f:
                        norm_stats = json.load(f)
                    if "n_eeg_channels" not in extras:
                        n_channels = norm_stats.get("n_eeg_channels")
                        if n_channels is not None:
                            extras["n_eeg_channels"] = n_channels
                            LOGGER.info(
                                "Loaded n_eeg_channels=%s from %s",
                                extras["n_eeg_channels"],
                                stats_path.name,
                            )
                except (OSError, json.JSONDecodeError) as exc:
                    LOGGER.warning("Failed to load normalization stats from %s: %s", stats_path, exc)
            else:
                LOGGER.warning(
                    "Normalization stats not found at %s. "
                    "Run preprocessing first or set 'n_eeg_channels' in [dataset.extras].",
                    stats_path,
                )

        if "sequence_length" not in extras:
            extras["sequence_length"] = cfg.dataset.resolution
            LOGGER.info("Auto-populated sequence_length=%s from dataset.resolution", cfg.dataset.resolution)

        if "n_eeg_channels" not in extras:
            raise ValueError(
                "Time-series diffusion requires 'n_eeg_channels' in [dataset.extras]. "
                "This value specifies the number of EEG channels in your data. "
                "Either: (1) Set it manually in config (e.g., n_eeg_channels = 20 for SEED), or "
                "(2) Use a dataset identifier from config/default.toml (e.g., 'seed_timeseries') "
                "instead of an absolute path to enable auto-loading from normalization stats."
            )

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

    # Enable TF32 for faster matmul on Ampere+ GPUs with minimal accuracy impact
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if accelerator.is_main_process:
            LOGGER.info("TF32 enabled for CUDA matmul and cuDNN operations")

    best_eval_dir = run_dir / "best_eval"
    best_eval_metadata_path = best_eval_dir / "metadata.json"
    best_kid_checkpoints: list[tuple[float, int]] = []

    if accelerator.is_main_process:
        best_eval_dir.mkdir(parents=True, exist_ok=True)

    def write_best_eval_metadata() -> None:
        if not accelerator.is_main_process:
            return
        payload = [{"step": step, "kid_mean": score} for score, step in best_kid_checkpoints]
        with best_eval_metadata_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)

    if accelerator.is_main_process:
        stored_entries: list[dict[str, Any]] = []
        if best_eval_metadata_path.exists():
            try:
                with best_eval_metadata_path.open("r", encoding="utf-8") as fp:
                    data = json.load(fp)
                if isinstance(data, list):
                    stored_entries = data
            except (json.JSONDecodeError, OSError) as exc:
                LOGGER.warning("Failed to load existing best-eval metadata from %s: %s", best_eval_metadata_path, exc)
        elif best_eval_dir.exists():
            for checkpoint_dir in sorted(
                (
                    d
                    for d in best_eval_dir.iterdir()
                    if d.is_dir() and d.name.startswith("checkpoint-")
                ),
                key=lambda path: int(path.name.split("-")[-1]) if path.name.split("-")[-1].isdigit() else 0,
            ):
                metrics_path = checkpoint_dir / "metrics.json"
                if not metrics_path.exists():
                    continue
                try:
                    with metrics_path.open("r", encoding="utf-8") as fp:
                        metrics_payload = json.load(fp)
                    stored_entries.append(metrics_payload)
                except (OSError, json.JSONDecodeError):
                    continue
        for entry in stored_entries:
            try:
                step = int(entry["step"])
                kid_mean = float(entry["kid_mean"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(kid_mean):
                best_kid_checkpoints.append((kid_mean, step))
        best_kid_checkpoints.sort(key=lambda item: (item[0], item[1]))
        if len(best_kid_checkpoints) > 3:
            for _, stale_step in best_kid_checkpoints[3:]:
                stale_dir = best_eval_dir / f"checkpoint-{stale_step}"
                if stale_dir.exists():
                    shutil.rmtree(stale_dir)
            del best_kid_checkpoints[3:]
            write_best_eval_metadata()

    if accelerator.is_main_process and log_with:
        project_name = cfg.logging.wandb_project or "signal_diffusion"
        init_kwargs: dict[str, Any] = {}
        if cfg.logging.wandb_project:
            init_kwargs["wandb"] = {"run_name": run_name}
        accelerator.init_trackers(
            project_name,
            init_kwargs=init_kwargs,
        )
        if cfg.logging.wandb_project:
            hps = flatten_and_sanitize_hparams(asdict(cfg))
            wandb_tracker = accelerator.get_tracker("wandb")
            if wandb_tracker is not None:
                wandb_run = getattr(wandb_tracker, "run", None)
                if wandb_run is not None:
                    wandb_run.config.update(hps, allow_val_change=True)
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
        if cfg.training.gradient_checkpointing:
            LOGGER.info("Gradient checkpointing is enabled")

    train_loader, val_loader = build_dataloaders(cfg.dataset, tokenizer=tokenizer, settings_path=cfg.settings_config)
    modules = adapter.build_modules(accelerator, cfg, tokenizer=tokenizer)

    ema_model: EMAModel | None = None
    if cfg.training.ema_decay is not None and cfg.training.ema_decay > 0:
        ema_model = EMAModel(
            modules.denoiser.parameters(),
            decay=float(cfg.training.ema_decay),
            inv_gamma=float(cfg.training.ema_inv_gamma),
            power=float(cfg.training.ema_power),
            update_after_step=int(cfg.training.ema_update_after_step),
            use_ema_warmup=bool(cfg.training.ema_use_ema_warmup),
            model_cls=type(modules.denoiser),
            model_config=getattr(modules.denoiser, "config", None),
        )
        modules.ema = ema_model
        if accelerator.is_main_process:
            LOGGER.info(
                "Initialized EMA with decay=%.5f inv_gamma=%.3f power=%.3f update_after_step=%d warmup=%s",
                float(cfg.training.ema_decay),
                float(cfg.training.ema_inv_gamma),
                float(cfg.training.ema_power),
                int(cfg.training.ema_update_after_step),
                bool(cfg.training.ema_use_ema_warmup),
            )

    def save_best_checkpoint(step: int, kid_mean: float) -> None:
        if not accelerator.is_main_process:
            return
        best_eval_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = best_eval_dir / f"checkpoint-{step}"
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        adapter.save_checkpoint(accelerator, cfg, modules, str(checkpoint_dir))
        if modules.ema is not None:
            ema_dir = checkpoint_dir / "ema"
            ema_dir.mkdir(exist_ok=True)
            with ema_weights_context(accelerator, modules):
                adapter.save_checkpoint(accelerator, cfg, modules, str(ema_dir))
        metrics_path = checkpoint_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as fp:
            json.dump({"step": step, "kid_mean": kid_mean}, fp, indent=2)
        LOGGER.info(
            "Saved best-eval checkpoint (step=%d, kid_mean=%.6f) to %s",
            step,
            kid_mean,
            checkpoint_dir,
        )

    def update_best_checkpoints(kid_mean: float | None, step: int) -> None:
        if not accelerator.is_main_process or kid_mean is None:
            return
        try:
            kid_value = float(kid_mean)
        except (TypeError, ValueError):
            return
        if not math.isfinite(kid_value):
            return
        existing_index: int | None = None
        existing_score: float | None = None
        for idx, (stored_score, stored_step) in enumerate(best_kid_checkpoints):
            if stored_step == step:
                existing_index = idx
                existing_score = stored_score
                break
        if existing_score is not None and kid_value >= existing_score:
            return
        if existing_index is not None:
            del best_kid_checkpoints[existing_index]
        should_add = len(best_kid_checkpoints) < 3
        if not should_add and best_kid_checkpoints:
            worst_score, worst_step = best_kid_checkpoints[-1]
            if kid_value < worst_score or (math.isclose(kid_value, worst_score) and step < worst_step):
                should_add = True
        if not should_add:
            return
        best_kid_checkpoints.append((kid_value, step))
        best_kid_checkpoints.sort(key=lambda item: (item[0], item[1]))
        while len(best_kid_checkpoints) > 3:
            removed_score, removed_step = best_kid_checkpoints.pop()
            stale_dir = best_eval_dir / f"checkpoint-{removed_step}"
            if stale_dir.exists():
                shutil.rmtree(stale_dir)
                LOGGER.info(
                    "Removed stale best-eval checkpoint at step %d (kid_mean=%.6f)",
                    removed_step,
                    removed_score,
                )
        save_best_checkpoint(step, kid_value)
        write_best_eval_metadata()

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

    if modules.ema is not None:
        modules.ema.to(accelerator.device)
        accelerator.register_for_checkpointing(modules.ema)


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
        with ema_weights_context(accelerator, modules):
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
            update_best_checkpoints(eval_metrics.get("eval/kid_mean"), 0)
    accelerator.wait_for_everyone()

    first_epoch = 0
    global_step = 0

    if resume_from_checkpoint:
        if not resume_from_checkpoint.is_dir():
            raise ValueError(f"Checkpoint path {resume_from_checkpoint} is not a directory.")

        accelerator.load_state(resume_from_checkpoint)
        if modules.ema is not None:
            modules.ema.to(accelerator.device)
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

            if modules.ema is not None and accelerator.sync_gradients:
                modules.ema.step(accelerator.unwrap_model(modules.denoiser).parameters())

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
                        with ema_weights_context(accelerator, modules):
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
                        update_best_checkpoints(kid_score, global_step)
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
        with ema_weights_context(accelerator, modules):
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
            update_best_checkpoints(eval_metrics.get("eval/kid_mean"), global_step)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        final_dir = run_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        adapter.save_checkpoint(accelerator, cfg, modules, str(final_dir))
        LOGGER.info("Saved final checkpoint to %s", final_dir)
        if modules.ema is not None:
            ema_dir = final_dir / "ema"
            with ema_weights_context(accelerator, modules):
                adapter.save_checkpoint(accelerator, cfg, modules, str(ema_dir))
            LOGGER.info("Saved EMA checkpoint to %s", ema_dir)
        LOGGER.info("Completed training after %d steps", global_step)
        if progress_bar is not None:
            progress_bar.close()


if __name__ == "__main__":  # pragma: no cover
    app()
