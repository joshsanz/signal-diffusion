"""Unified diffusion training entrypoint."""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import torch
import typer
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler

from signal_diffusion.diffusion import load_diffusion_config
from signal_diffusion.diffusion.data import build_dataloaders
from signal_diffusion.diffusion.models import registry


app = typer.Typer(add_completion=False, no_args_is_help=True)


class DiffusionLogger:
    """Minimal logger supporting TensorBoard and Weights & Biases."""

    def __init__(self, logging_cfg, run_dir: Path) -> None:
        self._tensorboard = None
        self._wandb_run = None

        if logging_cfg.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("TensorBoard logging requested but not installed") from exc
            log_dir = logging_cfg.log_dir or (run_dir / "tensorboard")
            log_dir.mkdir(parents=True, exist_ok=True)
            self._tensorboard = SummaryWriter(str(log_dir))

        if logging_cfg.wandb_project:
            try:
                import wandb  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("Weights & Biases logging requested but not installed") from exc
            init_kwargs: dict[str, Any] = {
                "project": logging_cfg.wandb_project,
                "dir": str(run_dir),
            }
            if logging_cfg.wandb_run_name:
                init_kwargs["name"] = logging_cfg.wandb_run_name
            if logging_cfg.wandb_entity:
                init_kwargs["entity"] = logging_cfg.wandb_entity
            self._wandb_run = wandb.init(**init_kwargs)

    def log(self, step: int, metrics: Mapping[str, float]) -> None:
        if self._tensorboard is not None:
            for key, value in metrics.items():
                self._tensorboard.add_scalar(key, value, step)
        if self._wandb_run is not None:
            self._wandb_run.log(dict(metrics), step=step)

    def close(self) -> None:
        if self._tensorboard is not None:
            self._tensorboard.flush()
            self._tensorboard.close()
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except AttributeError:  # pragma: no cover - older wandb
                pass


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


def _should_validate(global_step: int, step_in_epoch: int, steps_per_epoch: int, interval) -> bool:
    if interval is None:
        return False
    if interval == "epoch":
        return step_in_epoch == (steps_per_epoch - 1)
    if isinstance(interval, int) and interval > 0:
        return global_step % interval == 0
    return False


@app.command()
def train(
    config_path: Path = typer.Argument(..., help="Path to diffusion training TOML"),
    output_dir: Optional[Path] = typer.Option(None, help="Override the configured output directory"),
) -> None:
    cfg = load_diffusion_config(config_path)
    run_dir = _resolve_output_dir(cfg, config_path, output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.training.output_dir = run_dir

    adapter = registry.get(cfg.model.name)
    tokenizer = adapter.create_tokenizer(cfg)

    project_config = ProjectConfiguration(project_dir=str(run_dir), automatic_checkpoint_naming=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        project_config=project_config,
    )
    set_seed(cfg.training.seed)

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
    if modules.text_encoder is not None and any(param.requires_grad for param in modules.text_encoder.parameters()):
        objects_to_prepare.insert(1, modules.text_encoder)
    if val_loader is not None:
        objects_to_prepare.append(val_loader)

    prepared = list(accelerator.prepare(*objects_to_prepare))

    idx = 0
    modules.denoiser = prepared[idx]
    idx += 1
    prepare_text_encoder = modules.text_encoder is not None and any(
        param.requires_grad for param in modules.text_encoder.parameters()
    )
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

    logger = DiffusionLogger(cfg.logging, run_dir) if accelerator.is_main_process else None
    if accelerator.is_main_process:
        with (run_dir / "config.json").open("w", encoding="utf-8") as fp:
            json.dump(asdict(cfg), fp, default=str, indent=2)

    global_step = 0
    for epoch in range(cfg.training.epochs):
        modules.denoiser.train()
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
                if logger is not None:
                    logger.log(global_step, log_payload)

                if cfg.training.checkpoint_interval and global_step % cfg.training.checkpoint_interval == 0:
                    save_dir = run_dir / f"checkpoint-{global_step}"
                    accelerator.save_state(str(save_dir))

                if val_loader is not None and _should_validate(global_step, step, len(train_loader), cfg.training.validation_interval):
                    modules.denoiser.eval()
                    val_losses: list[float] = []
                    with torch.no_grad():
                        for val_batch in val_loader:
                            val_metrics = adapter.validation_step(accelerator, cfg, modules, val_batch)
                            val_losses.append(val_metrics.get("loss", 0.0))
                    mean_val = sum(val_losses) / max(1, len(val_losses))
                    metrics_map = {"val/loss": mean_val}
                    accelerator.log(metrics_map, step=global_step)
                    if logger is not None:
                        logger.log(global_step, metrics_map)
                    modules.denoiser.train()

            if global_step >= max_train_steps:
                break
        if global_step >= max_train_steps:
            break

    if accelerator.is_main_process:
        final_dir = run_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        adapter.save_checkpoint(accelerator, cfg, modules, str(final_dir))

    if logger is not None:
        logger.close()


if __name__ == "__main__":  # pragma: no cover
    app()
