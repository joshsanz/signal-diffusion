"""Training loop for Signal Diffusion classifier experiments."""
from __future__ import annotations

import gc
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, cast

import tomllib
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import typer

from signal_diffusion.classification import build_classifier, build_dataset, build_task_specs, ClassifierConfig, TaskSpec
from signal_diffusion.config import load_settings
from signal_diffusion.log_setup import get_logger
from signal_diffusion.losses import FocalLoss
from signal_diffusion.classification.config import (
    DatasetConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from signal_diffusion.classification.metrics import (
    create_metrics_logger,
)
from signal_diffusion.training.schedulers import SchedulerType, create_scheduler

LOGGER = get_logger(__name__)


class EvaluationManager:
    """Coordinator for running validation according to a chosen strategy."""

    def __init__(
        self,
        strategy: str,
        eval_steps: int | None,
        evaluate_fn: Callable[[], dict[str, Any]],
    ) -> None:
        self.strategy = strategy
        self.eval_steps = eval_steps
        self.evaluate_fn = evaluate_fn
        self.latest_result: dict[str, Any] | None = None
        self._pending: list[dict[str, Any]] = []
        self._next_step = eval_steps if strategy == "steps" else None

    def on_step(self, global_step: int) -> bool:
        if self.strategy != "steps" or self.eval_steps is None:
            return False
        triggered = False
        while self._next_step is not None and global_step >= self._next_step:
            self._trigger(global_step)
            self._next_step += self.eval_steps
            triggered = True
        return triggered

    def on_epoch_end(self, global_step: int) -> bool:
        if self.strategy != "epoch":
            return False
        self._trigger(global_step)
        return True

    def drain_new_results(self) -> list[dict[str, Any]]:
        results = self._pending[:]
        self._pending.clear()
        return results

    def _trigger(self, global_step: int) -> None:
        result = dict(self.evaluate_fn())
        result["_global_step"] = global_step
        self.latest_result = result
        self._pending.append(result)


class CheckpointManager:
    """Coordinator for saving checkpoints according to a chosen strategy."""

    def __init__(
        self,
        strategy: str,
        checkpoint_steps: int | None,
    ) -> None:
        self.strategy = strategy
        self.checkpoint_steps = checkpoint_steps
        self._next_step = checkpoint_steps if strategy == "steps" else None

    def on_step(self, global_step: int) -> bool:
        if self.strategy != "steps" or self.checkpoint_steps is None:
            return False
        triggered = False
        while self._next_step is not None and global_step >= self._next_step:
            triggered = True
            self._next_step += self.checkpoint_steps
        return triggered

    def on_epoch_end(self, epoch: int) -> bool:
        if self.strategy != "epoch":
            return False
        return True


def _validate_backbone_data_type(data_type: str, backbone: str) -> None:
    """Ensure backbone choice matches configured data type."""
    is_timeseries = data_type == "timeseries"
    is_1d_backbone = "1d" in backbone.lower()

    if is_1d_backbone and not is_timeseries:
        raise ValueError(
            f"Backbone '{backbone}' requires time-domain data (settings.data_type='timeseries'). "
            f"Current data_type='{data_type}'."
        )
    if is_timeseries and not is_1d_backbone:
        LOGGER.warning(
            "Configured data_type='timeseries' but backbone '%s' is 2D-focused. "
            "Consider using 'cnn_1d' or 'cnn_1d_light' for time-domain inputs.",
            backbone,
        )


@dataclass(slots=True)
class ClassificationExperimentConfig:
    """Top-level experiment configuration."""

    path: Path
    settings_path: Path
    dataset: DatasetConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingConfig
    data_overrides: dict[str, Any] = field(default_factory=dict)  # Overrides from [data] section


@dataclass(slots=True)
class EpochMetrics:
    """Captured metrics for an epoch."""

    epoch: int
    train_loss: float
    train_losses: dict[str, float]
    val_loss: float | None
    val_losses: dict[str, float | None]
    train_accuracy: dict[str, float | None]
    val_accuracy: dict[str, float | None]
    train_mse: dict[str, float | None]
    val_mse: dict[str, float | None]
    lr: float
    train_mae: dict[str, float | None] = field(default_factory=dict)
    val_mae: dict[str, float | None] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingSummary:
    """Summary returned after training completes."""

    run_dir: Path
    best_checkpoint: Path
    history: list[EpochMetrics]
    best_metric: float | None
    best_epoch: int | None
    best_global_step: int | None
    top_checkpoints: list["CheckpointRecord"]
    swa_checkpoint: Path | None = None


@dataclass(slots=True)
class CheckpointRecord:
    """Metadata describing a saved checkpoint."""

    path: Path
    metric: float
    epoch: int
    global_step: int


def load_experiment_config(path: str | Path) -> ClassificationExperimentConfig:
    """Load a TOML configuration file describing a classification experiment."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    base_dir = config_path.parent
    settings_section = data.get("settings", {})
    settings_path = _resolve_path(settings_section.get("config", "config/default.toml"), base_dir)

    dataset_section = data.get("dataset")
    if not dataset_section:
        raise ValueError("Configuration missing [dataset] section")
    dataset = DatasetConfig(
        name=_require(dataset_section, "name"),
        tasks=tuple(dataset_section.get("tasks", ())),
        train_split=dataset_section.get("train_split", "train"),
        val_split=dataset_section.get("val_split", "val"),
        batch_size=int(dataset_section.get("batch_size", 32)),
        num_workers=int(dataset_section.get("num_workers", 4)),
        pin_memory=bool(dataset_section.get("pin_memory", True)),
        shuffle=bool(dataset_section.get("shuffle", True)),
    )
    if not dataset.tasks:
        raise ValueError("[dataset] section must define at least one task")

    model_section = data.get("model")
    if not model_section:
        raise ValueError("Configuration missing [model] section")
    model = ModelConfig(
        backbone=_require(model_section, "backbone"),
        input_channels=int(_require(model_section, "input_channels")),
        embedding_dim=int(model_section.get("embedding_dim", 256)),
        dropout=float(model_section.get("dropout", 0.3)),
        activation=str(model_section.get("activation", "gelu")),
        extras=dict(model_section.get("extras", {})),
    )

    optimizer_section = data.get("optimizer", {})
    optimizer = OptimizerConfig(
        name=str(optimizer_section.get("name", "adamw")).lower(),
        learning_rate=float(optimizer_section.get("learning_rate", 3e-4)),
        weight_decay=float(optimizer_section.get("weight_decay", 1e-4)),
        betas=tuple(optimizer_section.get("betas", [0.9, 0.999])),
    )

    scheduler_section = data.get("scheduler", {})
    scheduler = SchedulerConfig(
        name=str(scheduler_section.get("name", "constant")).lower(),
        warmup_steps=int(scheduler_section.get("warmup_steps", 0)),
        kwargs=dict(scheduler_section.get("kwargs", {})),
    )

    training_section = data.get("training", {})
    training = TrainingConfig(
        epochs=int(training_section.get("epochs", 25)),
        max_steps=int(training_section.get("max_steps", -1)),
        clip_grad_norm=_optional_float(training_section.get("clip_grad_norm", 1.0)),
        device=training_section.get("device"),
        log_every_batches=int(training_section.get("log_every_batches", training_section.get("log_every", 10))),
        eval_strategy=str(training_section.get("eval_strategy", "epoch")).lower(),
        eval_steps=_optional_int(training_section.get("eval_steps")),
        output_dir=_optional_path(training_section.get("output_dir"), base_dir),
        log_dir=_optional_path(training_section.get("log_dir"), base_dir),
        tensorboard=bool(training_section.get("tensorboard", False)),
        wandb_project=training_section.get("wandb_project"),
        wandb_entity=training_section.get("wandb_entity"),
        wandb_tags=tuple(training_section.get("wandb_tags", ())),
        run_name=training_section.get("run_name"),
        checkpoint_total_limit=_optional_int(training_section.get("checkpoint_total_limit")),
        checkpoint_strategy=str(training_section.get("checkpoint_strategy", "epoch")).lower(),
        checkpoint_steps=_optional_int(training_section.get("checkpoint_steps")),
        task_weights={str(k): float(v) for k, v in training_section.get("task_weights", {}).items()},
        use_amp=bool(training_section.get("use_amp", False)),
        metrics_summary_path=_optional_path(training_section.get("metrics_summary_path"), base_dir),
        max_best_checkpoints=int(training_section.get("max_best_checkpoints", 1)),
        early_stopping=bool(training_section.get("early_stopping", False)),
        early_stopping_patience=int(training_section.get("early_stopping_patience", 5)),
        compile_model=bool(training_section.get("compile_model", True)),
        compile_mode=str(training_section.get("compile_mode", "default")).lower(),
        swa_enabled=bool(training_section.get("swa_enabled", False)),
        swa_extra_ratio=float(training_section.get("swa_extra_ratio", 0.333)),
        swa_lr_frac=float(training_section.get("swa_lr_frac", 0.25)),
        label_smoothing=float(training_section.get("label_smoothing", 0.0)),
    )

    _validate_eval_config(training)
    _validate_checkpoint_config(training)
    if training.max_best_checkpoints < 1:
        raise ValueError("[training] max_best_checkpoints must be >= 1")

    # Extract optional [data] section overrides
    data_section = data.get("data", {})
    data_overrides = dict(data_section) if isinstance(data_section, Mapping) else {}

    return ClassificationExperimentConfig(
        path=config_path,
        settings_path=settings_path,
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training=training,
        data_overrides=data_overrides,
    )


def train_from_config(
    config: ClassificationExperimentConfig,
    trial: Any | None = None,
) -> TrainingSummary:
    """Run a classification experiment from a parsed configuration.

    Args:
        config: Experiment configuration
        trial: Optional Optuna trial for hyperparameter optimization.
               If provided, intermediate metrics will be reported and
               pruning will be checked after each validation.
    """

    # Set random seeds for reproducibility
    if config.training.seed is not None:
        import random
        import numpy as np

        seed = config.training.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Note: Setting deterministic mode can impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        LOGGER.info(f"Set random seed to {seed} for reproducibility")
    else:
        LOGGER.warning("=" * 80)
        LOGGER.warning("WARNING: No random seed specified!")
        LOGGER.warning("Training is non-deterministic and results will vary between runs.")
        LOGGER.warning("Set [training] seed = 42 in your config for reproducible results.")
        LOGGER.warning("=" * 80)

    settings = load_settings(config.settings_path)

    # Apply [data] section overrides from classification config
    if config.data_overrides:
        if "output_type" in config.data_overrides:
            settings.output_type = str(config.data_overrides["output_type"])
        if "data_type" in config.data_overrides:
            settings.data_type = str(config.data_overrides["data_type"])

    dataset_cfg = config.dataset
    training_cfg = config.training
    _validate_backbone_data_type(getattr(settings, "data_type", "spectrogram"), config.model.backbone)

    tasks = dataset_cfg.tasks
    task_specs = build_task_specs(dataset_cfg.name, tasks)
    task_lookup = {spec.name: spec for spec in task_specs}
    classifier_config = ClassifierConfig(
        backbone=config.model.backbone,
        input_channels=config.model.input_channels,
        tasks=task_specs,
        embedding_dim=config.model.embedding_dim,
        dropout=config.model.dropout,
        activation=config.model.activation,
        depth=config.model.depth,
        layer_repeats=config.model.layer_repeats,
        extras=config.model.extras,
    )
    model: nn.Module = build_classifier(classifier_config)

    if training_cfg.device:
        device = torch.device(training_cfg.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Enable TF32 for faster matmul on Ampere+ GPUs with minimal accuracy impact
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        LOGGER.info("TF32 enabled for CUDA matmul and cuDNN operations")

    model.to(device)

    # Compile model if requested
    if training_cfg.compile_model:
        LOGGER.info(f"Compiling model with torch.compile(mode='{training_cfg.compile_mode}')...")
        model = cast(nn.Module, torch.compile(model, mode=training_cfg.compile_mode))
        LOGGER.info("Model compilation successful")

    # Initialize extras dict from dataset config for time-series metadata
    extras = dict(dataset_cfg.extras) if hasattr(dataset_cfg, 'extras') and dataset_cfg.extras else {}

    # For time-series datasets, add resolution as expected_length for signal validation
    if "_timeseries" in dataset_cfg.name or "timeseries" in dataset_cfg.name.lower():
        if "expected_length" not in extras and hasattr(dataset_cfg, 'resolution'):
            extras["expected_length"] = dataset_cfg.resolution

    train_dataset = build_dataset(
        settings,
        dataset_name=dataset_cfg.name,
        split=dataset_cfg.train_split,
        tasks=tasks,
        target_format="dict",
        extras=extras,
    )

    # Propagate populated extras back to dataset_cfg for backbone initialization
    if extras:
        if not hasattr(dataset_cfg, 'extras'):
            dataset_cfg.extras = {}
        for key, value in extras.items():
            dataset_cfg.extras[key] = value

    val_dataset = build_dataset(
        settings,
        dataset_name=dataset_cfg.name,
        split=dataset_cfg.val_split,
        tasks=tasks,
        target_format="dict",
        extras=extras,
    )

    # Configure DataLoader settings for optimal training performance.
    # Note: When using HPO with multiple trials, file handle accumulation can occur
    # with persistent_workers and pin_memory across sequential DataLoader creation.
    # Current strategy: Use config defaults and rely on explicit cleanup (del + gc.collect)
    use_pin_memory = dataset_cfg.pin_memory and torch.cuda.is_available()
    use_persistent_workers = dataset_cfg.num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_cfg.batch_size,
        shuffle=dataset_cfg.shuffle,
        num_workers=dataset_cfg.num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
        prefetch_factor=2 if dataset_cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=dataset_cfg.batch_size,
        shuffle=False,
        num_workers=dataset_cfg.num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
        prefetch_factor=2 if dataset_cfg.num_workers > 0 else None,
    )

    run_dir = _prepare_run_dir(config)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    optimizer = _build_optimizer(model, config.optimizer)

    # Create learning rate scheduler (always configured, defaults to constant with 0 warmup)
    if training_cfg.swa_enabled:
        swa_epoch_count = max(1, int(training_cfg.epochs * training_cfg.swa_extra_ratio))
        total_epochs = training_cfg.epochs + swa_epoch_count
        num_training_steps = training_cfg.epochs * len(train_loader)
    else:
        total_epochs = training_cfg.epochs
        num_training_steps = training_cfg.epochs * len(train_loader)
    if training_cfg.max_steps > 0:
        num_training_steps = min(num_training_steps, training_cfg.max_steps)
    scheduler_type = cast(SchedulerType, config.scheduler.name)
    lr_scheduler = create_scheduler(
        optimizer,
        scheduler_type=scheduler_type,
        num_warmup_steps=config.scheduler.warmup_steps,
        num_training_steps=num_training_steps,
        **config.scheduler.kwargs,
    )

    # Initialize SWA components if enabled
    swa_model = None
    swa_scheduler = None
    swa_start_epoch = None
    swa_epoch_count = 0

    if training_cfg.swa_enabled:
        if training_cfg.swa_extra_ratio <= 0.0:
            raise ValueError(f"swa_extra_ratio must be > 0, got {training_cfg.swa_extra_ratio}")

        from torch.optim.swa_utils import AveragedModel, SWALR

        # Append SWA epochs after base training completes.
        swa_epoch_count = max(1, int(training_cfg.epochs * training_cfg.swa_extra_ratio))
        swa_start_epoch = training_cfg.epochs + 1

        # Create averaged model wrapper
        swa_model = AveragedModel(model)

        # Create SWA learning rate scheduler
        swa_lr = config.optimizer.learning_rate * training_cfg.swa_lr_frac
        # We step per batch, so anneal_epochs represents step count for ~90% of one epoch.
        anneal_epochs = max(1, int(0.9 * len(train_loader)))
        swa_scheduler = SWALR(
            optimizer,
            swa_lr=swa_lr,
            anneal_epochs=anneal_epochs,
            anneal_strategy='linear',
        )

        LOGGER.info(
            "SWA enabled: %s base epochs + %s SWA epochs = %s total. "
            "SWA LR = %.6f (%sx base LR) with linear anneal over %s steps.",
            training_cfg.epochs,
            swa_epoch_count,
            total_epochs,
            swa_lr,
            training_cfg.swa_lr_frac,
            anneal_epochs,
        )

    scaler = torch.amp.GradScaler(enabled=training_cfg.use_amp and device.type == "cuda")
    criteria: dict[str, nn.Module] = {}
    for name in tasks:
        spec = task_lookup[name]
        if spec.task_type == "classification":
            if name == "health" and training_cfg.use_focal_loss_health:
                criteria[name] = FocalLoss(
                    alpha=training_cfg.focal_alpha,
                    gamma=training_cfg.focal_gamma,
                    reduction="mean",
                )
                LOGGER.info(
                    "Using focal loss for health task (alpha=%.3f, gamma=%.3f)",
                    training_cfg.focal_alpha,
                    training_cfg.focal_gamma,
                )
            else:
                criteria[name] = nn.CrossEntropyLoss(label_smoothing=training_cfg.label_smoothing)
        else:
            criteria[name] = nn.HuberLoss(delta=2.0)
    task_weights = _resolve_task_weights(tasks, training_cfg.task_weights)

    def _get_eval_model() -> nn.Module:
        if in_swa_phase and swa_model is not None:
            return swa_model
        return model

    def run_validation(val_loader=val_loader) -> dict[str, Any]:
        result, _ = _run_epoch(
            _get_eval_model(),
            data_loader=val_loader,
            criteria=criteria,
            task_weights=task_weights,
            task_specs=task_lookup,
            device=device,
            optimizer=None,
            scaler=None,
            clip_grad=None,
            log_every=0,
            train=False,
            trial=trial,
        )
        return result

    eval_manager = _create_evaluation_manager(training_cfg, run_validation)
    metrics_logger = create_metrics_logger(training_cfg, tasks, run_dir)

    # Initialize early stopping (disabled if trial is not None)
    early_stopping_enabled = training_cfg.early_stopping and trial is None
    if training_cfg.early_stopping and trial is not None:
        LOGGER.warning("Early stopping disabled during HPO trials (using Optuna pruning instead)")
    early_stopping_patience = training_cfg.early_stopping_patience
    patience_counter = 0
    best_metric_for_patience = float("-inf")
    early_stopped_at_epoch: int | None = None

    # Track whether we're in SWA phase
    in_swa_phase = False

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        strategy=training_cfg.checkpoint_strategy,
        checkpoint_steps=training_cfg.checkpoint_steps,
    )

    # Override checkpoint settings if early stopping enabled (with warnings)
    if early_stopping_enabled:
        if training_cfg.checkpoint_strategy != training_cfg.eval_strategy:
            LOGGER.warning(
                "Overriding checkpoint_strategy from '%s' to '%s' to match eval_strategy for early stopping",
                training_cfg.checkpoint_strategy,
                training_cfg.eval_strategy,
            )
            checkpoint_manager.strategy = training_cfg.eval_strategy
            checkpoint_manager.checkpoint_steps = training_cfg.eval_steps
            checkpoint_manager._next_step = training_cfg.eval_steps if training_cfg.eval_strategy == "steps" else None
        if training_cfg.checkpoint_steps != training_cfg.eval_steps:
            LOGGER.warning(
                "Overriding checkpoint_steps from %s to %s to match eval_steps for early stopping",
                training_cfg.checkpoint_steps,
                training_cfg.eval_steps,
            )

    history: list[EpochMetrics] = []
    best_metric = float("-inf")
    best_checkpoint = checkpoints_dir / "best.pt"
    best_epoch: int | None = None
    best_metric_step: int | None = None
    best_records: list[CheckpointRecord] = []
    max_best_checkpoints = training_cfg.max_best_checkpoints
    global_step = 0

    progress_bar = tqdm(
        total=total_epochs,
        desc="Training",
        dynamic_ncols=True,
    )

    try:
        for epoch in range(1, total_epochs + 1):
            if training_cfg.max_steps > 0 and global_step >= training_cfg.max_steps:
                break

            # Check if we're entering SWA phase
            if training_cfg.swa_enabled and swa_start_epoch is not None and epoch == swa_start_epoch:
                LOGGER.info("Entering SWA phase at epoch %s/%s", epoch, total_epochs)
                in_swa_phase = True
                # Disable early stopping during SWA
                if early_stopping_enabled:
                    LOGGER.info("Early stopping disabled during SWA phase")

            progress_bar.set_description(f"Epoch {epoch}/{total_epochs}")

            # Train one epoch. If using Optuna HPO, the trial may be pruned during validation
            # (when _run_epoch calls _report_and_check_pruning). Ensure proper cleanup by
            # catching exceptions and explicitly releasing DataLoader resources.
            try:
                # Select appropriate scheduler based on SWA phase
                active_lr_scheduler = swa_scheduler if (in_swa_phase and swa_scheduler is not None) else lr_scheduler

                train_result, global_step = _run_epoch(
                    model,
                    data_loader=train_loader,
                    criteria=criteria,
                    task_weights=task_weights,
                    task_specs=task_lookup,
                    device=device,
                    optimizer=optimizer,
                    lr_scheduler=active_lr_scheduler,
                    scheduler_step_per_batch=True,
                    scaler=scaler,
                    clip_grad=training_cfg.clip_grad_norm,
                    log_every=training_cfg.log_every_batches,
                    train=True,
                    global_step=global_step,
                    eval_manager=eval_manager,
                    max_steps=training_cfg.max_steps,
                    trial=trial,
                )
                if global_step is None:
                    raise RuntimeError("Expected global_step during training.")
            except Exception as e:
                # Clean up DataLoader resources (especially important for HPO trials)
                # to prevent "too many open files" errors in subsequent trials
                del train_loader
                del val_loader
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise e

            if metrics_logger is not None:
                metrics_logger.log("train", global_step, train_result, epoch=epoch)

            # Process any validation outputs that occurred during epoch
            # (either via eval_strategy="steps" or end-of-epoch validation)
            eval_outputs = eval_manager.drain_new_results() if eval_manager else []
            for eval_output in eval_outputs:
                # Compute weighted mean validation accuracy across all tasks
                mean_eval_acc = _compute_weighted_mean_accuracy(
                    eval_output["accuracy"],
                    task_weights,
                )
                if mean_eval_acc > 0.0:
                    mean_eval_display = f"{mean_eval_acc:.4f}"
                else:
                    # Fallback: use negative loss if no valid accuracies
                    mean_eval_acc = -eval_output["loss"]
                    mean_eval_display = "n/a"
                step_info = eval_output.get("_global_step")
                step_for_log = int(step_info) if step_info is not None else global_step
                print(
                    f"[eval] step={step_info} val_loss={eval_output['loss']:.4f} "
                    f"val_acc_mean={mean_eval_display}"
                )
                if metrics_logger is not None:
                    metrics_logger.log("val", step_for_log, eval_output, epoch=epoch)

                # Manage best checkpoints: keep top-k checkpoints based on validation metric
                # This allows recovery of any high-performing checkpoint, not just the single best
                state_dict: dict[str, torch.Tensor] | None = None
                eval_model = _get_eval_model()
                if max_best_checkpoints > 0:
                    # Create checkpoint record for this validation result
                    record = CheckpointRecord(
                        path=checkpoints_dir / f"best-epoch{epoch:03d}-step{step_for_log:08d}.pt",
                        metric=float(mean_eval_acc),
                        epoch=epoch,
                        global_step=step_for_log,
                    )

                    # Insert record into sorted list (highest metric first)
                    insert_index: int | None = None
                    for idx, existing in enumerate(best_records):
                        if record.metric > existing.metric:
                            insert_index = idx
                            break
                    if insert_index is None:
                        best_records.append(record)
                    else:
                        best_records.insert(insert_index, record)

                    # Keep only top-k checkpoints; remove oldest/worst ones
                    if len(best_records) > max_best_checkpoints:
                        trimmed = best_records[max_best_checkpoints:]
                        del best_records[max_best_checkpoints:]
                    else:
                        trimmed = []

                    # Save checkpoint if it made the top-k list
                    if record in best_records:
                        if not record.path.exists():
                            state_dict = eval_model.state_dict()
                            torch.save(state_dict, record.path)
                    else:
                        record = None

                    # Delete checkpoints that are no longer in top-k
                    for trimmed_record in trimmed:
                        if trimmed_record.path.exists():
                            trimmed_record.path.unlink(missing_ok=True)
                else:
                    record = None

                # Always save overall best checkpoint (highest validation metric seen)
                if mean_eval_acc > best_metric:
                    best_metric = mean_eval_acc
                    best_epoch = epoch
                    best_metric_step = step_for_log
                    if state_dict is None:
                        # Extract state_dict, handling compiled models
                        if hasattr(eval_model, '_orig_mod'):
                            state_dict = eval_model._orig_mod.state_dict()
                        else:
                            state_dict = eval_model.state_dict()
                    torch.save(state_dict, best_checkpoint)

                # Check early stopping: update patience counter based on metric improvement
                # Note: Early stopping is disabled during SWA phase to ensure full averaging period
                if early_stopping_enabled and not in_swa_phase:
                    if mean_eval_acc > best_metric_for_patience:
                        # Metric improved: reset patience counter
                        best_metric_for_patience = mean_eval_acc
                        patience_counter = 0
                    else:
                        # Metric did not improve: increment patience counter
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            early_stopped_at_epoch = epoch
                            LOGGER.info(
                                "Early stopping triggered at epoch %d: "
                                "No improvement for %d validation checks. Best metric: %.4f",
                                epoch,
                                early_stopping_patience,
                                best_metric_for_patience,
                            )
                            break  # Exit eval_output loop to stop training

            # If early stopping was triggered, exit the epoch loop
            if early_stopped_at_epoch is not None:
                break

            if eval_outputs:
                latest_val = eval_outputs[-1]
            else:
                latest_val = eval_manager.latest_result if eval_manager else None
            train_losses = {name: float(train_result["losses"][name]) for name in tasks}
            train_accuracy: dict[str, float | None] = {}
            train_mse: dict[str, float | None] = {}
            train_mae: dict[str, float | None] = {}
            for name in tasks:
                value = train_result["accuracy"].get(name)
                train_accuracy[name] = float(value) if value is not None else None
                mse_value = train_result.get("mse", {}).get(name)
                train_mse[name] = float(mse_value) if mse_value is not None else None
                mae_value = train_result.get("mae", {}).get(name)
                train_mae[name] = float(mae_value) if mae_value is not None else None

            if latest_val is not None:
                val_loss: float | None = float(latest_val["loss"])
                val_losses: dict[str, float | None] = {
                    name: float(latest_val["losses"][name]) for name in tasks
                }
                val_accuracy: dict[str, float | None] = {}
                val_mse: dict[str, float | None] = {}
                val_mae: dict[str, float | None] = {}
                for name in tasks:
                    value = latest_val["accuracy"].get(name)
                    val_accuracy[name] = float(value) if value is not None else None
                    mse_value = latest_val.get("mse", {}).get(name)
                    val_mse[name] = float(mse_value) if mse_value is not None else None
                    mae_value = latest_val.get("mae", {}).get(name)
                    val_mae[name] = float(mae_value) if mae_value is not None else None
            else:
                val_loss = None
                val_losses = {name: None for name in tasks}
                val_accuracy = {name: None for name in tasks}
                val_mse = {name: None for name in tasks}
                val_mae = {name: None for name in tasks}

            lr = optimizer.param_groups[0].get("lr", config.optimizer.learning_rate)
            epoch_metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_result["loss"],
                train_losses=train_losses,
                val_loss=val_loss,
                val_losses=val_losses,
                train_accuracy=train_accuracy,
                val_accuracy=val_accuracy,
                train_mse=train_mse,
                val_mse=val_mse,
                lr=lr,
                train_mae=train_mae,
                val_mae=val_mae,
            )
            history.append(epoch_metrics)

            valid_acc = [acc for acc in val_accuracy.values() if acc is not None]
            mean_val_acc = (sum(valid_acc) / len(valid_acc)) if valid_acc else None

            for task_name in tasks:
                spec = task_lookup[task_name]
                train_value = train_accuracy[task_name]
                train_display = f"{train_value:.4f}" if train_value is not None else "n/a"
                val_value = val_accuracy[task_name]
                val_display = f"{val_value:.4f}" if val_value is not None else "n/a"

                if spec.task_type == "regression":
                    # For regression, show both MAE and MSE
                    metric_label = "mae"
                    print(
                        f"  - {task_name}: train_{metric_label}={train_display} val_{metric_label}={val_display}"
                    )
                    # Also show MSE if available
                    train_mse = train_result.get("mse", {}).get(task_name)
                    train_mse_display = f"{train_mse:.4f}" if train_mse is not None else "n/a"
                    val_mse_value = latest_val.get("mse", {}).get(task_name) if latest_val else None
                    val_mse_display = f"{val_mse_value:.4f}" if val_mse_value is not None else "n/a"
                    print(
                        f"         train_mse={train_mse_display} val_mse={val_mse_display}"
                    )
                else:
                    metric_label = "acc"
                    print(
                        f"  - {task_name}: train_{metric_label}={train_display} val_{metric_label}={val_display}"
                    )

            train_loss_line = ", ".join(f"{name}: {train_losses[name]:.4f}" for name in tasks)
            val_loss_entries = []
            for name in tasks:
                loss_value = val_losses[name]
                val_loss_entries.append(f"{name}: {loss_value:.4f}" if loss_value is not None else f"{name}: n/a")
            val_loss_line = ", ".join(val_loss_entries)
            print(f"    train_losses: {train_loss_line}")
            print(f"    val_losses: {val_loss_line}")

            postfix_payload: dict[str, str] = {
                "train_loss": f"{train_result['loss']:.4f}",
            }
            valid_acc = [acc for acc in train_accuracy.values() if acc is not None]
            if valid_acc:
                mean_train_acc = sum(valid_acc) / len(valid_acc)
                postfix_payload["train_acc"] = f"{mean_train_acc:.4f}"
            if val_loss is not None:
                postfix_payload["val_loss"] = f"{val_loss:.4f}"
            if mean_val_acc is not None:
                postfix_payload["val_acc"] = f"{mean_val_acc:.4f}"
            progress_bar.set_postfix(**postfix_payload)
            progress_bar.update(1)

            # Handle periodic checkpointing based on checkpoint_strategy
            should_save_checkpoint = False
            if checkpoint_manager.strategy == "epoch":
                should_save_checkpoint = checkpoint_manager.on_epoch_end(epoch)
            elif checkpoint_manager.strategy == "steps":
                should_save_checkpoint = checkpoint_manager.on_step(global_step)

            if should_save_checkpoint:
                # Extract state_dict, handling compiled models
                if hasattr(model, '_orig_mod'):
                    epoch_state_dict = model._orig_mod.state_dict()
                else:
                    epoch_state_dict = model.state_dict()
                torch.save(epoch_state_dict, checkpoints_dir / f"epoch-{epoch:03d}.pt")
                if training_cfg.checkpoint_total_limit is not None and training_cfg.checkpoint_total_limit > 0:
                    periodic_checkpoints = sorted(checkpoints_dir.glob("epoch-*.pt"))
                    while len(periodic_checkpoints) > training_cfg.checkpoint_total_limit:
                        oldest_checkpoint = periodic_checkpoints.pop(0)
                        oldest_checkpoint.unlink(missing_ok=True)

            # Update SWA model and scheduler (if in SWA phase)
            if in_swa_phase:
                # Update SWA averaged model weights
                if swa_model is not None:
                    swa_model.update_parameters(cast(nn.Module, model))
                    LOGGER.debug(f"Updated SWA model weights at epoch {epoch}")

    except KeyboardInterrupt:
        # User interrupted training - ensure progress bar is closed before re-raising
        LOGGER.info("Training interrupted by user")
        progress_bar.close()
        raise

    finally:
        # Ensure cleanup always happens (even if exception was raised)
        # This guarantees progress_bar.close() is called to prevent signal handler issues
        if not progress_bar.disable:  # Only close if not already closed
            try:
                progress_bar.close()
            except Exception:
                pass  # Ignore errors during progress bar cleanup

    # Update batch normalization statistics for SWA model
    if training_cfg.swa_enabled and swa_model is not None:
        LOGGER.info("Updating batch normalization statistics for SWA model...")
        from torch.optim.swa_utils import update_bn

        # Use the existing train_loader
        try:
            update_bn(train_loader, swa_model, device=device)
            LOGGER.info("Batch normalization statistics updated successfully")
        except Exception as e:
            LOGGER.warning(f"Failed to update BN statistics: {e}")

    # Save SWA checkpoint if enabled
    swa_checkpoint_path = None
    if training_cfg.swa_enabled and swa_model is not None:
        swa_checkpoint_path = checkpoints_dir / "swa.pt"
        # Extract state_dict, handling compiled models
        if hasattr(swa_model, '_orig_mod'):
            swa_state_dict = swa_model._orig_mod.state_dict()
        else:
            swa_state_dict = swa_model.state_dict()
        torch.save(swa_state_dict, swa_checkpoint_path)
        LOGGER.info(f"SWA model checkpoint saved to {swa_checkpoint_path}")

    # Save final checkpoint for recovery if needed
    last_checkpoint = checkpoints_dir / "last.pt"
    # Extract state_dict, handling compiled models
    if hasattr(model, '_orig_mod'):
        last_state_dict = model._orig_mod.state_dict()
    else:
        last_state_dict = model.state_dict()
    torch.save(last_state_dict, last_checkpoint)

    # Save epoch-level metrics history
    history_path = run_dir / "history.json"
    _save_history(history, history_path)

    # If no best checkpoint was found during training, use last checkpoint
    if best_metric == float("-inf") or not best_checkpoint.exists():
        best_checkpoint = last_checkpoint

    # Log checkpoints as MLflow artifacts
    if metrics_logger is not None:
        # Log best checkpoint
        if best_checkpoint.exists():
            metrics_logger.log_artifact(best_checkpoint, artifact_path="checkpoints")
        # Log SWA checkpoint if available
        if swa_checkpoint_path is not None and Path(swa_checkpoint_path).exists():
            metrics_logger.log_artifact(swa_checkpoint_path, artifact_path="checkpoints")

    # Close logging backends
    if metrics_logger is not None:
        metrics_logger.close()

    # Report early stopping status
    if early_stopping_enabled and early_stopped_at_epoch is not None:
        LOGGER.info("Training stopped early at epoch %d", early_stopped_at_epoch)
        LOGGER.info("Best validation metric achieved: %.4f", best_metric_for_patience)
        LOGGER.info("Best model checkpoint: %s", best_checkpoint)

    # Build final training summary with checkpoint paths and metrics
    resolved_best_metric = None if best_metric == float("-inf") else float(best_metric)
    summary = TrainingSummary(
        run_dir=run_dir,
        best_checkpoint=best_checkpoint,
        history=history,
        best_metric=resolved_best_metric,
        best_epoch=best_epoch,
        best_global_step=best_metric_step,
        top_checkpoints=list(best_records),
        swa_checkpoint=swa_checkpoint_path,
    )
    _write_summary(summary, run_dir / "summary.json")
    if training_cfg.metrics_summary_path is not None:
        _export_metrics_summary(summary, training_cfg.metrics_summary_path)

    # Explicitly release DataLoader resources to prevent file handle accumulation
    # (critical when running multiple trials in HPO with sequential DataLoaders)
    del train_loader
    del val_loader
    gc.collect()

    return summary


def _create_evaluation_manager(
    training_cfg: TrainingConfig,
    evaluate_fn: Callable[[], dict[str, Any]],
) -> EvaluationManager | None:
    strategy = training_cfg.eval_strategy
    if strategy == "none":
        return None
    if strategy not in {"steps", "epoch"}:
        raise ValueError(f"Unsupported eval_strategy '{strategy}'")
    return EvaluationManager(strategy, training_cfg.eval_steps, evaluate_fn)


def _validate_eval_config(config: TrainingConfig) -> None:
    allowed = {"none", "steps", "epoch"}
    if config.eval_strategy not in allowed:
        raise ValueError(
            f"eval_strategy must be one of {sorted(allowed)}, got '{config.eval_strategy}'"
        )
    if config.eval_strategy == "steps":
        if config.eval_steps is None or config.eval_steps <= 0:
            raise ValueError("eval_steps must be a positive integer when eval_strategy='steps'")
    if config.eval_strategy != "steps" and config.eval_steps is not None and config.eval_steps <= 0:
        raise ValueError("eval_steps must be positive when provided")


def _validate_checkpoint_config(config: TrainingConfig) -> None:
    allowed = {"steps", "epoch"}
    if config.checkpoint_strategy not in allowed:
        raise ValueError(
            f"checkpoint_strategy must be one of {sorted(allowed)}, got '{config.checkpoint_strategy}'"
        )
    if config.checkpoint_strategy == "steps":
        if config.checkpoint_steps is None or config.checkpoint_steps <= 0:
            raise ValueError("checkpoint_steps must be a positive integer when checkpoint_strategy='steps'")
    if config.checkpoint_strategy != "steps" and config.checkpoint_steps is not None and config.checkpoint_steps <= 0:
        raise ValueError("checkpoint_steps must be positive when provided")


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _compute_combined_objective(
    task_specs: Mapping[str, TaskSpec],
    val_accuracy: dict[str, float | None],
    val_mse: dict[str, float | None],
) -> float:
    """Compute combined objective score for mixed classification and regression tasks.

    For classification tasks: uses accuracy directly (higher is better)
    For regression tasks: uses 1/(1+mse) transformation (higher is better)
    Returns: unweighted mean of all task scores

    Args:
        task_specs: Mapping of task names to TaskSpec objects
        val_accuracy: Dict of validation accuracy/MAE values per task
        val_mse: Dict of validation MSE values per task

    Returns:
        Combined objective score in [0, 1] range
    """
    scores = []
    for task_name, spec in task_specs.items():
        if spec.task_type == "classification":
            # Classification: use accuracy directly
            accuracy = val_accuracy.get(task_name)
            if accuracy is not None:
                scores.append(float(accuracy))
        else:
            # Regression: normalize MSE to [0, 1] scale using 1/(1+mse)
            mse = val_mse.get(task_name)
            if mse is not None:
                # Ensure MSE is non-negative and compute normalized score
                mse_val = max(0.0, float(mse))
                normalized_score = 1.0 / (1.0 + mse_val)
                scores.append(normalized_score)

    if not scores:
        # No valid metrics available
        return 0.0

    return sum(scores) / len(scores)


def _report_and_check_pruning(
    trial: Any,
    mean_accuracy: float,
    global_step: int,
) -> None:
    """Report intermediate value to Optuna trial and check for pruning.

    Args:
        trial: Optuna trial object
        mean_accuracy: Mean validation accuracy to report
        global_step: Current training step

    Raises:
        optuna.TrialPruned: If the trial should be pruned
    """
    if trial is None:
        return

    try:
        import optuna
        # Report the mean accuracy as intermediate value
        trial.report(mean_accuracy, global_step)
        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()
    except ImportError:
        pass  # Optuna not available, skip reporting


def _run_epoch(
    model: nn.Module,
    *,
    data_loader: DataLoader,
    criteria: Mapping[str, nn.Module],
    task_weights: Mapping[str, float],
    task_specs: Mapping[str, TaskSpec],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scheduler_step_per_batch: bool = True,
    scaler: torch.amp.GradScaler | None = None,
    clip_grad: float | None,
    log_every: int,
    train: bool,
    global_step: int | None = None,
    eval_manager: EvaluationManager | None = None,
    max_steps: int = -1,
    trial: Any | None = None,
) -> tuple[dict[str, Any], int | None]:
    if global_step is None:
        global_step = 0
    if train:
        if optimizer is None:
            raise ValueError("Optimizer must be provided when train=True")
        optimizer = cast(torch.optim.Optimizer, optimizer)
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_examples = 0
    task_loss_sums = {name: 0.0 for name in task_weights}
    task_counts = {name: 0 for name in task_weights}
    classification_stats = {
        name: {"correct": 0, "total": 0}
        for name, spec in task_specs.items()
        if spec.task_type == "classification"
    }
    regression_stats = {
        name: {"abs_error_sum": 0.0, "squared_error_sum": 0.0, "total": 0}
        for name, spec in task_specs.items()
        if spec.task_type == "regression"
    }
    grad_norm_sum = 0.0
    grad_norm_count = 0

    batch_progress_bar = None
    if log_every > 0 or not train:  # Show progress for validation always
        phase = "train" if train else "val"
        batch_progress_bar = tqdm(
            total=len(data_loader),
            desc=f"  {phase}",
            dynamic_ncols=True,
            leave=False,
        )

    for batch_idx, batch in enumerate(data_loader, start=1):
        if max_steps > 0 and global_step >= max_steps:
            break
        if "signal" in batch:
            inputs = batch["signal"].to(device).contiguous()
        elif "image" in batch:
            inputs = batch["image"].to(device).contiguous()
        else:
            raise KeyError("Batch must contain 'signal' or 'image'")
        targets = {}
        for name in task_weights:
            tensor = _to_device_tensor(batch["targets"][name], device)
            spec = task_specs[name]
            if spec.task_type == "classification":
                tensor = tensor.long()
            else:
                tensor = tensor.float()
            targets[name] = tensor
        batch_size = inputs.shape[0]

        if train:
            optimizer = cast(torch.optim.Optimizer, optimizer)
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=device.type, enabled=scaler is not None and scaler.is_enabled()):
                outputs = model(inputs)
                loss, per_task_batch = _compute_loss(
                    outputs,
                    targets,
                    criteria,
                    task_weights,
                    task_specs,
                )

        if train:
            # Backward pass and optimizer step
            grad_norm = None
            if scaler is not None and scaler.is_enabled():
                # Mixed precision training with gradient scaling
                scaler.scale(loss).backward()
                if clip_grad is not None:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard full-precision backward pass
                loss.backward()
                if clip_grad is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()

            # Update learning rate according to scheduler (per-batch for both base and SWA).
            if lr_scheduler is not None and scheduler_step_per_batch:
                lr_scheduler.step()

            # Accumulate gradient norm for logging
            if grad_norm is not None:
                grad_norm_sum += grad_norm.item()
                grad_norm_count += 1

            # Check if we should perform intermediate validation (for eval_strategy="steps")
            global_step += 1
            triggered = False
            if eval_manager is not None:
                triggered = eval_manager.on_step(global_step)
            if triggered:
                # Validation was triggered - immediately check for Optuna pruning
                # This allows early stopping of poor-performing trials during training
                if eval_manager is not None and eval_manager.latest_result is not None:
                    # Compute combined objective (accuracy for classification, 1/(1+mse) for regression)
                    val_accuracy = eval_manager.latest_result.get("accuracy", {})
                    val_mse = eval_manager.latest_result.get("mse", {})
                    combined_objective = _compute_combined_objective(task_specs, val_accuracy, val_mse)

                    # Fallback if no valid metrics available
                    if combined_objective == 0.0:
                        combined_objective = -eval_manager.latest_result["loss"]

                    # Report to Optuna and check if trial should be pruned
                    _report_and_check_pruning(trial, combined_objective, global_step)
                model.train()

        # Accumulate batch metrics for epoch-level reporting
        batch_loss_value = float(loss.detach().item())
        total_loss += batch_loss_value * batch_size
        total_examples += batch_size
        for name in task_counts:
            task_counts[name] += batch_size

        # Compute per-task accuracy/error metrics
        for name, logits in outputs.items():
            spec = task_specs[name]
            if spec.task_type == "classification":
                # Classification: count correct predictions
                preds = logits.argmax(dim=1)
                correct = (preds == targets[name]).sum().item()
                stats = classification_stats[name]
                stats["correct"] += correct
                stats["total"] += batch_size
            elif spec.task_type == "regression":
                # Regression: compute absolute and squared errors for MAE/MSE
                error = logits.squeeze() - targets[name].squeeze()
                abs_error = torch.abs(error).sum().item()
                squared_error = (error ** 2).sum().item()
                stats = regression_stats[name]
                stats["abs_error_sum"] += abs_error
                stats["squared_error_sum"] += squared_error
                stats["total"] += batch_size

        # Accumulate per-task losses
        for name, task_loss_tensor in per_task_batch.items():
            loss_value = float(task_loss_tensor.detach().item())
            task_loss_sums[name] += loss_value * batch_size

        if batch_progress_bar is not None:
            mean_loss = total_loss / total_examples if total_examples else 0.0
            classification_values = [
                stats["correct"] / max(stats["total"], 1)
                for stats in classification_stats.values()
            ]
            mean_acc = (
                sum(classification_values) / len(classification_values)
                if classification_values
                else 0.0
            )
            batch_progress_bar.set_postfix(loss=f"{mean_loss:.4f}", acc=f"{mean_acc:.4f}")
            batch_progress_bar.update(1)

    # Check for end-of-epoch validation (for eval_strategy="epoch")
    if train and eval_manager is not None:
        triggered = eval_manager.on_epoch_end(global_step)
        if triggered:
            # Validation was triggered - immediately check for Optuna pruning
            # This allows early stopping of poor-performing trials after epoch completes
            if eval_manager.latest_result is not None:
                # Compute combined objective (accuracy for classification, 1/(1+mse) for regression)
                val_accuracy = eval_manager.latest_result.get("accuracy", {})
                val_mse = eval_manager.latest_result.get("mse", {})
                combined_objective = _compute_combined_objective(task_specs, val_accuracy, val_mse)

                # Fallback if no valid metrics available
                if combined_objective == 0.0:
                    combined_objective = -eval_manager.latest_result["loss"]

                # Report to Optuna and check if trial should be pruned
                _report_and_check_pruning(trial, combined_objective, global_step)
            model.train()

    # Compute final epoch-level metrics from accumulated statistics
    mean_loss = total_loss / max(total_examples, 1)
    accuracy: dict[str, float | None] = {}
    mae_metrics: dict[str, float | None] = {}
    mse_metrics: dict[str, float | None] = {}
    for name in task_weights:
        spec = task_specs[name]
        if spec.task_type == "classification":
            # Classification accuracy: correct predictions / total examples
            stats = classification_stats[name]
            accuracy[name] = stats["correct"] / max(stats["total"], 1)
        elif spec.task_type == "regression":
            # Regression metrics: compute both MAE and MSE
            stats = regression_stats[name]
            num_examples = max(stats["total"], 1)
            mae_metrics[name] = stats["abs_error_sum"] / num_examples
            mse_metrics[name] = stats["squared_error_sum"] / num_examples
            # Store MAE in accuracy dict for backward compatibility with reporting code
            accuracy[name] = mae_metrics[name]
        else:
            accuracy[name] = None

    # Compute per-task mean losses
    per_task_mean = {
        name: (task_loss_sums[name] / max(task_counts[name], 1)) if task_counts[name] else 0.0
        for name in task_loss_sums
    }

    # Build final metrics dict with all available metrics
    metrics = {"loss": mean_loss, "accuracy": accuracy, "losses": per_task_mean}
    if mae_metrics:
        metrics["mae"] = mae_metrics
    if mse_metrics:
        metrics["mse"] = mse_metrics
    if train and grad_norm_count > 0:
        # Average gradient norm for training stability monitoring
        metrics["grad_norm"] = grad_norm_sum / grad_norm_count

    if batch_progress_bar is not None:
        batch_progress_bar.close()

    return metrics, (global_step if train else None)


def _compute_loss(
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    criteria: Mapping[str, nn.Module],
    task_weights: Mapping[str, float],
    task_specs: Mapping[str, TaskSpec],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_tensor: torch.Tensor | None = None
    per_task_losses: dict[str, torch.Tensor] = {}
    for name, weight in task_weights.items():
        logits = outputs[name]
        target = targets[name]
        spec = task_specs[name]
        criterion = criteria[name]
        if spec.task_type == "regression":
            target = target.to(dtype=logits.dtype)
            if target.dim() == logits.dim() - 1:
                target = target.unsqueeze(-1)
        task_loss = criterion(logits, target)
        per_task_losses[name] = task_loss
        weighted = task_loss * weight
        if loss_tensor is None:
            loss_tensor = weighted
        else:
            loss_tensor = loss_tensor + weighted
    if loss_tensor is None:
        raise ValueError("At least one task required")
    return loss_tensor, per_task_losses


def _compute_weighted_mean_accuracy(
    accuracy_dict: Mapping[str, float | None],
    task_weights: Mapping[str, float],
) -> float:
    """Compute weighted mean accuracy across tasks, ignoring None values.

    Args:
        accuracy_dict: Task name → accuracy (or None for regression tasks with no accuracy)
        task_weights: Task name → weight for loss computation

    Returns:
        Weighted mean accuracy, or 0.0 if no valid accuracies available.
    """
    valid_accuracies = []
    valid_weights = []
    for task_name, accuracy in accuracy_dict.items():
        if accuracy is not None:
            valid_accuracies.append(accuracy)
            valid_weights.append(task_weights.get(task_name, 1.0))

    if not valid_accuracies:
        return 0.0

    # Normalize weights to sum to 1.0
    total_weight = sum(valid_weights)
    normalized_weights = [w / total_weight for w in valid_weights]

    # Compute weighted mean
    return sum(acc * w for acc, w in zip(valid_accuracies, normalized_weights))


def _build_optimizer(model: nn.Module, optimizer_cfg: OptimizerConfig) -> torch.optim.Optimizer:
    params = model.parameters()
    name = optimizer_cfg.name
    lr = optimizer_cfg.learning_rate
    wd = optimizer_cfg.weight_decay
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=optimizer_cfg.betas)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd, betas=optimizer_cfg.betas)
    if name == "sgd":
        momentum = 0.9
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=momentum)
    raise ValueError(f"Unsupported optimizer '{optimizer_cfg.name}'")




def _to_device_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.as_tensor(value)

    # Ensure MPS-compatible dtypes: float64 -> float32, int64 -> int32
    if tensor.dtype == torch.float64:
        tensor = tensor.to(torch.float32)
    elif tensor.dtype == torch.int64:
        tensor = tensor.to(torch.int32)

    return tensor.to(device).contiguous()


def _resolve_task_weights(tasks: Iterable[str], weights: Mapping[str, float]) -> dict[str, float]:
    tasks = tuple(tasks)
    if not tasks:
        raise ValueError("No tasks provided for weighting")
    if weights:
        resolved = {}
        for name in tasks:
            resolved[name] = float(weights.get(name, 1.0))
        return resolved
    uniform = 1.0 / len(tasks)
    return {name: uniform for name in tasks}


def _prepare_run_dir(config: ClassificationExperimentConfig) -> Path:
    base = config.training.output_dir or (config.path.parent / "runs")
    base = base if base.is_absolute() else base.resolve()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if config.training.run_name:
        run_name = f"{config.training.run_name}-{timestamp}"
    else:
        task_part = "-".join(config.dataset.tasks)
        # Use only the basename of dataset.name to avoid absolute path issues
        dataset_basename = Path(config.dataset.name).name
        run_name = f"{dataset_basename}-{config.model.backbone}-{task_part}-{timestamp}"
    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict = asdict(config.dataset)
    dataset_dict["tasks"] = list(dataset_dict.get("tasks", ()))
    model_dict = asdict(config.model)
    optimizer_dict = asdict(config.optimizer)
    scheduler_dict = asdict(config.scheduler)
    training_dict = asdict(config.training)
    for key in ("output_dir", "log_dir", "metrics_summary_path"):
        if training_dict.get(key) is not None:
            training_dict[key] = str(training_dict[key])
    if training_dict.get("wandb_tags") is not None:
        training_dict["wandb_tags"] = list(training_dict["wandb_tags"])

    config_copy = {
        "config_path": str(config.path),
        "settings_path": str(config.settings_path),
        "dataset": dataset_dict,
        "model": model_dict,
        "optimizer": optimizer_dict,
        "scheduler": scheduler_dict,
        "training": training_dict,
    }
    with (run_dir / "config_resolved.json").open("w", encoding="utf-8") as handle:
        json.dump(config_copy, handle, indent=2)
    return run_dir


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected float-compatible value, got {value!r}") from exc


def _optional_path(value: Any, base_dir: Path) -> Path | None:
    if value in (None, ""):
        return None
    return _resolve_path(value, base_dir)


def _resolve_path(value: Any, base_dir: Path) -> Path:
    if isinstance(value, Path):
        path = value
    else:
        path = Path(str(value))
    path = path.expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _require(section: Mapping[str, Any], key: str) -> Any:
    if key not in section:
        raise ValueError(f"Configuration section missing required key '{key}'")
    return section[key]


def _save_history(history: list[EpochMetrics], path: Path) -> None:
    payload = [
        {
            "epoch": item.epoch,
            "train_loss": item.train_loss,
            "train_losses": item.train_losses,
            "val_loss": item.val_loss,
            "val_losses": item.val_losses,
            "train_accuracy": item.train_accuracy,
            "val_accuracy": item.val_accuracy,
            "train_mse": item.train_mse,
            "val_mse": item.val_mse,
            "train_mae": item.train_mae,
            "val_mae": item.val_mae,
            "lr": item.lr,
        }
        for item in history
    ]
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_summary(summary: TrainingSummary, path: Path) -> None:
    payload = {
        "run_dir": str(summary.run_dir),
        "best_checkpoint": str(summary.best_checkpoint),
        "history_file": str(summary.run_dir / "history.json"),
        "best_metric": summary.best_metric,
        "best_epoch": summary.best_epoch,
        "best_global_step": summary.best_global_step,
        "top_checkpoints": [
            {
                "path": str(item.path),
                "metric": item.metric,
                "epoch": item.epoch,
                "global_step": item.global_step,
            }
            for item in summary.top_checkpoints
        ],
        "swa_checkpoint": str(summary.swa_checkpoint) if summary.swa_checkpoint else None,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _export_metrics_summary(summary: TrainingSummary, path: Path) -> None:
    payload = _build_metrics_summary(summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _build_metrics_summary(summary: TrainingSummary) -> dict[str, Any]:
    history = summary.history
    per_task_best_loss: dict[str, dict[str, float | int]] = {}
    per_task_best_accuracy: dict[str, dict[str, float | int]] = {}
    per_task_best_mae: dict[str, dict[str, float | int]] = {}
    per_task_best_mse: dict[str, dict[str, float | int]] = {}

    for epoch_metrics in history:
        for task_name, loss in epoch_metrics.val_losses.items():
            if loss is None:
                continue
            best_entry = per_task_best_loss.get(task_name)
            if best_entry is None or loss < best_entry["loss"]:
                per_task_best_loss[task_name] = {"loss": float(loss), "epoch": epoch_metrics.epoch}
        for task_name, accuracy in epoch_metrics.val_accuracy.items():
            if accuracy is None:
                continue
            best_entry = per_task_best_accuracy.get(task_name)
            if best_entry is None or accuracy > best_entry["accuracy"]:
                per_task_best_accuracy[task_name] = {
                    "accuracy": float(accuracy),
                    "epoch": epoch_metrics.epoch,
                }
        # Track best MAE and MSE for regression tasks (MAE should be minimized, as should MSE)
        # Note: accuracy values for regression tasks actually contain MAE
        for task_name, mae in epoch_metrics.val_accuracy.items():
            if mae is None:
                continue
            # Assuming task is regression if in val_accuracy (simplified logic)
            best_mae_entry = per_task_best_mae.get(task_name)
            if best_mae_entry is None or mae < best_mae_entry["mae"]:
                per_task_best_mae[task_name] = {"mae": float(mae), "epoch": epoch_metrics.epoch}
        for task_name, mse in epoch_metrics.val_mse.items():
            if mse is None:
                continue
            best_mse_entry = per_task_best_mse.get(task_name)
            if best_mse_entry is None or mse < best_mse_entry["mse"]:
                per_task_best_mse[task_name] = {"mse": float(mse), "epoch": epoch_metrics.epoch}

    payload: dict[str, Any] = {
        "run_dir": str(summary.run_dir),
        "best_checkpoint": str(summary.best_checkpoint),
        "history_file": str(summary.run_dir / "history.json"),
        "epochs_completed": len(history),
        "best_metric": summary.best_metric,
        "best_epoch": summary.best_epoch,
        "best_global_step": summary.best_global_step,
        "top_checkpoints": [
            {
                "path": str(item.path),
                "metric": item.metric,
                "epoch": item.epoch,
                "global_step": item.global_step,
            }
            for item in summary.top_checkpoints
        ],
        "per_task_best": {
            "accuracy": per_task_best_accuracy,
            "loss": per_task_best_loss,
            "mae": per_task_best_mae,
            "mse": per_task_best_mse,
        },
    }

    if history:
        final = history[-1]
        payload["final_epoch"] = {
            "epoch": final.epoch,
            "train_loss": float(final.train_loss),
            "val_loss": float(final.val_loss) if final.val_loss is not None else None,
            "train_losses": {name: float(value) for name, value in final.train_losses.items()},
            "val_losses": {
                name: (float(value) if value is not None else None)
                for name, value in final.val_losses.items()
            },
            "train_accuracy": {
                name: (float(value) if value is not None else None)
                for name, value in final.train_accuracy.items()
            },
            "val_accuracy": {
                name: (float(value) if value is not None else None)
                for name, value in final.val_accuracy.items()
            },
            "lr": float(final.lr),
        }
    return payload


app = typer.Typer(help="Classification training utilities for Signal Diffusion")


@app.command()
def train(
    config: Path,
    output_dir: Path | None = typer.Option(None, help="Override output directory"),
    metrics_summary_path: Path | None = typer.Option(
        None,
        "--metrics-summary-path",
        help="Optional path to write a summary of key metrics as JSON.",
    ),
    max_steps: int | None = typer.Option(
        None,
        "--max-steps",
        "--max_steps",
        help="Override training.max_steps from the config.",
    ),
) -> None:
    """Train a classifier experiment from a TOML configuration."""

    experiment = load_experiment_config(config)
    if output_dir is not None:
        experiment.training.output_dir = output_dir.resolve()
    if metrics_summary_path is not None:
        experiment.training.metrics_summary_path = metrics_summary_path.resolve()
    if max_steps is not None:
        experiment.training.max_steps = max_steps
    summary = train_from_config(experiment)
    print(f"Training complete. Best checkpoint saved at {summary.best_checkpoint}")


if __name__ == "__main__":  # pragma: no cover
    app()
