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

import numpy as np
import typer
from sklearn.metrics import f1_score

from signal_diffusion.classification import build_classifier, build_dataset, build_task_specs, ClassifierConfig, TaskSpec
from signal_diffusion.config import load_settings
from signal_diffusion.log_setup import get_logger
from signal_diffusion.losses import FocalLoss
from signal_diffusion.classification.config import (
    ClassificationConfig,
    load_classification_config,
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
    """Coordinator for saving checkpoints according to a chosen strategy.

    Enhanced to handle checkpoint saving, top-k management, and periodic cleanup.
    """

    def __init__(
        self,
        strategy: str,
        checkpoint_steps: int | None,
        checkpoints_dir: Path,
    ) -> None:
        self.strategy = strategy
        self.checkpoint_steps = checkpoint_steps
        self.checkpoints_dir = checkpoints_dir
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

    @staticmethod
    def extract_state_dict(model: nn.Module) -> dict:
        """Extract state_dict from model, handling torch.compile() models.

        Compiled models have state dict under _orig_mod attribute.
        """
        if hasattr(model, '_orig_mod'):
            return model._orig_mod.state_dict()
        return model.state_dict()

    def save_best_checkpoint(
        self,
        model: nn.Module,
        state: "TrainingState",
        metric: float,
        epoch: int,
        step: int,
    ) -> CheckpointRecord | None:
        """Save checkpoint if it makes top-k list.

        Manages sorted list of best checkpoints, keeping only top-k.

        Args:
            model: Model to save
            state: Training state with best_records list
            metric: Validation metric for this checkpoint
            epoch: Current epoch
            step: Current global step

        Returns:
            CheckpointRecord if saved, None otherwise
        """
        if state.max_best_checkpoints <= 0:
            return None

        record = CheckpointRecord(
            path=self.checkpoints_dir / f"best-epoch{epoch:03d}-step{step:08d}.pt",
            metric=float(metric),
            epoch=epoch,
            global_step=step,
        )

        # Insert into sorted list (highest metric first)
        insert_index: int | None = None
        for idx, existing in enumerate(state.best_records):
            if record.metric > existing.metric:
                insert_index = idx
                break
        if insert_index is None:
            state.best_records.append(record)
        else:
            state.best_records.insert(insert_index, record)

        # Keep only top-k checkpoints
        if len(state.best_records) > state.max_best_checkpoints:
            trimmed = state.best_records[state.max_best_checkpoints:]
            del state.best_records[state.max_best_checkpoints:]
        else:
            trimmed = []

        # Save if it made the list
        if record in state.best_records:
            if not record.path.exists():
                torch.save(self.extract_state_dict(model), record.path)
        else:
            record = None

        # Delete trimmed checkpoints
        for trimmed_record in trimmed:
            if trimmed_record.path.exists():
                trimmed_record.path.unlink(missing_ok=True)

        return record

    def update_best_overall(
        self,
        model: nn.Module,
        best_checkpoint_path: Path,
    ) -> None:
        """Save the overall best checkpoint.

        Args:
            model: Model to save
            best_checkpoint_path: Path to best.pt file
        """
        torch.save(self.extract_state_dict(model), best_checkpoint_path)

    def save_periodic_checkpoint(
        self,
        model: nn.Module,
        epoch: int,
        checkpoint_total_limit: int | None = None,
    ) -> None:
        """Save periodic checkpoint and enforce total limit.

        Args:
            model: Model to save
            epoch: Current epoch
            checkpoint_total_limit: Maximum number of periodic checkpoints to keep
        """
        checkpoint_path = self.checkpoints_dir / f"epoch-{epoch:03d}.pt"
        torch.save(self.extract_state_dict(model), checkpoint_path)

        if checkpoint_total_limit is not None and checkpoint_total_limit > 0:
            periodic_checkpoints = sorted(self.checkpoints_dir.glob("epoch-*.pt"))
            while len(periodic_checkpoints) > checkpoint_total_limit:
                oldest_checkpoint = periodic_checkpoints.pop(0)
                oldest_checkpoint.unlink(missing_ok=True)

    def save_last_checkpoint(self, model: nn.Module) -> Path:
        """Save the last checkpoint (most recent model state).

        Args:
            model: Model to save

        Returns:
            Path to saved checkpoint
        """
        last_checkpoint = self.checkpoints_dir / "last.pt"
        torch.save(self.extract_state_dict(model), last_checkpoint)
        return last_checkpoint

    def save_swa_checkpoint(self, swa_model: Any) -> Path:
        """Save SWA model checkpoint.

        Args:
            swa_model: AveragedModel from torch.optim.swa_utils

        Returns:
            Path to saved checkpoint
        """
        swa_checkpoint = self.checkpoints_dir / "swa.pt"
        torch.save(self.extract_state_dict(swa_model), swa_checkpoint)
        return swa_checkpoint


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
    train_f1: dict[str, float | None] = field(default_factory=dict)
    val_f1: dict[str, float | None] = field(default_factory=dict)


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


@dataclass(slots=True)
class TrainingContext:
    """Immutable training configuration and context.

    Encapsulates all configuration, paths, and derived settings that remain
    constant throughout training.
    """

    config: ClassificationConfig
    settings: Any  # Settings from load_settings()
    device: torch.device
    task_specs: Mapping[str, TaskSpec]
    task_weights: Mapping[str, float]
    run_dir: Path
    checkpoints_dir: Path
    trial: Any | None
    total_epochs: int
    num_training_steps: int

    @staticmethod
    def from_config(
        config: ClassificationConfig,
        trial: Any | None = None,
    ) -> "TrainingContext":
        """Create TrainingContext from configuration.

        Extracts and validates all configuration, sets up directories,
        and computes derived values like total_epochs and num_training_steps.

        Args:
            config: Experiment configuration
            trial: Optional Optuna trial for HPO

        Returns:
            Initialized TrainingContext
        """
        training_cfg = config.training

        # Validate configuration
        _validate_eval_config(training_cfg)
        _validate_checkpoint_config(training_cfg)
        if training_cfg.max_best_checkpoints < 1:
            raise ValueError("[training] max_best_checkpoints must be >= 1")

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
            LOGGER.info(f"Set random seed to {seed} for reproducibility")
        else:
            LOGGER.warning("=" * 80)
            LOGGER.warning("WARNING: No random seed specified!")
            LOGGER.warning("Training is non-deterministic and results will vary between runs.")
            LOGGER.warning("Set [training] seed = 42 in your config for reproducible results.")
            LOGGER.warning("=" * 80)

        # Load settings and apply overrides
        settings = load_settings(config.settings_path)
        if config.data_overrides:
            if "output_type" in config.data_overrides:
                settings.output_type = str(config.data_overrides["output_type"])
            if "data_type" in config.data_overrides:
                settings.data_type = str(config.data_overrides["data_type"])

        # Validate backbone compatibility
        dataset_cfg = config.dataset
        _validate_backbone_data_type(
            getattr(settings, "data_type", "spectrogram"),
            config.model.backbone
        )

        # Build task specifications
        tasks = dataset_cfg.tasks
        task_specs = build_task_specs(dataset_cfg.name, tasks)
        task_lookup = {spec.name: spec for spec in task_specs}
        task_weights = _resolve_task_weights(tasks, training_cfg.task_weights)

        # Determine device
        if training_cfg.device:
            device = torch.device(training_cfg.device)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        # Enable TF32 for faster matmul on Ampere+ GPUs
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            LOGGER.info("TF32 enabled for CUDA matmul and cuDNN operations")

        # Prepare output directories
        run_dir = _prepare_run_dir(config)
        checkpoints_dir = run_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Compute total epochs and training steps
        if training_cfg.swa_enabled:
            swa_epoch_count = max(1, int(training_cfg.epochs * training_cfg.swa_extra_ratio))
            total_epochs = training_cfg.epochs + swa_epoch_count
            num_training_steps = training_cfg.epochs * 1  # Placeholder, needs train_loader
        else:
            total_epochs = training_cfg.epochs
            num_training_steps = training_cfg.epochs * 1  # Placeholder, needs train_loader

        return TrainingContext(
            config=config,
            settings=settings,
            device=device,
            task_specs=task_lookup,
            task_weights=task_weights,
            run_dir=run_dir,
            checkpoints_dir=checkpoints_dir,
            trial=trial,
            total_epochs=total_epochs,
            num_training_steps=num_training_steps,  # Will be updated after DataLoader creation
        )


class TrainingState:
    """Mutable training state.

    Tracks all state that changes during training: steps, metrics,
    early stopping counters, etc.
    """

    def __init__(self, max_best_checkpoints: int) -> None:
        self.global_step: int = 0
        self.best_metric: float = float("-inf")
        self.best_epoch: int | None = None
        self.best_metric_step: int | None = None
        self.patience_counter: int = 0
        self.best_metric_for_patience: float = float("-inf")
        self.in_swa_phase: bool = False
        self.early_stopped_at_epoch: int | None = None
        self.history: list[EpochMetrics] = []
        self.best_records: list[CheckpointRecord] = []
        self.max_best_checkpoints: int = max_best_checkpoints

    def should_stop_early(self, patience: int) -> bool:
        """Check if early stopping criteria met."""
        return self.patience_counter >= patience

    def update_patience(self, current_metric: float) -> None:
        """Update patience counter based on metric improvement."""
        if current_metric > self.best_metric_for_patience:
            self.best_metric_for_patience = current_metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1


@dataclass(slots=True)
class TrainingResources:
    """PyTorch training objects.

    Groups all PyTorch modules and optimizers used during training.
    """

    model: nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler
    scaler: torch.amp.GradScaler
    criteria: dict[str, nn.Module]
    swa_model: Any | None = None  # AveragedModel from torch.optim.swa_utils
    swa_scheduler: Any | None = None  # SWALR from torch.optim.swa_utils
    swa_start_epoch: int | None = None

    def get_eval_model(self) -> nn.Module:
        """Return the model to use for evaluation (SWA model if in SWA phase)."""
        # Note: in_swa_phase is tracked in TrainingState, not here
        # Caller must check state.in_swa_phase and pass appropriate flag
        return self.swa_model if self.swa_model is not None else self.model

    def active_scheduler(self, in_swa_phase: bool) -> torch.optim.lr_scheduler.LRScheduler:
        """Return the scheduler to use based on training phase."""
        if in_swa_phase and self.swa_scheduler is not None:
            return self.swa_scheduler
        return self.lr_scheduler


class EarlyStoppingCoordinator:
    """Manages early stopping logic based on validation metric improvements.

    Tracks patience counter and determines when to stop training.
    """

    def __init__(self, enabled: bool, patience: int) -> None:
        self.enabled = enabled
        self.patience = patience

    def should_stop(self, state: TrainingState, in_swa_phase: bool) -> bool:
        """Check if early stopping should trigger.

        Args:
            state: Training state with patience info
            in_swa_phase: Whether in SWA phase (early stopping disabled during SWA)

        Returns:
            True if training should stop
        """
        if not self.enabled or in_swa_phase:
            return False
        return state.should_stop_early(self.patience)

    def update(self, state: TrainingState, current_metric: float) -> None:
        """Update patience counter based on metric improvement."""
        if self.enabled:
            state.update_patience(current_metric)


class CleanupManager:
    """Context manager ensuring proper resource cleanup on all exit paths.

    Handles cleanup of dataloaders, progress bars, metrics loggers, and CUDA cache.
    Ensures cleanup happens even on exceptions or KeyboardInterrupt.
    """

    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        progress_bar: tqdm,
        metrics_logger: Any | None,
    ) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.progress_bar = progress_bar
        self.metrics_logger = metrics_logger

    def __enter__(self) -> "CleanupManager":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Clean up resources in correct order."""
        # Close metrics logger first
        if self.metrics_logger is not None:
            try:
                self.metrics_logger.close()
            except Exception:
                pass  # Ignore cleanup errors

        # Close progress bar
        if not self.progress_bar.disable:
            try:
                self.progress_bar.close()
            except Exception:
                pass

        # Release DataLoader resources (critical for HPO with multiple trials)
        try:
            del self.train_loader
            del self.val_loader
        except Exception:
            pass

        # Force garbage collection and clear CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# Helper Functions for train_from_config
# ============================================================================


def _build_training_resources(
    model: nn.Module,
    context: TrainingContext,
    config: ClassificationConfig,
    train_loader: DataLoader,
    tasks: list[str],
    task_lookup: Mapping[str, TaskSpec],
) -> TrainingResources:
    """Build all training resources: optimizer, scheduler, criteria, SWA.

    Args:
        model: The classifier model
        context: Training context with configuration
        config: Full classification config
        train_loader: Training dataloader (needed for SWA scheduler steps)
        tasks: List of task names
        task_lookup: Mapping of task names to TaskSpec

    Returns:
        TrainingResources with all components initialized
    """
    training_cfg = config.training

    # Build optimizer
    optimizer = _build_optimizer(model, config.optimizer)

    # Build learning rate scheduler
    scheduler_type = cast(SchedulerType, config.scheduler.name)
    lr_scheduler = create_scheduler(
        optimizer,
        scheduler_type=scheduler_type,
        num_warmup_steps=config.scheduler.warmup_steps,
        num_training_steps=context.num_training_steps,
        **config.scheduler.kwargs,
    )

    # Build gradient scaler for mixed precision
    scaler = torch.amp.GradScaler(
        enabled=training_cfg.use_amp and context.device.type == "cuda"
    )

    # Build loss criteria for each task
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
                criteria[name] = nn.CrossEntropyLoss(
                    label_smoothing=training_cfg.label_smoothing
                )
        else:
            criteria[name] = nn.HuberLoss(delta=2.0)

    # Initialize SWA components if enabled
    swa_model = None
    swa_scheduler = None
    swa_start_epoch = None

    if training_cfg.swa_enabled:
        if training_cfg.swa_extra_ratio <= 0.0:
            raise ValueError(
                f"swa_extra_ratio must be > 0, got {training_cfg.swa_extra_ratio}"
            )

        from torch.optim.swa_utils import AveragedModel, SWALR

        swa_epoch_count = max(1, int(training_cfg.epochs * training_cfg.swa_extra_ratio))
        swa_start_epoch = training_cfg.epochs + 1
        swa_model = AveragedModel(model)

        swa_lr = config.optimizer.learning_rate * training_cfg.swa_lr_frac
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
            context.total_epochs,
            swa_lr,
            training_cfg.swa_lr_frac,
            anneal_epochs,
        )

    return TrainingResources(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        criteria=criteria,
        swa_model=swa_model,
        swa_scheduler=swa_scheduler,
        swa_start_epoch=swa_start_epoch,
    )


def _finalize_swa(
    swa_model: Any,
    train_loader: DataLoader,
    device: torch.device,
    checkpoint_manager: CheckpointManager,
) -> Path:
    """Finalize SWA model: update batch norm statistics and save checkpoint.

    Args:
        swa_model: AveragedModel from torch.optim.swa_utils
        train_loader: Training dataloader for BN statistics
        device: Device to use
        checkpoint_manager: Manager for saving checkpoint

    Returns:
        Path to saved SWA checkpoint
    """
    LOGGER.info("Updating batch normalization statistics for SWA model...")
    from torch.optim.swa_utils import update_bn

    try:
        update_bn(train_loader, swa_model, device=device)
        LOGGER.info("Batch normalization statistics updated successfully")
    except Exception as e:
        LOGGER.warning(f"Failed to update BN statistics: {e}")

    swa_checkpoint_path = checkpoint_manager.save_swa_checkpoint(swa_model)
    LOGGER.info(f"SWA model checkpoint saved to {swa_checkpoint_path}")
    return swa_checkpoint_path


def _print_epoch_metrics(
    tasks: list[str],
    task_lookup: Mapping[str, TaskSpec],
    train_result: dict,
    latest_val: dict | None,
    train_accuracy: dict[str, float | None],
    val_accuracy: dict[str, float | None],
    train_losses: dict[str, float],
    val_losses: dict[str, float | None],
) -> None:
    """Print per-task metrics for the epoch.

    Args:
        tasks: List of task names
        task_lookup: Mapping of task names to TaskSpec
        train_result: Training results dict
        latest_val: Latest validation results (or None)
        train_accuracy: Training accuracy per task
        val_accuracy: Validation accuracy per task
        train_losses: Training losses per task
        val_losses: Validation losses per task
    """
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
            print(f"         train_mse={train_mse_display} val_mse={val_mse_display}")
        else:
            metric_label = "acc"
            print(
                f"  - {task_name}: train_{metric_label}={train_display} val_{metric_label}={val_display}"
            )

    train_loss_line = ", ".join(f"{name}: {train_losses[name]:.4f}" for name in tasks)
    val_loss_entries = []
    for name in tasks:
        loss_value = val_losses[name]
        val_loss_entries.append(
            f"{name}: {loss_value:.4f}" if loss_value is not None else f"{name}: n/a"
        )
    val_loss_line = ", ".join(val_loss_entries)
    print(f"    train_losses: {train_loss_line}")
    print(f"    val_losses: {val_loss_line}")


def _update_progress_display(
    progress_bar: tqdm,
    train_result: dict,
    train_accuracy: dict[str, float | None],
    val_loss: float | None,
    val_accuracy: dict[str, float | None],
) -> None:
    """Update progress bar display with current metrics.

    Args:
        progress_bar: tqdm progress bar
        train_result: Training results dict
        train_accuracy: Training accuracy per task
        val_loss: Validation loss (or None)
        val_accuracy: Validation accuracy per task
    """
    postfix_payload: dict[str, str] = {
        "train_loss": f"{train_result['loss']:.4f}",
    }
    valid_acc = [acc for acc in train_accuracy.values() if acc is not None]
    if valid_acc:
        mean_train_acc = sum(valid_acc) / len(valid_acc)
        postfix_payload["train_acc"] = f"{mean_train_acc:.4f}"
    if val_loss is not None:
        postfix_payload["val_loss"] = f"{val_loss:.4f}"

    valid_val_acc = [acc for acc in val_accuracy.values() if acc is not None]
    if valid_val_acc:
        mean_val_acc = sum(valid_val_acc) / len(valid_val_acc)
        postfix_payload["val_acc"] = f"{mean_val_acc:.4f}"

    progress_bar.set_postfix(**postfix_payload)
    progress_bar.update(1)


def _build_epoch_metrics(
    epoch: int,
    tasks: list[str],
    train_result: dict,
    eval_outputs: list[dict],
    eval_manager: EvaluationManager | None,
    resources: TrainingResources,
    config: ClassificationConfig,
) -> tuple[EpochMetrics, dict, dict[str, float | None], float | None, dict[str, float | None]]:
    """Build EpochMetrics from training and validation results.

    Args:
        epoch: Current epoch number
        tasks: List of task names
        train_result: Training results dict
        eval_outputs: List of validation outputs
        eval_manager: Evaluation manager
        resources: Training resources
        config: Classification config

    Returns:
        Tuple of (epoch_metrics, latest_val, train_accuracy, val_loss, val_accuracy)
    """
    # Get latest validation result
    if eval_outputs:
        latest_val = eval_outputs[-1]
    else:
        latest_val = eval_manager.latest_result if eval_manager else None

    # Extract training metrics
    train_losses = {name: float(train_result["losses"][name]) for name in tasks}
    train_accuracy: dict[str, float | None] = {}
    train_f1: dict[str, float | None] = {}
    train_mse: dict[str, float | None] = {}
    train_mae: dict[str, float | None] = {}
    for name in tasks:
        value = train_result["accuracy"].get(name)
        train_accuracy[name] = float(value) if value is not None else None
        f1_value = train_result.get("f1", {}).get(name)
        train_f1[name] = float(f1_value) if f1_value is not None else None
        mse_value = train_result.get("mse", {}).get(name)
        train_mse[name] = float(mse_value) if mse_value is not None else None
        mae_value = train_result.get("mae", {}).get(name)
        train_mae[name] = float(mae_value) if mae_value is not None else None

    # Extract validation metrics
    if latest_val is not None:
        val_loss: float | None = float(latest_val["loss"])
        val_losses: dict[str, float | None] = {
            name: float(latest_val["losses"][name]) for name in tasks
        }
        val_accuracy: dict[str, float | None] = {}
        val_f1: dict[str, float | None] = {}
        val_mse: dict[str, float | None] = {}
        val_mae: dict[str, float | None] = {}
        for name in tasks:
            value = latest_val["accuracy"].get(name)
            val_accuracy[name] = float(value) if value is not None else None
            f1_value = latest_val.get("f1", {}).get(name)
            val_f1[name] = float(f1_value) if f1_value is not None else None
            mse_value = latest_val.get("mse", {}).get(name)
            val_mse[name] = float(mse_value) if mse_value is not None else None
            mae_value = latest_val.get("mae", {}).get(name)
            val_mae[name] = float(mae_value) if mae_value is not None else None
    else:
        val_loss = None
        val_losses = {name: None for name in tasks}
        val_accuracy = {name: None for name in tasks}
        val_f1 = {name: None for name in tasks}
        val_mse = {name: None for name in tasks}
        val_mae = {name: None for name in tasks}

    # Get current learning rate
    lr = resources.optimizer.param_groups[0].get("lr", config.optimizer.learning_rate)

    # Build epoch metrics
    epoch_metrics = EpochMetrics(
        epoch=epoch,
        train_loss=train_result["loss"],
        train_losses=train_losses,
        val_loss=val_loss,
        val_losses=val_losses,
        train_accuracy=train_accuracy,
        val_accuracy=val_accuracy,
        train_f1=train_f1,
        val_f1=val_f1,
        train_mse=train_mse,
        val_mse=val_mse,
        lr=lr,
        train_mae=train_mae,
        val_mae=val_mae,
    )

    return epoch_metrics, latest_val, train_accuracy, val_loss, val_accuracy


def train_from_config(
    config: ClassificationConfig,
    trial: Any | None = None,
) -> TrainingSummary:
    """Run a classification experiment from a parsed configuration.

    Args:
        config: Experiment configuration
        trial: Optional Optuna trial for hyperparameter optimization.
               If provided, intermediate metrics will be reported and
               pruning will be checked after each validation.
    """
    # Phase 1: Initialize context (immutable configuration)
    context = TrainingContext.from_config(config, trial)
    training_cfg = config.training
    settings = context.settings
    device = context.device
    task_lookup = context.task_specs
    tasks = config.dataset.tasks
    dataset_cfg = config.dataset

    # Phase 2: Build model
    task_specs_list = list(task_lookup.values())
    classifier_config = ClassifierConfig(
        backbone=config.model.backbone,
        input_channels=config.model.input_channels,
        tasks=task_specs_list,
        embedding_dim=config.model.embedding_dim,
        dropout=config.model.dropout,
        activation=config.model.activation,
        depth=config.model.depth,
        layer_repeats=config.model.layer_repeats,
        extras=config.model.extras,
    )
    model: nn.Module = build_classifier(classifier_config)
    model.to(device)

    # Compile model if requested
    if training_cfg.compile_model:
        LOGGER.info(f"Compiling model with torch.compile(mode='{training_cfg.compile_mode}')...")
        model = cast(nn.Module, torch.compile(model, mode=training_cfg.compile_mode))
        LOGGER.info("Model compilation successful")

    # Phase 3: Build datasets and dataloaders
    extras = dict(dataset_cfg.extras) if hasattr(dataset_cfg, 'extras') and dataset_cfg.extras else {}
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

    # Update context with actual num_training_steps now that we have train_loader
    if training_cfg.swa_enabled:
        context.num_training_steps = training_cfg.epochs * len(train_loader)
    else:
        context.num_training_steps = training_cfg.epochs * len(train_loader)
    if training_cfg.max_steps > 0:
        context.num_training_steps = min(context.num_training_steps, training_cfg.max_steps)

    # Phase 4: Build training resources (optimizer, scheduler, criteria, SWA)
    resources = _build_training_resources(
        model, context, config, train_loader, tasks, task_lookup
    )

    # Phase 5: Initialize training state
    state = TrainingState(max_best_checkpoints=training_cfg.max_best_checkpoints)

    # Define validation function (closure over context, resources, state)
    def _get_eval_model() -> nn.Module:
        if state.in_swa_phase and resources.swa_model is not None:
            return resources.swa_model
        return resources.model

    def run_validation(val_loader=val_loader) -> dict[str, Any]:
        result, _ = _run_epoch(
            _get_eval_model(),
            data_loader=val_loader,
            criteria=resources.criteria,
            task_weights=context.task_weights,
            task_specs=context.task_specs,
            device=context.device,
            optimizer=None,
            scaler=None,
            clip_grad=None,
            log_every=0,
            train=False,
            trial=context.trial,
        )
        return result

    eval_manager = _create_evaluation_manager(training_cfg, run_validation)
    metrics_logger = create_metrics_logger(training_cfg, settings, tasks, context.run_dir)

    # Initialize early stopping coordinator (disabled if trial is not None)
    early_stopping_enabled = training_cfg.early_stopping and trial is None
    if training_cfg.early_stopping and trial is not None:
        LOGGER.warning("Early stopping disabled during HPO trials (using Optuna pruning instead)")
    early_stopping = EarlyStoppingCoordinator(
        enabled=early_stopping_enabled,
        patience=training_cfg.early_stopping_patience,
    )

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        strategy=training_cfg.checkpoint_strategy,
        checkpoint_steps=training_cfg.checkpoint_steps,
        checkpoints_dir=context.checkpoints_dir,
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

    best_checkpoint = context.checkpoints_dir / "best.pt"

    progress_bar = tqdm(
        total=context.total_epochs,
        desc="Training",
        dynamic_ncols=True,
    )

    try:
        for epoch in range(1, context.total_epochs + 1):
            if training_cfg.max_steps > 0 and state.global_step >= training_cfg.max_steps:
                break

            # Check if we're entering SWA phase
            if training_cfg.swa_enabled and resources.swa_start_epoch is not None and epoch == resources.swa_start_epoch:
                LOGGER.info("Entering SWA phase at epoch %s/%s", epoch, context.total_epochs)
                state.in_swa_phase = True
                # Disable early stopping during SWA
                if early_stopping_enabled:
                    LOGGER.info("Early stopping disabled during SWA phase")

            progress_bar.set_description(f"Epoch {epoch}/{context.total_epochs}")

            # Train one epoch. If using Optuna HPO, the trial may be pruned during validation
            # (when _run_epoch calls _report_and_check_pruning). Ensure proper cleanup by
            # catching exceptions and explicitly releasing DataLoader resources.
            try:
                active_lr_scheduler = resources.active_scheduler(state.in_swa_phase)

                train_result, state.global_step = _run_epoch(
                    resources.model,
                    data_loader=train_loader,
                    criteria=resources.criteria,
                    task_weights=context.task_weights,
                    task_specs=task_lookup,
                    device=device,
                    optimizer=resources.optimizer,
                    lr_scheduler=active_lr_scheduler,
                    scheduler_step_per_batch=True,
                    scaler=resources.scaler,
                    clip_grad=training_cfg.clip_grad_norm,
                    log_every=training_cfg.log_every_batches,
                    train=True,
                    global_step=state.global_step,
                    eval_manager=eval_manager,
                    max_steps=training_cfg.max_steps,
                    trial=trial,
                    metrics_logger=metrics_logger,
                )
                if state.global_step is None:
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
                metrics_logger.log("train", state.global_step, train_result, epoch=epoch)

            # Process any validation outputs that occurred during epoch
            # (either via eval_strategy="steps" or end-of-epoch validation)
            eval_outputs = eval_manager.drain_new_results() if eval_manager else []
            for eval_output in eval_outputs:
                # Compute weighted mean validation accuracy across all tasks
                mean_eval_acc = _compute_weighted_mean_accuracy(
                    eval_output["accuracy"],
                    context.task_weights,
                )
                if mean_eval_acc > 0.0:
                    mean_eval_display = f"{mean_eval_acc:.4f}"
                else:
                    # Fallback: use negative loss if no valid accuracies
                    mean_eval_acc = -eval_output["loss"]
                    mean_eval_display = "n/a"
                step_info = eval_output.get("_global_step")
                step_for_log = int(step_info) if step_info is not None else state.global_step
                print(
                    f"[eval] step={step_info} val_loss={eval_output['loss']:.4f} "
                    f"val_acc_mean={mean_eval_display}"
                )
                if metrics_logger is not None:
                    metrics_logger.log("val", step_for_log, eval_output, epoch=epoch)

                # Manage best checkpoints: keep top-k checkpoints based on validation metric
                eval_model = _get_eval_model()

                # Save to top-k list if checkpoint qualifies
                checkpoint_manager.save_best_checkpoint(
                    eval_model, state, mean_eval_acc, epoch, step_for_log
                )

                # Always save overall best checkpoint (highest validation metric seen)
                if mean_eval_acc > state.best_metric:
                    state.best_metric = mean_eval_acc
                    state.best_epoch = epoch
                    state.best_metric_step = step_for_log
                    checkpoint_manager.update_best_overall(eval_model, best_checkpoint)

                # Check early stopping: update patience counter based on metric improvement
                # Note: Early stopping is disabled during SWA phase to ensure full averaging period
                early_stopping.update(state, mean_eval_acc)
                if early_stopping.should_stop(state, state.in_swa_phase):
                    state.early_stopped_at_epoch = epoch
                    LOGGER.info(
                        "Early stopping triggered at epoch %d: "
                        "No improvement for %d validation checks. Best metric: %.4f",
                        epoch,
                        early_stopping.patience,
                        state.best_metric_for_patience,
                    )
                    break  # Exit eval_output loop to stop training

            # If early stopping was triggered, exit the epoch loop
            if state.early_stopped_at_epoch is not None:
                break

            # Build and store epoch metrics
            epoch_metrics, latest_val, train_accuracy, val_loss, val_accuracy = _build_epoch_metrics(
                epoch, tasks, train_result, eval_outputs, eval_manager, resources, config
            )
            state.history.append(epoch_metrics)
            train_losses = epoch_metrics.train_losses
            val_losses = epoch_metrics.val_losses

            # Print and display epoch metrics
            _print_epoch_metrics(
                tasks, task_lookup, train_result, latest_val,
                train_accuracy, val_accuracy, train_losses, val_losses
            )
            _update_progress_display(
                progress_bar, train_result, train_accuracy, val_loss, val_accuracy
            )

            # Handle periodic checkpointing based on checkpoint_strategy
            should_save_checkpoint = False
            if checkpoint_manager.strategy == "epoch":
                should_save_checkpoint = checkpoint_manager.on_epoch_end(epoch)
            elif checkpoint_manager.strategy == "steps":
                should_save_checkpoint = checkpoint_manager.on_step(state.global_step)

            if should_save_checkpoint:
                checkpoint_manager.save_periodic_checkpoint(
                    resources.model,
                    epoch,
                    training_cfg.checkpoint_total_limit,
                )

            # Update SWA model and scheduler (if in SWA phase)
            if state.in_swa_phase:
                # Update SWA averaged model weights
                if resources.swa_model is not None:
                    resources.swa_model.update_parameters(cast(nn.Module, resources.model))
                    LOGGER.debug(f"Updated SWA model weights at epoch {epoch}")

    except KeyboardInterrupt:
        # User interrupted training - ensure progress bar is closed before re-raising
        LOGGER.info("Training interrupted by user")
        progress_bar.close()
        raise

    except Exception as e:
        # On training exception (e.g., CUDA OOM), close MLflow run immediately
        # so that subsequent HPO trials can start fresh runs
        if metrics_logger is not None:
            try:
                metrics_logger.close()
            except Exception:
                pass  # Ignore errors during logger cleanup
        raise

    finally:
        # Ensure cleanup always happens (even if exception was raised)
        # This guarantees progress_bar.close() is called to prevent signal handler issues
        if not progress_bar.disable:  # Only close if not already closed
            try:
                progress_bar.close()
            except Exception:
                pass  # Ignore errors during progress bar cleanup

    # Finalize SWA model if enabled
    swa_checkpoint_path = None
    if training_cfg.swa_enabled and resources.swa_model is not None:
        swa_checkpoint_path = _finalize_swa(
            resources.swa_model, train_loader, device, checkpoint_manager
        )

    # Save final checkpoint for recovery if needed
    last_checkpoint = checkpoint_manager.save_last_checkpoint(resources.model)

    # Save epoch-level metrics history
    history_path = context.run_dir / "history.json"
    _save_history(state.history, history_path)

    # If no best checkpoint was found during training, use last checkpoint
    if state.best_metric == float("-inf") or not best_checkpoint.exists():
        best_checkpoint = last_checkpoint

    # Log checkpoints as MLflow artifacts
    if metrics_logger is not None:
        # Log best checkpoint
        if best_checkpoint.exists():
            metrics_logger.log_artifact(best_checkpoint, artifact_path="checkpoints")
        # Log SWA checkpoint if available
        if swa_checkpoint_path is not None and Path(swa_checkpoint_path).exists():
            metrics_logger.log_artifact(swa_checkpoint_path, artifact_path="checkpoints")

    # Report early stopping status
    if early_stopping_enabled and state.early_stopped_at_epoch is not None:
        LOGGER.info("Training stopped early at epoch %d", state.early_stopped_at_epoch)
        LOGGER.info("Best validation metric achieved: %.4f", state.best_metric_for_patience)
        LOGGER.info("Best model checkpoint: %s", best_checkpoint)

    # Build final training summary with checkpoint paths and metrics
    resolved_best_metric = None if state.best_metric == float("-inf") else float(state.best_metric)
    summary = TrainingSummary(
        run_dir=context.run_dir,
        best_checkpoint=best_checkpoint,
        history=state.history,
        best_metric=resolved_best_metric,
        best_epoch=state.best_epoch,
        best_global_step=state.best_metric_step,
        top_checkpoints=list(state.best_records),
        swa_checkpoint=swa_checkpoint_path,
    )
    _write_summary(summary, context.run_dir / "summary.json")
    if training_cfg.metrics_summary_path is not None:
        _export_metrics_summary(summary, training_cfg.metrics_summary_path)

    # Explicitly release DataLoader resources to prevent file handle accumulation
    # (critical when running multiple trials in HPO with sequential DataLoaders)
    try:
        del train_loader
        del val_loader
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    metrics_logger: Any = None,
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
        name: {"correct": 0, "total": 0, "targets": [], "predictions": []}
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
                # Store predictions and targets for F1 computation
                stats["targets"].append(targets[name].detach().cpu().numpy())
                stats["predictions"].append(preds.detach().cpu().numpy())
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

        # Log batch-level metrics at regular intervals during training
        if train and log_every > 0 and (batch_idx % log_every == 0 or batch_idx == len(data_loader)):
            if metrics_logger is not None:
                # Compute current batch-level metrics
                batch_metrics: dict[str, Any] = {
                    "loss": total_loss / total_examples if total_examples else 0.0,
                    "accuracy": {},
                    "f1": {},
                    "losses": {},
                }
                for name in task_weights:
                    spec = task_specs[name]
                    if spec.task_type == "classification":
                        stats = classification_stats[name]
                        batch_metrics["accuracy"][name] = stats["correct"] / max(stats["total"], 1)
                        # Compute macro F1 for batch-level logging
                        all_targets = np.concatenate(stats["targets"]) if stats["targets"] else np.array([])
                        all_predictions = np.concatenate(stats["predictions"]) if stats["predictions"] else np.array([])
                        macro_f1 = f1_score(all_targets, all_predictions, average="macro", zero_division=0) if len(all_targets) > 0 else None
                        batch_metrics["f1"][name] = macro_f1
                    elif spec.task_type == "regression":
                        stats = regression_stats[name]
                        num_examples = max(stats["total"], 1)
                        batch_metrics["accuracy"][name] = stats["abs_error_sum"] / num_examples
                    batch_metrics["losses"][name] = (task_loss_sums[name] / max(task_counts[name], 1)) if task_counts[name] else 0.0
                if train and grad_norm_count > 0:
                    batch_metrics["grad_norm"] = grad_norm_sum / grad_norm_count
                metrics_logger.log("train", global_step, batch_metrics)

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
    f1_metrics: dict[str, float | None] = {}
    mae_metrics: dict[str, float | None] = {}
    mse_metrics: dict[str, float | None] = {}
    for name in task_weights:
        spec = task_specs[name]
        if spec.task_type == "classification":
            # Classification accuracy: correct predictions / total examples
            stats = classification_stats[name]
            accuracy[name] = stats["correct"] / max(stats["total"], 1)
            # Compute macro F1 score
            all_targets = np.concatenate(stats["targets"]) if stats["targets"] else np.array([])
            all_predictions = np.concatenate(stats["predictions"]) if stats["predictions"] else np.array([])
            macro_f1 = f1_score(all_targets, all_predictions, average="macro", zero_division=0) if len(all_targets) > 0 else None
            f1_metrics[name] = macro_f1
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
    if f1_metrics:
        metrics["f1"] = f1_metrics
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


def _prepare_run_dir(config: ClassificationConfig) -> Path:
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
            "train_f1": item.train_f1,
            "val_f1": item.val_f1,
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
    per_task_best_f1: dict[str, dict[str, float | int]] = {}
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
        for task_name, f1 in epoch_metrics.val_f1.items():
            if f1 is None:
                continue
            best_entry = per_task_best_f1.get(task_name)
            if best_entry is None or f1 > best_entry["f1"]:
                per_task_best_f1[task_name] = {
                    "f1": float(f1),
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
            "f1": per_task_best_f1,
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

    experiment = load_classification_config(config)
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
