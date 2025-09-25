"""Training loop for Signal Diffusion classifier experiments."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import tomllib
import torch
from torch import nn
from torch.utils.data import DataLoader

import typer

from signal_diffusion.classification import build_dataset, build_task_specs
from signal_diffusion.config import load_settings
from signal_diffusion.models import ClassifierConfig, TaskSpec, build_classifier


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


class MetricsLogger:
    """Log training metrics to TensorBoard and/or Weights & Biases."""

    def __init__(
        self,
        *,
        tasks: Iterable[str],
        training_cfg: TrainingConfig,
        run_dir: Path,
    ) -> None:
        self.tasks = tuple(tasks)
        self._tensorboard = None
        self._wandb_run = None

        if training_cfg.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "TensorBoard logging requested but 'torch.utils.tensorboard' is unavailable"
                ) from exc
            log_dir = training_cfg.log_dir or (run_dir / "tensorboard")
            log_dir.mkdir(parents=True, exist_ok=True)
            self._tensorboard = SummaryWriter(log_dir=str(log_dir))

        if training_cfg.wandb_project:
            try:
                import wandb  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("Weights & Biases logging requested but 'wandb' is not installed") from exc
            init_kwargs: dict[str, Any] = {
                "project": training_cfg.wandb_project,
                "dir": str(run_dir),
            }
            if training_cfg.wandb_run_name:
                init_kwargs["name"] = training_cfg.wandb_run_name
            if training_cfg.wandb_entity:
                init_kwargs["entity"] = training_cfg.wandb_entity
            if training_cfg.wandb_tags:
                init_kwargs["tags"] = list(training_cfg.wandb_tags)
            self._wandb_run = wandb.init(**init_kwargs)

    def log(self, phase: str, step: int, metrics: Mapping[str, Any], *, epoch: int | None = None) -> None:
        scalars: dict[str, float] = {}
        loss = metrics.get("loss")
        if loss is not None:
            scalars[f"{phase}/loss"] = float(loss)
        for name, value in metrics.get("losses", {}).items():
            if value is None:
                continue
            scalars[f"{phase}/loss/{name}"] = float(value)
        for name, value in metrics.get("accuracy", {}).items():
            if value is None:
                continue
            scalars[f"{phase}/accuracy/{name}"] = float(value)
        if epoch is not None:
            scalars[f"{phase}/epoch"] = float(epoch)

        if not scalars:
            return

        if self._tensorboard is not None:
            for key, value in scalars.items():
                self._tensorboard.add_scalar(key, value, step)

        if self._wandb_run is not None:
            self._wandb_run.log(scalars, step=step)

    def close(self) -> None:
        if self._tensorboard is not None:
            self._tensorboard.flush()
            self._tensorboard.close()
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except AttributeError:  # pragma: no cover - older wandb versions
                pass


def _create_metrics_logger(
    training_cfg: TrainingConfig,
    tasks: Iterable[str],
    run_dir: Path,
) -> MetricsLogger | None:
    if not training_cfg.tensorboard and not training_cfg.wandb_project:
        return None
    return MetricsLogger(tasks=tasks, training_cfg=training_cfg, run_dir=run_dir)


@dataclass(slots=True)
class DatasetConfig:
    """Configuration for dataset loading."""

    name: str
    tasks: tuple[str, ...]
    train_split: str = "train"
    val_split: str = "val"
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True


@dataclass(slots=True)
class ModelConfig:
    """Configuration for classifier model construction."""

    backbone: str
    input_channels: int
    embedding_dim: int = 256
    dropout: float = 0.3
    activation: str = "gelu"
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingConfig:
    """Optimisation and runtime configuration."""

    epochs: int = 25
    max_steps: int = -1
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    clip_grad_norm: float | None = 1.0
    device: str | None = None
    log_every_batches: int = 10
    eval_strategy: str = "epoch"
    eval_steps: int | None = None
    output_dir: Path | None = None
    log_dir: Path | None = None
    tensorboard: bool = False
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_entity: str | None = None
    wandb_tags: tuple[str, ...] = ()
    checkpoint_every: int = 0
    task_weights: dict[str, float] = field(default_factory=dict)
    use_amp: bool = False
    metrics_summary_path: Path | None = None
    max_best_checkpoints: int = 1


@dataclass(slots=True)
class ClassificationExperimentConfig:
    """Top-level experiment configuration."""

    path: Path
    settings_path: Path
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig


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
    lr: float


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

    training_section = data.get("training", {})
    training = TrainingConfig(
        epochs=int(training_section.get("epochs", 25)),
        max_steps=int(training_section.get("max_steps", -1)),
        optimizer=str(training_section.get("optimizer", "adamw")).lower(),
        learning_rate=float(training_section.get("learning_rate", 3e-4)),
        weight_decay=float(training_section.get("weight_decay", 1e-4)),
        clip_grad_norm=_optional_float(training_section.get("clip_grad_norm", 1.0)),
        device=training_section.get("device"),
        log_every_batches=int(training_section.get("log_every_batches", training_section.get("log_every", 10))),
        eval_strategy=str(training_section.get("eval_strategy", "epoch")).lower(),
        eval_steps=_optional_int(training_section.get("eval_steps")),
        output_dir=_optional_path(training_section.get("output_dir"), base_dir),
        log_dir=_optional_path(training_section.get("log_dir"), base_dir),
        tensorboard=bool(training_section.get("tensorboard", False)),
        wandb_project=training_section.get("wandb_project"),
        wandb_run_name=training_section.get("wandb_run_name"),
        wandb_entity=training_section.get("wandb_entity"),
        wandb_tags=tuple(training_section.get("wandb_tags", ())),
        checkpoint_every=int(training_section.get("checkpoint_every", 0)),
        task_weights={str(k): float(v) for k, v in training_section.get("task_weights", {}).items()},
        use_amp=bool(training_section.get("use_amp", False)),
        metrics_summary_path=_optional_path(training_section.get("metrics_summary_path"), base_dir),
        max_best_checkpoints=int(training_section.get("max_best_checkpoints", 1)),
    )

    _validate_eval_config(training)
    if training.max_best_checkpoints < 1:
        raise ValueError("[training] max_best_checkpoints must be >= 1")

    return ClassificationExperimentConfig(
        path=config_path,
        settings_path=settings_path,
        dataset=dataset,
        model=model,
        training=training,
    )


def train_from_config(config: ClassificationExperimentConfig) -> TrainingSummary:
    """Run a classification experiment from a parsed configuration."""

    settings = load_settings(config.settings_path)
    dataset_cfg = config.dataset
    training_cfg = config.training

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
        extras=config.model.extras,
    )
    model = build_classifier(classifier_config)

    if training_cfg.device:
        device = torch.device(training_cfg.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)

    train_dataset = build_dataset(
        settings,
        dataset_name=dataset_cfg.name,
        split=dataset_cfg.train_split,
        tasks=tasks,
        target_format="dict",
    )
    val_dataset = build_dataset(
        settings,
        dataset_name=dataset_cfg.name,
        split=dataset_cfg.val_split,
        tasks=tasks,
        target_format="dict",
    )

    use_pin_memory = dataset_cfg.pin_memory and torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_cfg.batch_size,
        shuffle=dataset_cfg.shuffle,
        num_workers=dataset_cfg.num_workers,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=dataset_cfg.batch_size,
        shuffle=False,
        num_workers=dataset_cfg.num_workers,
        pin_memory=use_pin_memory,
    )

    run_dir = _prepare_run_dir(config)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    optimizer = _build_optimizer(model, training_cfg)
    scaler = torch.amp.GradScaler(enabled=training_cfg.use_amp and device.type == "cuda")
    criteria: dict[str, nn.Module] = {}
    for name in tasks:
        spec = task_lookup[name]
        if spec.task_type == "classification":
            criteria[name] = nn.CrossEntropyLoss()
        else:
            criteria[name] = nn.MSELoss()
    task_weights = _resolve_task_weights(tasks, training_cfg.task_weights)

    def run_validation() -> dict[str, Any]:
        result, _ = _run_epoch(
            model,
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
        )
        return result

    eval_manager = _create_evaluation_manager(training_cfg, run_validation)
    metrics_logger = _create_metrics_logger(training_cfg, tasks, run_dir)

    history: list[EpochMetrics] = []
    best_metric = float("-inf")
    best_checkpoint = checkpoints_dir / "best.pt"
    best_epoch: int | None = None
    best_metric_step: int | None = None
    best_records: list[CheckpointRecord] = []
    max_best_checkpoints = training_cfg.max_best_checkpoints
    global_step = 0

    for epoch in range(1, training_cfg.epochs + 1):
        if training_cfg.max_steps > 0 and global_step >= training_cfg.max_steps:
            break

        train_result, global_step = _run_epoch(
            model,
            data_loader=train_loader,
            criteria=criteria,
            task_weights=task_weights,
            task_specs=task_lookup,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            clip_grad=training_cfg.clip_grad_norm,
            log_every=training_cfg.log_every_batches,
            train=True,
            global_step=global_step,
            eval_manager=eval_manager,
            max_steps=training_cfg.max_steps,
        )


        if metrics_logger is not None:
            metrics_logger.log("train", global_step, train_result, epoch=epoch)

        eval_outputs = eval_manager.drain_new_results() if eval_manager else []
        for eval_output in eval_outputs:
            accuracy_values = [value for value in eval_output["accuracy"].values() if value is not None]
            if accuracy_values:
                mean_eval_acc = sum(accuracy_values) / len(accuracy_values)
                mean_eval_display = f"{mean_eval_acc:.4f}"
            else:
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

            state_dict: dict[str, torch.Tensor] | None = None
            if max_best_checkpoints > 0:
                record = CheckpointRecord(
                    path=checkpoints_dir / f"best-epoch{epoch:03d}-step{step_for_log:08d}.pt",
                    metric=float(mean_eval_acc),
                    epoch=epoch,
                    global_step=step_for_log,
                )
                insert_index: int | None = None
                for idx, existing in enumerate(best_records):
                    if record.metric > existing.metric:
                        insert_index = idx
                        break
                if insert_index is None:
                    best_records.append(record)
                else:
                    best_records.insert(insert_index, record)

                if len(best_records) > max_best_checkpoints:
                    trimmed = best_records[max_best_checkpoints:]
                    del best_records[max_best_checkpoints:]
                else:
                    trimmed = []

                if record in best_records:
                    if not record.path.exists():
                        state_dict = model.state_dict()
                        torch.save(state_dict, record.path)
                else:
                    record = None

                for trimmed_record in trimmed:
                    if trimmed_record.path.exists():
                        trimmed_record.path.unlink(missing_ok=True)
            else:
                record = None

            if mean_eval_acc > best_metric:
                best_metric = mean_eval_acc
                best_epoch = epoch
                best_metric_step = step_for_log
                if state_dict is None:
                    state_dict = model.state_dict()
                torch.save(state_dict, best_checkpoint)

        if eval_outputs:
            latest_val = eval_outputs[-1]
        else:
            latest_val = eval_manager.latest_result if eval_manager else None
        train_losses = {name: float(train_result["losses"][name]) for name in tasks}
        train_accuracy: dict[str, float | None] = {}
        for name in tasks:
            value = train_result["accuracy"].get(name)
            train_accuracy[name] = float(value) if value is not None else None

        if latest_val is not None:
            val_loss: float | None = float(latest_val["loss"])
            val_losses: dict[str, float | None] = {
                name: float(latest_val["losses"][name]) for name in tasks
            }
            val_accuracy: dict[str, float | None] = {}
            for name in tasks:
                value = latest_val["accuracy"].get(name)
                val_accuracy[name] = float(value) if value is not None else None
        else:
            val_loss = None
            val_losses = {name: None for name in tasks}
            val_accuracy = {name: None for name in tasks}

        lr = optimizer.param_groups[0].get("lr", training_cfg.learning_rate)
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_result["loss"],
            train_losses=train_losses,
            val_loss=val_loss,
            val_losses=val_losses,
            train_accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            lr=lr,
        )
        history.append(epoch_metrics)

        valid_acc = [acc for acc in val_accuracy.values() if acc is not None]
        mean_val_acc = (sum(valid_acc) / len(valid_acc)) if valid_acc else None
        val_loss_display = f"{val_loss:.4f}" if val_loss is not None else "n/a"
        mean_val_display = f"{mean_val_acc:.4f}" if mean_val_acc is not None else "n/a"
        print(
            f"Epoch {epoch}/{training_cfg.epochs}: "
            f"train_loss={train_result['loss']:.4f} val_loss={val_loss_display} "
            f"val_acc_mean={mean_val_display}"
        )
        for task_name in tasks:
            train_value = train_accuracy[task_name]
            train_acc_display = f"{train_value:.4f}" if train_value is not None else "n/a"
            val_value = val_accuracy[task_name]
            val_acc_display = f"{val_value:.4f}" if val_value is not None else "n/a"
            print(
                f"  - {task_name}: train_acc={train_acc_display} val_acc={val_acc_display}"
            )

        train_loss_line = ", ".join(f"{name}: {train_losses[name]:.4f}" for name in tasks)
        val_loss_entries = []
        for name in tasks:
            loss_value = val_losses[name]
            val_loss_entries.append(f"{name}: {loss_value:.4f}" if loss_value is not None else f"{name}: n/a")
        val_loss_line = ", ".join(val_loss_entries)
        print(f"    train_losses: {train_loss_line}")
        print(f"    val_losses: {val_loss_line}")

        if training_cfg.checkpoint_every and epoch % training_cfg.checkpoint_every == 0:
            torch.save(model.state_dict(), checkpoints_dir / f"epoch-{epoch:03d}.pt")

    last_checkpoint = checkpoints_dir / "last.pt"
    torch.save(model.state_dict(), last_checkpoint)

    history_path = run_dir / "history.json"
    _save_history(history, history_path)

    if best_metric == float("-inf") or not best_checkpoint.exists():
        best_checkpoint = last_checkpoint

    if metrics_logger is not None:
        metrics_logger.close()

    resolved_best_metric = None if best_metric == float("-inf") else float(best_metric)
    summary = TrainingSummary(
        run_dir=run_dir,
        best_checkpoint=best_checkpoint,
        history=history,
        best_metric=resolved_best_metric,
        best_epoch=best_epoch,
        best_global_step=best_metric_step,
        top_checkpoints=list(best_records),
    )
    _write_summary(summary, run_dir / "summary.json")
    if training_cfg.metrics_summary_path is not None:
        _export_metrics_summary(summary, training_cfg.metrics_summary_path)
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


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)



def _run_epoch(
    model: nn.Module,
    *,
    data_loader: DataLoader,
    criteria: Mapping[str, nn.Module],
    task_weights: Mapping[str, float],
    task_specs: Mapping[str, TaskSpec],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    clip_grad: float | None,
    log_every: int,
    train: bool,
    global_step: int | None = None,
    eval_manager: EvaluationManager | None = None,
    max_steps: int = -1,
) -> tuple[dict[str, Any], int | None]:
    if global_step is None:
        global_step = 0
    if train:
        if optimizer is None:
            raise ValueError("Optimizer must be provided when train=True")
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

    for batch_idx, batch in enumerate(data_loader, start=1):
        if max_steps > 0 and global_step >= max_steps:
            break
        images = batch["image"].to(device)
        targets = {}
        for name in task_weights:
            tensor = _to_device_tensor(batch["targets"][name], device)
            spec = task_specs[name]
            if spec.task_type == "classification":
                tensor = tensor.long()
            else:
                tensor = tensor.float()
            targets[name] = tensor
        batch_size = images.shape[0]

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=device.type, enabled=scaler is not None and scaler.is_enabled()):
                outputs = model(images)
                loss, per_task_batch = _compute_loss(
                    outputs,
                    targets,
                    criteria,
                    task_weights,
                    task_specs,
                )

        if train:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                if clip_grad is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()

            global_step += 1
            triggered = False
            if eval_manager is not None:
                triggered = eval_manager.on_step(global_step)
            if triggered:
                model.train()

        batch_loss_value = float(loss.detach().item())
        total_loss += batch_loss_value * batch_size
        total_examples += batch_size
        for name in task_counts:
            task_counts[name] += batch_size

        for name, logits in outputs.items():
            spec = task_specs[name]
            if spec.task_type != "classification":
                continue
            preds = logits.argmax(dim=1)
            correct = (preds == targets[name]).sum().item()
            stats = classification_stats[name]
            stats["correct"] += correct
            stats["total"] += batch_size

        for name, task_loss_tensor in per_task_batch.items():
            loss_value = float(task_loss_tensor.detach().item())
            task_loss_sums[name] += loss_value * batch_size

        if log_every > 0 and batch_idx % log_every == 0:
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
            phase = "train" if train else "eval"
            print(
                f"[{phase}] step {batch_idx}/{len(data_loader)} loss={mean_loss:.4f} "
                f"acc={mean_acc:.4f}"
            )

    if train and eval_manager is not None:
        triggered = eval_manager.on_epoch_end(global_step)
        if triggered:
            model.train()

    mean_loss = total_loss / max(total_examples, 1)
    accuracy: dict[str, float | None] = {}
    for name in task_weights:
        spec = task_specs[name]
        if spec.task_type == "classification":
            stats = classification_stats[name]
            accuracy[name] = stats["correct"] / max(stats["total"], 1)
        else:
            accuracy[name] = None
    per_task_mean = {
        name: (task_loss_sums[name] / max(task_counts[name], 1)) if task_counts[name] else 0.0
        for name in task_loss_sums
    }
    metrics = {"loss": mean_loss, "accuracy": accuracy, "losses": per_task_mean}
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



def _build_optimizer(model: nn.Module, training_cfg: TrainingConfig) -> torch.optim.Optimizer:
    params = model.parameters()
    name = training_cfg.optimizer
    lr = training_cfg.learning_rate
    wd = training_cfg.weight_decay
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        momentum = 0.9
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=momentum)
    raise ValueError(f"Unsupported optimizer '{training_cfg.optimizer}'")




def _to_device_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.as_tensor(value, device=device)


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
    task_part = "-".join(config.dataset.tasks)
    run_name = f"{config.dataset.name}-{config.model.backbone}-{task_part}-{timestamp}"
    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict = asdict(config.dataset)
    dataset_dict["tasks"] = list(dataset_dict.get("tasks", ()))
    model_dict = asdict(config.model)
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
) -> None:
    """Train a classifier experiment from a TOML configuration."""

    experiment = load_experiment_config(config)
    if output_dir is not None:
        experiment.training.output_dir = output_dir.resolve()
    if metrics_summary_path is not None:
        experiment.training.metrics_summary_path = metrics_summary_path.resolve()
    summary = train_from_config(experiment)
    print(f"Training complete. Best checkpoint saved at {summary.best_checkpoint}")


if __name__ == "__main__":  # pragma: no cover
    app()
