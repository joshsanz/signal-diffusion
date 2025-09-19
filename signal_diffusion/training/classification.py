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
from signal_diffusion.models import ClassifierConfig, build_classifier


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
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    clip_grad_norm: float | None = 1.0
    device: str | None = None
    log_every_batches: int = 10
    eval_strategy: str = "epoch"
    eval_steps: int | None = None
    output_dir: Path | None = None
    checkpoint_every: int = 0
    task_weights: dict[str, float] = field(default_factory=dict)
    use_amp: bool = False


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
    val_loss: float | None
    train_accuracy: dict[str, float]
    val_accuracy: dict[str, float | None]
    lr: float


@dataclass(slots=True)
class TrainingSummary:
    """Summary returned after training completes."""

    run_dir: Path
    best_checkpoint: Path
    history: list[EpochMetrics]


def load_experiment_config(path: str | Path) -> ClassificationExperimentConfig:
    """Load a TOML configuration file describing a classification experiment."""

    config_path = Path(path).resolve()
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
        optimizer=str(training_section.get("optimizer", "adamw")).lower(),
        learning_rate=float(training_section.get("learning_rate", 3e-4)),
        weight_decay=float(training_section.get("weight_decay", 1e-4)),
        clip_grad_norm=_optional_float(training_section.get("clip_grad_norm", 1.0)),
        device=training_section.get("device"),
        log_every_batches=int(training_section.get("log_every_batches", training_section.get("log_every", 10))),
        eval_strategy=str(training_section.get("eval_strategy", "epoch")).lower(),
        eval_steps=_optional_int(training_section.get("eval_steps")),
        output_dir=_optional_path(training_section.get("output_dir"), base_dir),
        checkpoint_every=int(training_section.get("checkpoint_every", 0)),
        task_weights={str(k): float(v) for k, v in training_section.get("task_weights", {}).items()},
        use_amp=bool(training_section.get("use_amp", False)),
    )

    _validate_eval_config(training)

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

    device_str = training_cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_cfg.batch_size,
        shuffle=dataset_cfg.shuffle,
        num_workers=dataset_cfg.num_workers,
        pin_memory=dataset_cfg.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=dataset_cfg.batch_size,
        shuffle=False,
        num_workers=dataset_cfg.num_workers,
        pin_memory=dataset_cfg.pin_memory,
    )

    run_dir = _prepare_run_dir(config)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    optimizer = _build_optimizer(model, training_cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=training_cfg.use_amp and device.type == "cuda")
    criteria = {name: nn.CrossEntropyLoss() for name in tasks}
    task_weights = _resolve_task_weights(tasks, training_cfg.task_weights)

    def run_validation() -> dict[str, Any]:
        result, _ = _run_epoch(
            model,
            data_loader=val_loader,
            criteria=criteria,
            task_weights=task_weights,
            device=device,
            optimizer=None,
            scaler=None,
            clip_grad=None,
            log_every=0,
            train=False,
        )
        return result

    eval_manager = _create_evaluation_manager(training_cfg, run_validation)

    history: list[EpochMetrics] = []
    best_metric = float("-inf")
    best_checkpoint = checkpoints_dir / "best.pt"
    global_step = 0

    for epoch in range(1, training_cfg.epochs + 1):
        train_result, global_step = _run_epoch(
            model,
            data_loader=train_loader,
            criteria=criteria,
            task_weights=task_weights,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            clip_grad=training_cfg.clip_grad_norm,
            log_every=training_cfg.log_every_batches,
            train=True,
            global_step=global_step,
            eval_manager=eval_manager,
        )

        eval_outputs = eval_manager.drain_new_results() if eval_manager else []
        for eval_output in eval_outputs:
            mean_eval_acc = sum(eval_output["accuracy"].values()) / len(tasks)
            step_info = eval_output.get("_global_step")
            print(
                f"[eval] step={step_info} val_loss={eval_output['loss']:.4f} "
                f"val_acc_mean={mean_eval_acc:.4f}"
            )
            if mean_eval_acc > best_metric:
                best_metric = mean_eval_acc
                torch.save(model.state_dict(), best_checkpoint)

        latest_val = eval_manager.latest_result if eval_manager else None
        if latest_val is not None:
            val_loss: float | None = float(latest_val["loss"])
            val_accuracy: dict[str, float | None] = {
                name: float(latest_val["accuracy"][name]) for name in tasks
            }
        else:
            val_loss = None
            val_accuracy = {name: None for name in tasks}

        lr = optimizer.param_groups[0].get("lr", training_cfg.learning_rate)
        epoch_metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_result["loss"],
            val_loss=val_loss,
            train_accuracy=train_result["accuracy"],
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
            train_acc_display = f"{train_result['accuracy'][task_name]:.4f}"
            val_value = val_accuracy[task_name]
            val_acc_display = f"{val_value:.4f}" if val_value is not None else "n/a"
            print(
                f"  - {task_name}: train_acc={train_acc_display} val_acc={val_acc_display}"
            )

        if training_cfg.checkpoint_every and epoch % training_cfg.checkpoint_every == 0:
            torch.save(model.state_dict(), checkpoints_dir / f"epoch-{epoch:03d}.pt")

    last_checkpoint = checkpoints_dir / "last.pt"
    torch.save(model.state_dict(), last_checkpoint)

    history_path = run_dir / "history.json"
    _save_history(history, history_path)

    if best_metric == float("-inf") or not best_checkpoint.exists():
        best_checkpoint = last_checkpoint

    summary = TrainingSummary(run_dir=run_dir, best_checkpoint=best_checkpoint, history=history)
    _write_summary(summary, run_dir / "summary.json")
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
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    clip_grad: float | None,
    log_every: int,
    train: bool,
    global_step: int | None = None,
    eval_manager: EvaluationManager | None = None,
) -> tuple[dict[str, Any], int | None]:
    if train:
        if optimizer is None:
            raise ValueError("Optimizer must be provided when train=True")
        if global_step is None:
            global_step = 0
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_examples = 0
    task_correct = {name: 0 for name in task_weights}
    task_counts = {name: 0 for name in task_weights}

    for batch_idx, batch in enumerate(data_loader, start=1):
        images = batch["image"].to(device)
        targets = {
            name: _to_device_tensor(batch["targets"][name], device)
            for name in task_weights
        }
        batch_size = images.shape[0]

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=scaler is not None and scaler.is_enabled()):
                outputs = model(images)
                loss = _compute_loss(outputs, targets, criteria, task_weights)

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

        total_loss += loss.item() * batch_size
        total_examples += batch_size

        for name, logits in outputs.items():
            preds = logits.argmax(dim=1)
            correct = (preds == targets[name]).sum().item()
            task_correct[name] += correct
            task_counts[name] += batch_size

        if log_every > 0 and batch_idx % log_every == 0:
            mean_loss = total_loss / total_examples if total_examples else 0.0
            mean_acc = sum(task_correct[n] / max(task_counts[n], 1) for n in task_correct) / len(task_correct)
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
    accuracy = {
        name: task_correct[name] / max(task_counts[name], 1)
        for name in task_correct
    }
    metrics = {"loss": mean_loss, "accuracy": accuracy}
    return metrics, (global_step if train else None)


def _compute_loss(
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    criteria: Mapping[str, nn.Module],
    task_weights: Mapping[str, float],
) -> torch.Tensor:
    loss_tensor: torch.Tensor | None = None
    for name, weight in task_weights.items():
        logits = outputs[name]
        target = targets[name]
        criterion = criteria[name]
        task_loss = criterion(logits, target)
        weighted = task_loss * weight
        if loss_tensor is None:
            loss_tensor = weighted
        else:
            loss_tensor = loss_tensor + weighted
    assert loss_tensor is not None, "At least one task required"
    return loss_tensor


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

    config_copy = {
        "config_path": str(config.path),
        "settings_path": str(config.settings_path),
        "dataset": asdict(config.dataset),
        "model": asdict(config.model),
        "training": asdict(config.training),
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
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _require(section: Mapping[str, Any], key: str) -> Any:
    if key not in section:
        raise ValueError(f"Configuration section missing required key '{key}'")
    return section[key]


def _save_history(history: list[EpochMetrics], path: Path) -> None:
    payload = [
        {
            "epoch": item.epoch,
            "train_loss": item.train_loss,
            "val_loss": item.val_loss,
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
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


app = typer.Typer(help="Classification training utilities for Signal Diffusion")


@app.command()
def train(config: Path, output_dir: Path | None = typer.Option(None, help="Override output directory")) -> None:
    """Train a classifier experiment from a TOML configuration."""

    experiment = load_experiment_config(config)
    if output_dir is not None:
        experiment.training.output_dir = output_dir.resolve()
    summary = train_from_config(experiment)
    print(f"Training complete. Best checkpoint saved at {summary.best_checkpoint}")


if __name__ == "__main__":  # pragma: no cover
    app()
