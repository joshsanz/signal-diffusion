"""Configuration schema for classification training."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import tomllib


@dataclass(slots=True)
class DatasetConfig:
    """Dataset-related configuration."""

    name: str
    tasks: tuple[str, ...]
    train_split: str = "train"
    val_split: str = "val"
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle: bool = True
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ModelConfig:
    """Model factory configuration for classification training."""

    backbone: str
    input_channels: int
    embedding_dim: int = 256
    dropout: float = 0.3
    activation: str = "gelu"
    depth: int = 3
    layer_repeats: int = 2
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OptimizerConfig:
    """Optimizer hyper-parameters."""

    name: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass(slots=True)
class SchedulerConfig:
    """Learning rate scheduler settings."""

    name: str = "constant"
    warmup_steps: int = 0
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingConfig:
    """Execution-related configuration values."""

    epochs: int = 25
    max_steps: int = -1
    clip_grad_norm: float | None = 1.0
    device: str | None = None
    log_every_batches: int = 10
    eval_strategy: str = "epoch"
    eval_steps: int | None = None
    output_dir: Path | None = None
    log_dir: Path | None = None
    tensorboard: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_tags: tuple[str, ...] = ()
    run_name: str | None = None
    checkpoint_total_limit: int | None = None
    checkpoint_strategy: str = "epoch"
    checkpoint_steps: int | None = None
    task_weights: dict[str, float] = field(default_factory=dict)
    use_amp: bool = False
    metrics_summary_path: Path | None = None
    max_best_checkpoints: int = 1
    early_stopping: bool = False
    early_stopping_patience: int = 5
    compile_model: bool = True
    compile_mode: str = "default"
    swa_enabled: bool = False
    swa_extra_ratio: float = 0.34
    swa_lr_frac: float = 0.25


@dataclass(slots=True)
class ClassificationConfig:
    """Root configuration for a classification training run."""

    path: Path
    settings_path: Path
    dataset: DatasetConfig
    model: ModelConfig
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data_overrides: dict[str, Any] = field(default_factory=dict)


def _path_from_value(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _split_value(value: Any, *, default: str | None = None) -> str | None:
    if value is None:
        return default
    if isinstance(value, str):
        if not value.strip():
            return default
        return value
    return str(value)


def _require_str(value: Any, default: str) -> str:
    """Return value as str, using default if value is None or empty."""
    if value is None:
        return default
    if isinstance(value, str):
        if not value.strip():
            return default
        return value
    return str(value)


def _load_dataset(section: Mapping[str, Any]) -> DatasetConfig:
    name = section.get("name")
    if not name:
        raise ValueError("Dataset configuration requires 'name'")
    tasks = section.get("tasks", ())
    if isinstance(tasks, str):
        tasks = tuple(t.strip() for t in tasks.split(",") if t.strip())
    elif isinstance(tasks, (list, tuple)):
        tasks = tuple(str(t) for t in tasks)
    else:
        tasks = ()

    if not tasks:
        raise ValueError("[dataset] section must define at least one task")

    extras_section = section.get("extras", {})
    if extras_section is None:
        extras_section = {}
    if not isinstance(extras_section, Mapping):
        raise TypeError("dataset.extras must be a mapping if provided")
    extras = dict(extras_section)

    return DatasetConfig(
        name=str(name),
        tasks=tasks,
        train_split=_require_str(section.get("train_split"), default="train"),
        val_split=_require_str(section.get("val_split"), default="val"),
        batch_size=int(section.get("batch_size", 32)),
        num_workers=int(section.get("num_workers", 4)),
        pin_memory=bool(section.get("pin_memory", True)),
        shuffle=bool(section.get("shuffle", True)),
        extras=extras,
    )


def _load_model(section: Mapping[str, Any]) -> ModelConfig:
    backbone = section.get("backbone")
    if not backbone:
        raise ValueError("Model configuration requires 'backbone'")
    input_channels = section.get("input_channels")
    if input_channels is None:
        raise ValueError("Model configuration requires 'input_channels'")

    extras_section = section.get("extras", {})
    if extras_section is None:
        extras_section = {}
    if not isinstance(extras_section, Mapping):
        raise TypeError("model.extras must be a mapping if provided")
    extras = dict(extras_section)

    return ModelConfig(
        backbone=str(backbone),
        input_channels=int(input_channels),
        embedding_dim=int(section.get("embedding_dim", 256)),
        dropout=float(section.get("dropout", 0.3)),
        activation=str(section.get("activation", "gelu")),
        depth=int(section.get("depth", 3)),
        layer_repeats=int(section.get("layer_repeats", 2)),
        extras=extras,
    )


def _load_optimizer(section: Mapping[str, Any] | None) -> OptimizerConfig:
    if not section:
        return OptimizerConfig()
    betas = section.get("betas", (0.9, 0.999))
    if isinstance(betas, (list, tuple)):
        beta_tuple = (float(betas[0]), float(betas[1]))
    else:
        raise TypeError("optimizer.betas must be a 2-element sequence")
    return OptimizerConfig(
        name=str(section.get("name", "adamw")).lower(),
        learning_rate=float(section.get("learning_rate", 3e-4)),
        weight_decay=float(section.get("weight_decay", 1e-4)),
        betas=beta_tuple,
    )


def _load_scheduler(section: Mapping[str, Any] | None) -> SchedulerConfig:
    if not section:
        return SchedulerConfig()
    kwargs = section.get("kwargs", {})
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, Mapping):
        raise TypeError("scheduler.kwargs must be a mapping if provided")
    return SchedulerConfig(
        name=str(section.get("name", "constant")).lower(),
        warmup_steps=int(section.get("warmup_steps", 0)),
        kwargs=dict(kwargs),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Expected float-compatible value, got {value!r}")


def _load_training(section: Mapping[str, Any]) -> TrainingConfig:
    output_dir = _path_from_value(section.get("output_dir"))
    log_dir = _path_from_value(section.get("log_dir"))
    metrics_summary_path = _path_from_value(section.get("metrics_summary_path"))

    task_weights = section.get("task_weights", {})
    if task_weights is None:
        task_weights = {}
    if not isinstance(task_weights, Mapping):
        raise TypeError("training.task_weights must be a mapping if provided")
    resolved_task_weights = {str(k): float(v) for k, v in task_weights.items()}

    wandb_tags = section.get("wandb_tags", ())
    if isinstance(wandb_tags, str):
        wandb_tags = tuple(t.strip() for t in wandb_tags.split(",") if t.strip())
    elif isinstance(wandb_tags, (list, tuple)):
        wandb_tags = tuple(str(t) for t in wandb_tags)
    else:
        wandb_tags = ()

    log_every = section.get("log_every_batches") or section.get("log_every", 10)

    eval_strategy = str(section.get("eval_strategy", "epoch")).strip().lower() or "epoch"
    checkpoint_strategy = str(section.get("checkpoint_strategy", "epoch")).strip().lower() or "epoch"

    return TrainingConfig(
        epochs=int(section.get("epochs", 25)),
        max_steps=int(section.get("max_steps", -1)),
        clip_grad_norm=_optional_float(section.get("clip_grad_norm", 1.0)),
        device=section.get("device"),
        log_every_batches=int(log_every),
        eval_strategy=eval_strategy,
        eval_steps=section.get("eval_steps"),
        output_dir=output_dir,
        log_dir=log_dir,
        tensorboard=bool(section.get("tensorboard", False)),
        wandb_project=section.get("wandb_project"),
        wandb_entity=section.get("wandb_entity"),
        wandb_tags=wandb_tags,
        run_name=section.get("run_name"),
        checkpoint_total_limit=section.get("checkpoint_total_limit"),
        checkpoint_strategy=checkpoint_strategy,
        checkpoint_steps=section.get("checkpoint_steps"),
        task_weights=resolved_task_weights,
        use_amp=bool(section.get("use_amp", False)),
        metrics_summary_path=metrics_summary_path,
        max_best_checkpoints=int(section.get("max_best_checkpoints", 1)),
        early_stopping=bool(section.get("early_stopping", False)),
        early_stopping_patience=int(section.get("early_stopping_patience", 5)),
        compile_model=bool(section.get("compile_model", True)),
        compile_mode=str(section.get("compile_mode", "default")).lower(),
        swa_enabled=bool(section.get("swa_enabled", False)),
        swa_extra_ratio=float(section.get("swa_extra_ratio", 0.333)),
        swa_lr_frac=float(section.get("swa_lr_frac", 0.25)),
    )


def load_classification_config(path: str | Path) -> ClassificationConfig:
    """Load a classification training configuration from TOML."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as fp:
        mapping = tomllib.load(fp)

    base_dir = config_path.parent

    settings_section = mapping.get("settings", {})
    settings_config_path = settings_section.get("config", "config/default.toml")
    if settings_config_path:
        settings_path = _path_from_value(settings_config_path)
        if settings_path is not None and not settings_path.is_absolute():
            settings_path = base_dir / settings_path
            settings_path = settings_path.resolve()
    else:
        settings_path = None

    dataset_cfg = _load_dataset(_load_section(mapping, "dataset"))
    model_cfg = _load_model(_load_section(mapping, "model"))
    optimizer_cfg = _load_optimizer(mapping.get("optimizer"))
    scheduler_cfg = _load_scheduler(mapping.get("scheduler"))
    training_cfg = _load_training(mapping.get("training", {}))

    # Extract optional [data] section overrides
    data_section = mapping.get("data", {})
    data_overrides: dict[str, Any] = {}
    if isinstance(data_section, Mapping):
        data_overrides = dict(data_section)

    cfg = ClassificationConfig(
        path=config_path,
        settings_path=settings_path if settings_path else base_dir / "config/default.toml",
        dataset=dataset_cfg,
        model=model_cfg,
        optimizer=optimizer_cfg,
        scheduler=scheduler_cfg,
        training=training_cfg,
        data_overrides=data_overrides,
    )

    return cfg


def _load_section(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    data = mapping.get(key)
    if data is None:
        raise KeyError(f"Configuration missing required section '{key}'")
    if not isinstance(data, Mapping):
        raise TypeError(f"Section '{key}' must be a mapping, got {type(data)!r}")
    return data
