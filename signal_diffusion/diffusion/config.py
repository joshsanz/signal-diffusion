"""Configuration schema for diffusion training."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import tomllib


@dataclass(slots=True)
class DatasetConfig:
    """Dataset-related configuration."""

    identifier: str
    train_split: str | None = "train"
    val_split: str | None = None
    cache_dir: Path | None = None
    batch_size: int = 4
    eval_batch_size: int | None = None
    num_workers: int = 4
    resolution: int = 256
    center_crop: bool = False
    random_flip: bool = False
    image_column: str | None = "image"
    caption_column: str | None = "text"
    class_column: str | None = "class_label"
    # Multi-attribute conditioning columns
    gender_column: str | None = "gender"  # M/F values
    health_column: str | None = "health"  # H/PD values (healthy/parkinsons)
    age_column: str | None = "age"        # Integer age values
    dataset_type: str = "auto"
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    num_classes: int = 0
    extras: MutableMapping[str, Any] = field(default_factory=dict)




@dataclass(slots=True)
class LoRAConfig:
    """Options for applying LoRA adapters during training."""

    enabled: bool = False
    rank: int = 4
    alpha: float = 4.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = field(default_factory=lambda: ("to_q", "to_k", "to_v", "to_out.0"))
    bias: str = "none"
    scaling: float | None = None


@dataclass(slots=True)
class ModelConfig:
    """Model factory configuration for diffusion training."""

    name: str
    pretrained: str | None = None
    revision: str | None = None
    sample_size: int | None = None
    conditioning: str | None = None
    extras: MutableMapping[str, Any] = field(default_factory=dict)
    lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass(slots=True)
class ObjectiveConfig:
    """Training objective configuration."""

    prediction_type: str = "epsilon"
    scheduler: str = "ddim"
    flow_match_timesteps: int = 1000


@dataclass(slots=True)
class OptimizerConfig:
    """Optimizer hyper-parameters."""

    name: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass(slots=True)
class SchedulerConfig:
    """Learning rate scheduler settings."""

    name: str = "constant"
    warmup_steps: int = 0


@dataclass(slots=True)
class LoggingConfig:
    """Training-time logging configuration."""

    tensorboard: bool = False
    log_dir: Path | None = None
    wandb_project: str | None = None
    wandb_entity: str | None = None
    run_name: str | None = None


@dataclass(slots=True)
class InferenceConfig:
    """Inference-time sampling configuration used during evaluation."""

    denoising_steps: int = 50
    cfg_scale: float = 7.5


@dataclass(slots=True)
class TrainingConfig:
    """Execution-related configuration values."""

    seed: int = 42
    output_dir: Path | None = None
    mixed_precision: str | None = "fp16"
    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 1
    epochs: int = 1
    max_train_steps: int | None = None
    log_every_steps: int = 10
    checkpoint_interval: int | None = None
    checkpoint_total_limit: int | None = None
    resume: str | None = None
    gradient_clip_norm: float | None = 1.0
    ema_decay: float | None = 0.999
    ema_power: float = 0.75
    ema_inv_gamma: float = 1.0
    ema_update_after_step: int = 5000
    ema_use_ema_warmup: bool = True
    allow_tf32: bool = True
    snr_gamma: float | None = None
    eval_num_examples: int = 0
    eval_mmd_samples: int = 0
    eval_mmd_fallback_ntrain: int = 0
    eval_strategy: str = "epoch"
    eval_num_steps: int = 0
    eval_batch_size: int = 0


@dataclass(slots=True)
class DiffusionConfig:
    """Root configuration for a diffusion training run."""

    dataset: DatasetConfig
    model: ModelConfig
    objective: ObjectiveConfig
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    settings_config: Path | None = None
    source_path: Path | None = None
    settings: Any = None  # Will be populated with Settings object after loading

    def validate(self):
        """Validate configuration for incompatible combinations."""
        import warnings

        # Validate conditioning mode requirements
        conditioning = (self.model.conditioning or "none").strip().lower()
        conditioning_mode = self.model.extras.get("conditioning_mode", "class_age")

        if conditioning == "classes" and conditioning_mode == "class_age":
            # Multi-attribute class conditioning requires gender and health columns
            if not self.dataset.gender_column:
                raise ValueError(
                    "class_age conditioning requires 'dataset.gender_column' to be set"
                )
            if not self.dataset.health_column:
                raise ValueError(
                    "class_age conditioning requires 'dataset.health_column' to be set"
                )
            # age_column is optional - missing values treated as CFG dropout

        if conditioning == "caption":
            if not self.dataset.caption_column:
                raise ValueError(
                    "caption conditioning requires 'dataset.caption_column' to be set"
                )

        # Warn if SD model used without latent space
        if self.model.name.startswith("stable-diffusion"):
            latent_space = self.model.extras.get("latent_space", True)
            if not latent_space:
                warnings.warn(
                    "Stable Diffusion models work best in latent space. "
                    "Consider setting model.extras.latent_space = true"
                )

        if self.settings is None:
            return  # Settings not loaded yet

        # Check for Stable Diffusion + time-series incompatibility
        if (
            hasattr(self.settings, "data_type")
            and self.settings.data_type == "timeseries"
            and self.model.name == "stable-diffusion-v1-5"
        ):
            raise ValueError(
                "Stable Diffusion models require 2D image data. "
                "For timeseries data, use 'dit', 'hourglass', or 'localmamba'."
            )

        # DiT does not support asymmetric patch sizes for time-series inputs
        if (
            hasattr(self.settings, "data_type")
            and self.settings.data_type == "timeseries"
            and self.model.name == "dit"
        ):
            patch_size = self.model.extras.get("patch_size")
            if isinstance(patch_size, (list, tuple)):
                raise ValueError(
                    "DiT adapter does not support asymmetric patches for time-series. "
                    "Use 'hourglass' or 'localmamba' instead for patch sizes like [1, 8]."
                )


def _path_from_value(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _load_section(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    data = mapping.get(key)
    if data is None:
        raise KeyError(f"Configuration missing required section '{key}'")
    if not isinstance(data, Mapping):
        raise TypeError(f"Section '{key}' must be a mapping, got {type(data)!r}")
    return data


def _load_dataset(section: Mapping[str, Any]) -> DatasetConfig:
    identifier = section.get("name") or section.get("identifier")
    if not identifier:
        raise ValueError("Dataset configuration requires 'name' or 'identifier'")
    cache_dir = _path_from_value(section.get("cache_dir"))

    extras_section = section.get("extras", {})
    if extras_section is None:
        extras_section = {}
    if not isinstance(extras_section, Mapping):
        raise TypeError("dataset.extras must be a mapping if provided")
    extras = dict(extras_section)

    def _with_default(value: str | None, default: str) -> str | None:
        """Apply default column name if value is None or empty string.

        Args:
            value: User-provided column name (from TOML config)
            default: Default column name to use if value is None or ""

        Returns:
            The default if value is None/"", otherwise the provided value
        """
        if value in (None, ""):
            return default
        return value

    def _split(value: Any, *, default: str | None = None) -> str | None:
        if value is None:
            return default
        if isinstance(value, str):
            if not value.strip():
                return None
            return value
        return str(value)

    return DatasetConfig(
        identifier=str(identifier),
        train_split=_split(section.get("train_split"), default="train"),
        val_split=_split(section.get("val_split")),
        cache_dir=cache_dir,
        batch_size=int(section.get("batch_size", 4)),
        eval_batch_size=section.get("eval_batch_size"),
        num_workers=int(section.get("num_workers", 4)),
        resolution=int(section.get("resolution", 256)),
        center_crop=bool(section.get("center_crop", False)),
        random_flip=bool(section.get("random_flip", False)),
        image_column=_with_default(section.get("image_column"), "image"),
        caption_column=_with_default(section.get("caption_column"), "text"),
        class_column=_with_default(section.get("class_column"), "class_label"),
        gender_column=_with_default(section.get("gender_column"), "gender"),
        health_column=_with_default(section.get("health_column"), "health"),
        age_column=_with_default(section.get("age_column"), "age"),
        dataset_type=str(section.get("dataset_type", "auto")),
        max_train_samples=section.get("max_train_samples"),
        max_eval_samples=section.get("max_eval_samples"),
        num_classes=int(section.get("num_classes", 0) or 0),
        extras=extras,
    )


def _load_lora(section: Mapping[str, Any] | None) -> LoRAConfig:
    if not section:
        return LoRAConfig()
    enabled = bool(section.get("enabled", False))
    target_modules = section.get("target_modules", ("to_q", "to_k", "to_v", "to_out.0"))
    if isinstance(target_modules, str):
        target_modules = tuple(token.strip() for token in target_modules.split(",") if token.strip())
    else:
        target_modules = tuple(target_modules)
    return LoRAConfig(
        enabled=enabled,
        rank=int(section.get("rank", 4)),
        alpha=float(section.get("alpha", 1.0)),
        dropout=float(section.get("dropout", 0.0)),
        target_modules=target_modules,
        bias=str(section.get("bias", "none")),
        scaling=section.get("scaling"),
    )


def _load_model(section: Mapping[str, Any]) -> ModelConfig:
    extras_section = section.get("extras", {})
    if extras_section is None:
        extras_section = {}
    if not isinstance(extras_section, Mapping):
        raise TypeError("model.extras must be a mapping if provided")
    extras = dict(extras_section)
    conditioning = section.get("conditioning")
    if conditioning is None and "conditioning" in extras:
        conditioning = extras.pop("conditioning")
    else:
        extras.pop("conditioning", None)
    for key, value in section.items():
        if key in {"name", "pretrained", "revision", "sample_size", "conditioning", "lora", "extras"}:
            continue
        extras[key] = value
    return ModelConfig(
        name=str(section.get("name")),
        pretrained=section.get("pretrained"),
        revision=section.get("revision"),
        sample_size=section.get("sample_size"),
        conditioning=conditioning,
        extras=extras,
        lora=_load_lora(section.get("lora")),
    )


def _load_objective(section: Mapping[str, Any]) -> ObjectiveConfig:
    return ObjectiveConfig(
        prediction_type=str(section.get("prediction_type", "epsilon")),
        scheduler=str(section.get("scheduler", "ddim")),
        flow_match_timesteps=int(section.get("flow_match_timesteps", 1000)),
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
        name=str(section.get("name", "adamw")),
        learning_rate=float(section.get("learning_rate", 1e-4)),
        weight_decay=float(section.get("weight_decay", 1e-2)),
        betas=beta_tuple,
        eps=float(section.get("eps", 1e-8)),
    )


def _load_scheduler(section: Mapping[str, Any] | None) -> SchedulerConfig:
    if not section:
        return SchedulerConfig()
    return SchedulerConfig(
        name=str(section.get("name", "constant")),
        warmup_steps=int(section.get("warmup_steps", 0)),
    )


def _load_logging(section: Mapping[str, Any] | None) -> LoggingConfig:
    if not section:
        return LoggingConfig()
    log_dir = _path_from_value(section.get("log_dir"))
    return LoggingConfig(
        tensorboard=bool(section.get("tensorboard", False)),
        log_dir=log_dir,
        wandb_project=section.get("wandb_project"),
        wandb_entity=section.get("wandb_entity"),
        run_name=section.get("run_name"),
    )


def _load_inference(section: Mapping[str, Any] | None) -> InferenceConfig:
    if not section:
        return InferenceConfig()
    return InferenceConfig(
        denoising_steps=int(section.get("denoising_steps", 50)),
        cfg_scale=float(section.get("cfg_scale", 7.5)),
    )


def _load_training(section: Mapping[str, Any]) -> TrainingConfig:
    output_dir = _path_from_value(section.get("output_dir"))
    resume = _path_from_value(section.get("resume"))
    strategy_value = section.get("eval_strategy", "epoch")
    strategy = str(strategy_value).strip().lower() or "epoch"

    raw_ema_decay = section.get("ema_decay", 0.999)
    ema_decay = float(raw_ema_decay) if raw_ema_decay is not None else None
    ema_use_warmup_default = section.get("ema_use_ema_warmup")
    if ema_use_warmup_default is None:
        ema_use_warmup_default = section.get("ema_warmup", True)

    return TrainingConfig(
        seed=int(section.get("seed", 42)),
        output_dir=output_dir,
        mixed_precision=section.get("mixed_precision", "fp16"),
        gradient_checkpointing=bool(
            section.get("gradient_checkpointing", section.get("grad_checkpointing", False))
        ),
        gradient_accumulation_steps=int(section.get("gradient_accumulation_steps", 1)),
        epochs=int(section.get("epochs", 1)),
        max_train_steps=section.get("max_train_steps"),
        log_every_steps=int(section.get("log_every_steps", 10)),
        checkpoint_interval=section.get("checkpoint_interval"),
        checkpoint_total_limit=section.get("checkpoint_total_limit"),
        resume=str(resume) if resume else None,
        gradient_clip_norm=section.get("gradient_clip_norm"),
        ema_decay=ema_decay,
        ema_power=float(section.get("ema_power", 0.75)),
        ema_inv_gamma=float(section.get("ema_inv_gamma", 1.0)),
        ema_update_after_step=int(section.get("ema_update_after_step", 5000)),
        ema_use_ema_warmup=bool(ema_use_warmup_default),
        allow_tf32=bool(section.get("allow_tf32", True)),
        snr_gamma=section.get("snr_gamma"),
        eval_num_examples=int(section.get("eval_num_examples", 0)),
        eval_mmd_samples=int(section.get("eval_mmd_samples", 0)),
        eval_mmd_fallback_ntrain=int(section.get("eval_mmd_fallback_ntrain", 0)),
        eval_strategy=strategy,
        eval_num_steps=int(section.get("eval_num_steps", 0)),
        eval_batch_size=int(section.get("eval_batch_size", 0)),
    )


def load_diffusion_config(path: str | Path) -> DiffusionConfig:
    """Load a diffusion training configuration from TOML."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as fp:
        mapping = tomllib.load(fp)

    dataset_cfg = _load_dataset(_load_section(mapping, "dataset"))
    model_cfg = _load_model(_load_section(mapping, "model"))
    objective_cfg = _load_objective(_load_section(mapping, "objective"))
    optimizer_cfg = _load_optimizer(mapping.get("optimizer"))
    scheduler_cfg = _load_scheduler(mapping.get("scheduler"))
    logging_cfg = _load_logging(mapping.get("logging"))
    training_cfg = _load_training(_load_section(mapping, "training"))
    inference_cfg = _load_inference(mapping.get("inference"))

    settings_section = mapping.get("settings", {})
    settings_config = _path_from_value(settings_section.get("config")) if isinstance(settings_section, Mapping) else None

    cfg = DiffusionConfig(
        dataset=dataset_cfg,
        model=model_cfg,
        objective=objective_cfg,
        optimizer=optimizer_cfg,
        scheduler=scheduler_cfg,
        training=training_cfg,
        logging=logging_cfg,
        inference=inference_cfg,
        settings_config=settings_config,
        source_path=config_path,
    )

    if settings_config:
        from signal_diffusion.config import load_settings

        cfg.settings = load_settings(settings_config)
        data_section = mapping.get("data", {})
        if isinstance(data_section, Mapping):
            if "data_type" in data_section:
                cfg.settings.data_type = str(data_section["data_type"])
            if "output_type" in data_section:
                cfg.settings.output_type = str(data_section["output_type"])

    cfg.validate()

    return cfg
