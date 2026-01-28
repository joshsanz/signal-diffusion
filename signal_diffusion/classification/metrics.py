"""Training metrics logging utilities.

Supports logging to TensorBoard, Weights & Biases, and MLflow.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from signal_diffusion.log_setup import get_logger
from signal_diffusion.classification.config import ClassificationConfig, TrainingConfig, TrainingConfig
from signal_diffusion.config import Settings

LOGGER = get_logger(__name__)

# Optional imports
try:
    import mlflow
except ImportError:
    mlflow = None  # type: ignore


def _load_dotenv() -> None:
    """Load .env file from repository root if it exists."""
    repo_root = Path(__file__).parent.parent.parent
    env_path = repo_root / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path, override=True)
            LOGGER.info(f"Loaded .env file from {env_path}")
        except ImportError:
            LOGGER.warning("python-dotenv not installed, skipping .env file load")


def get_mlflow_tracking_uri() -> str | None:
    """Get MLflow tracking URI from environment variable.

    Supports loading from .env file in repository root.
    Returns None if MLFLOW_TRACKING_URI is not set.
    """
    _load_dotenv()
    return os.environ.get("MLFLOW_TRACKING_URI")


def get_mlflow_experiment_name() -> str | None:
    """Get MLflow experiment name from environment variable.

    Supports loading from .env file in repository root.
    Returns None if MLFLOW_EXPERIMENT_NAME is not set.
    """
    _load_dotenv()
    return os.environ.get("MLFLOW_EXPERIMENT_NAME")


class MetricsLogger:
    """Log training metrics to TensorBoard, Weights & Biases, and/or MLflow."""

    def __init__(
        self,
        *,
        tasks: Iterable[str],
        training_cfg: TrainingConfig,
        settings: Settings,
        run_dir: Path,
    ) -> None:
        self.tasks = tuple(tasks)
        self._tensorboard = None
        self._wandb_run = None
        self._mlflow_run = None
        self._mlflow_run_id: str | None = None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if training_cfg.run_name:
            run_name = f"{training_cfg.run_name}-{timestamp}"
        else:
            if settings.data_type == "spectrogram":
                data_str = "spec-" + settings.output_type
            else:
                data_str = settings.data_type
            run_name = f"{data_str}-{timestamp}"

        if training_cfg.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "TensorBoard logging requested but 'torch.utils.tensorboard' is unavailable"
                ) from exc
            log_dir = training_cfg.log_dir or (run_dir / "tensorboard")
            if run_name:
                log_dir = log_dir / run_name
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
            if run_name:
                init_kwargs["name"] = run_name

            if training_cfg.wandb_entity:
                init_kwargs["entity"] = training_cfg.wandb_entity
            if training_cfg.wandb_tags:
                init_kwargs["tags"] = list(training_cfg.wandb_tags)
            self._wandb_run = wandb.init(**init_kwargs)

        # Initialize MLflow if tracking URI is configured
        mlflow_tracking_uri = get_mlflow_tracking_uri()
        if mlflow_tracking_uri:
            if mlflow is None:  # pragma: no cover - optional dependency
                LOGGER.warning("MLflow tracking URI configured but 'mlflow' is not installed")
            else:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
                experiment_name = get_mlflow_experiment_name() or "classification"
                mlflow.set_experiment(experiment_name)

                # Start MLflow run
                mlflow_run_name = run_name if run_name else f"classification-{timestamp}"
                self._mlflow_run = mlflow.start_run(
                    run_name=mlflow_run_name,
                    run_id=None,  # Let MLflow generate run_id
                )
                self._mlflow_run_id = self._mlflow_run.info.run_id
                LOGGER.info(f"MLflow tracking enabled: {mlflow_tracking_uri}, experiment='{experiment_name}'")

    def log(self, phase: str, step: int, metrics: Mapping[str, Any], *, epoch: int | None = None) -> None:
        scalars: dict[str, float] = {}
        loss = metrics.get("loss")
        if loss is not None:
            scalars[f"{phase}/loss"] = float(loss)
        for name, value in metrics.get("losses", {}).items():
            if value is None:
                continue
            scalars[f"{phase}/loss/{name}"] = float(value)

        # Log accuracy for classification tasks
        for name, value in metrics.get("accuracy", {}).items():
            if value is None:
                continue
            scalars[f"{phase}/accuracy/{name}"] = float(value)

        # Log F1 scores for classification tasks
        for name, value in metrics.get("f1", {}).items():
            if value is None:
                continue
            scalars[f"{phase}/f1/{name}"] = float(value)

        # Log MAE and MSE as accuracy/{task_name}_{metric_type} for clarity
        for name, value in metrics.get("mae", {}).items():
            if value is None:
                continue
            scalars[f"{phase}/accuracy/{name}_mae"] = float(value)
        for name, value in metrics.get("mse", {}).items():
            if value is None:
                continue
            scalars[f"{phase}/accuracy/{name}_mse"] = float(value)

        if epoch is not None:
            scalars[f"{phase}/epoch"] = float(epoch)

        grad_norm = metrics.get("grad_norm")
        if grad_norm is not None:
            scalars[f"{phase}/grad_norm"] = float(grad_norm)

        if not scalars:
            return

        if self._tensorboard is not None:
            for key, value in scalars.items():
                self._tensorboard.add_scalar(key, value, step)

        if self._wandb_run is not None:
            self._wandb_run.log(scalars, step=step)

        # Log to MLflow
        if self._mlflow_run is not None:
            try:
                mlflow.log_metrics(scalars, step=step)
            except Exception:
                LOGGER.warning("Failed to log metrics to MLflow")

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        if self._mlflow_run is not None:
            try:
                # Flatten nested dicts for MLflow
                flat_params: dict[str, Any] = {}
                for key, value in params.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            flat_params[f"{key}.{sub_key}"] = sub_value
                    else:
                        flat_params[key] = value
                mlflow.log_params(flat_params)
            except Exception:
                LOGGER.warning("Failed to log parameters to MLflow")

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        """Log an artifact to MLflow."""
        if self._mlflow_run is not None:
            try:
                mlflow.log_artifact(str(path), artifact_path=artifact_path)
            except Exception:
                LOGGER.warning(f"Failed to log artifact {path} to MLflow")

    def close(self) -> None:
        if self._tensorboard is not None:
            self._tensorboard.flush()
            self._tensorboard.close()
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except AttributeError:  # pragma: no cover - older wandb versions
                pass
        if self._mlflow_run is not None:
            try:
                mlflow.end_run()
            except Exception:
                LOGGER.warning("Failed to end MLflow run")


def create_metrics_logger(
    training_cfg: TrainingConfig,
    settings: Settings,
    tasks: Iterable[str],
    run_dir: Path,
) -> MetricsLogger | None:
    """Create a metrics logger based on training configuration.

    Args:
        training_cfg: Training configuration containing logging options.
        tasks: Tuple/list of task names being trained.
        run_dir: Directory where logs and artifacts will be stored.

    Returns:
        MetricsLogger instance if any logging backend is configured, None otherwise.
    """
    # Always create MetricsLogger to enable MLflow if configured
    # (MLflow is controlled by environment variables, not training config)
    if not training_cfg.tensorboard and not training_cfg.wandb_project and not get_mlflow_tracking_uri():
        return None
    return MetricsLogger(tasks=tasks, training_cfg=training_cfg,
                         settings=settings, run_dir=run_dir)
