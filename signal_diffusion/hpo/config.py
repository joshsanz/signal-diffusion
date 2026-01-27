"""HPO configuration mappings and constants."""

from __future__ import annotations

from pathlib import Path

# Spectrogram types
SPEC_TYPES = ["db-only", "db-polar", "db-iq", "timeseries"]

# Task objectives
TASK_TYPES = ["gender", "mixed"]

# Spec type → base config path mapping
SPEC_TO_CONFIG = {
    "db-only": "config/classification/baseline.toml",
    "db-polar": "config/classification/baseline-db-polar.toml",
    "db-iq": "config/classification/baseline-db-iq.toml",
    "timeseries": "config/classification/baseline-timeseries.toml",
}

# Task type → tasks list mapping
TASK_TYPE_TO_TASKS = {
    "gender": ["gender"],
    "mixed": ["gender", "health", "age"],
}


def get_base_config_path(spec_type: str, cwd: Path | None = None) -> Path:
    """
    Get the base configuration path for a spec type.

    Args:
        spec_type: Spectrogram type ('db-only', 'db-polar', 'db-iq', 'timeseries')
        cwd: Current working directory (defaults to Path.cwd())

    Returns:
        Resolved Path to base config file

    Raises:
        ValueError: If spec_type is not recognized
    """
    if spec_type not in SPEC_TO_CONFIG:
        raise ValueError(
            f"Unknown spec type: {spec_type}. Valid types: {list(SPEC_TO_CONFIG.keys())}"
        )

    base_config_rel = SPEC_TO_CONFIG[spec_type]
    if cwd is None:
        cwd = Path.cwd()

    return (cwd / base_config_rel).resolve()


def get_tasks_for_type(task_type: str) -> list[str]:
    """
    Get the task list for a task type.

    Args:
        task_type: Task type ('gender' or 'mixed')

    Returns:
        List of task names

    Raises:
        ValueError: If task_type is not recognized
    """
    if task_type not in TASK_TYPE_TO_TASKS:
        raise ValueError(
            f"Unknown task type: {task_type}. Valid types: {list(TASK_TYPE_TO_TASKS.keys())}"
        )

    return TASK_TYPE_TO_TASKS[task_type]


def get_optimize_task_arg(task_type: str) -> str:
    """
    Get the --optimize-task argument value for HPO script.

    Args:
        task_type: Task type ('gender' or 'mixed')

    Returns:
        Argument value for --optimize-task ('gender' or 'combined')

    Raises:
        ValueError: If task_type is not recognized
    """
    if task_type == "gender":
        return "gender"
    elif task_type == "mixed":
        return "combined"
    else:
        raise ValueError(f"Unknown task type: {task_type}. Valid types: {TASK_TYPES}")
