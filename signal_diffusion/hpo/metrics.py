"""HPO trial metrics extraction and parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TaskMetrics:
    """Metrics for a single task."""

    f1: float | None = None
    accuracy: float | None = None
    mse: float | None = None
    mae: float | None = None


@dataclass
class TrialSummary:
    """Summary of a trial's metrics and hyperparameters."""

    trial_num: int
    objective: float
    task_metrics: dict[str, TaskMetrics]
    hyperparams: dict[str, Any]

    def __repr__(self) -> str:
        parts = [f"trial_{self.trial_num}: obj={self.objective:.4f}"]
        for task_name, metrics in self.task_metrics.items():
            if metrics.f1 is not None:
                parts.append(f"{task_name}_f1={metrics.f1:.4f}")
            if metrics.accuracy is not None:
                parts.append(f"{task_name}_acc={metrics.accuracy:.4f}")
            if metrics.mae is not None:
                parts.append(f"{task_name}_mae={metrics.mae:.4f}")
            if metrics.mse is not None:
                parts.append(f"{task_name}_mse={metrics.mse:.4f}")
        return " | ".join(parts)


def extract_task_metrics(user_attrs: dict[str, Any]) -> dict[str, TaskMetrics]:
    """
    Extract task-specific metrics from user attributes.

    Parses attributes with patterns like:
    - task_{name}_f1
    - task_{name}_accuracy
    - task_{name}_mse
    - task_{name}_mae

    Args:
        user_attrs: User attributes dictionary from trial

    Returns:
        Dictionary mapping task name → TaskMetrics
    """
    task_metrics: dict[str, TaskMetrics] = {}

    for attr_name, attr_value in user_attrs.items():
        if not attr_name.startswith("task_"):
            continue

        # Parse task_<name>_<metric> pattern
        if attr_name.endswith("_f1"):
            task_name = attr_name.replace("task_", "").replace("_f1", "")
            if task_name not in task_metrics:
                task_metrics[task_name] = TaskMetrics()
            task_metrics[task_name].f1 = attr_value

        elif attr_name.endswith("_accuracy"):
            task_name = attr_name.replace("task_", "").replace("_accuracy", "")
            if task_name not in task_metrics:
                task_metrics[task_name] = TaskMetrics()
            task_metrics[task_name].accuracy = attr_value

        elif attr_name.endswith("_mse"):
            task_name = attr_name.replace("task_", "").replace("_mse", "")
            if task_name not in task_metrics:
                task_metrics[task_name] = TaskMetrics()
            task_metrics[task_name].mse = attr_value

        elif attr_name.endswith("_mae"):
            task_name = attr_name.replace("task_", "").replace("_mae", "")
            if task_name not in task_metrics:
                task_metrics[task_name] = TaskMetrics()
            task_metrics[task_name].mae = attr_value

    return task_metrics


def parse_best_trial(
    trial_num: int,
    user_attrs: dict[str, Any],
    params: dict[str, Any] | None = None,
) -> TrialSummary:
    """
    Parse trial data into structured TrialSummary object.

    Args:
        trial_num: Trial number
        user_attrs: User attributes from trial
        params: Hyperparameters for the trial

    Returns:
        TrialSummary object
    """
    objective = user_attrs.get("combined_objective", user_attrs.get("best_metric", 0.0))
    task_metrics = extract_task_metrics(user_attrs)

    return TrialSummary(
        trial_num=trial_num,
        objective=objective,
        task_metrics=task_metrics,
        hyperparams=params or {},
    )
