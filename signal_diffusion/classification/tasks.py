"""Helpers for mapping datasets to classifier task specifications."""
from __future__ import annotations

from typing import Iterable, Mapping

from signal_diffusion.data import (
    MATH_LABELS,
    MIT_LABELS,
    PARKINSONS_LABELS,
    SEED_LABELS,
)
from signal_diffusion.data.specs import LabelRegistry
from signal_diffusion.models import TaskSpec, tasks_from_registry

_DATASET_LABELS: Mapping[str, LabelRegistry] = {
    "math": MATH_LABELS,
    "parkinsons": PARKINSONS_LABELS,
    "seed": SEED_LABELS,
    "mit": MIT_LABELS,
}


def label_registry(dataset_name: str) -> LabelRegistry:
    try:
        return _DATASET_LABELS[dataset_name]
    except KeyError as exc:  # pragma: no cover - validated upstream
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. Available: {sorted(_DATASET_LABELS)}"
        ) from exc


def available_tasks(dataset_name: str) -> tuple[str, ...]:
    registry = label_registry(dataset_name)
    return tuple(registry.keys())


def build_task_specs(dataset_name: str, task_names: Iterable[str]) -> list[TaskSpec]:
    registry = label_registry(dataset_name)
    return tasks_from_registry(registry, task_names)

