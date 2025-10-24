"""Classification scaffolding for Signal Diffusion."""
from .datasets import build_dataset, default_transform
from .factory import ClassifierConfig, MultiTaskClassifier, TaskSpec, build_classifier, tasks_from_registry
from .tasks import available_tasks, build_task_specs, label_registry

__all__ = [
    "available_tasks",
    "build_classifier",
    "build_dataset",
    "build_task_specs",
    "ClassifierConfig",
    "default_transform",
    "label_registry",
    "MultiTaskClassifier",
    "TaskSpec",
    "tasks_from_registry",
]
