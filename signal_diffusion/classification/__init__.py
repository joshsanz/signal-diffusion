"""Classification scaffolding for Signal Diffusion."""
from .datasets import build_dataset, default_transform
from .tasks import available_tasks, build_task_specs, label_registry

__all__ = [
    "available_tasks",
    "build_dataset",
    "build_task_specs",
    "default_transform",
    "label_registry",
]
