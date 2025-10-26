"""Classifier factory building blocks for Signal Diffusion."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, MutableMapping, Sequence

import torch
from torch import nn

from signal_diffusion.data.specs import LabelRegistry

from .backbones import CNNBackbone, TransformerBackbone

_BACKBONES = {
    "cnn_light": CNNBackbone,
    "transformer": TransformerBackbone,
}


@dataclass(slots=True)
class TaskSpec:
    """Represents a classifier head to train."""

    name: str
    output_dim: int
    weight: float = 1.0
    task_type: str = "classification"


@dataclass(slots=True)
class ClassifierConfig:
    """Configuration inputs for :func:`build_classifier`."""

    backbone: str
    input_channels: int
    tasks: Sequence[TaskSpec]
    embedding_dim: int = 256
    dropout: float = 0.3
    activation: str = "gelu"
    extras: MutableMapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "backbone": self.backbone,
            "input_channels": self.input_channels,
            "tasks": [task.__dict__ for task in self.tasks],
            "embedding_dim": self.embedding_dim,
            "dropout": self.dropout,
            "activation": self.activation,
        }
        if self.extras:
            data["extras"] = dict(self.extras)
        return data


class MLP(nn.Module):
    """Simple multi-layer perceptron with one hidden layer."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(32, input_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskClassifier(nn.Module):
    """Classifier with a shared backbone and multiple prediction heads."""

    def __init__(self, backbone: nn.Module, tasks: Sequence[TaskSpec]) -> None:
        super().__init__()
        if not tasks:
            raise ValueError("MultiTaskClassifier requires at least one task")
        self.backbone = backbone
        self.tasks = list(tasks)
        if not hasattr(backbone, "embedding_dim"):
            raise AttributeError("Backbone must expose 'embedding_dim' attribute")
        embedding_dim = getattr(backbone, "embedding_dim")
        heads = {
            task.name: MLP(embedding_dim, task.output_dim, 2 * embedding_dim) for task in self.tasks
        }
        self.heads = nn.ModuleDict(heads)

    def forward(self, x: torch.Tensor, task: str | None = None) -> torch.Tensor | dict[str, torch.Tensor]:
        features = self.backbone(x)
        logits = {name: head(features) for name, head in self.heads.items()}
        if task is not None:
            try:
                return logits[task]
            except KeyError as exc:  # pragma: no cover - validated upstream
                raise KeyError(f"Classifier does not include task '{task}'") from exc
        return logits

    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convenience wrapper returning logits for every task."""
        outputs = self.forward(x)
        if isinstance(outputs, dict):
            return outputs
        # Single-task head returns tensor directly, normalise to dict
        name = next(iter(self.heads.keys()))
        return {name: outputs}


def build_classifier(config: ClassifierConfig) -> MultiTaskClassifier:
    """Instantiate a multi-task classifier from :class:`ClassifierConfig`."""

    try:
        backbone_cls = _BACKBONES[config.backbone]
    except KeyError as exc:  # pragma: no cover - validated upstream
        raise ValueError(f"Unknown backbone '{config.backbone}'. Available: {sorted(_BACKBONES)}") from exc

    extras = dict(config.extras)
    if config.backbone == "cnn_light":
        backbone = backbone_cls(
            config.input_channels,
            activation=config.activation,
            dropout=config.dropout,
            embedding_dim=config.embedding_dim,
            **extras,
        )
    elif config.backbone == "transformer":
        if "input_dim" not in extras:
            raise ValueError("Transformer backbone requires 'input_dim' in extras")
        if "seq_length" not in extras:
            raise ValueError("Transformer backbone requires 'seq_length' in extras")
        backbone = backbone_cls(
            input_dim=int(extras.pop("input_dim")),
            seq_length=int(extras.pop("seq_length")),
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            **extras,
        )
    else:  # pragma: no cover
        raise AssertionError("Unhandled backbone type")

    return MultiTaskClassifier(backbone=backbone, tasks=config.tasks)


def tasks_from_registry(registry: LabelRegistry, task_names: Iterable[str]) -> list[TaskSpec]:
    """Translate registry entries into :class:`TaskSpec` objects."""

    specs: list[TaskSpec] = []
    for name in task_names:
        spec = registry[name]
        if spec.task_type == "classification":
            if spec.num_classes is None:
                raise ValueError(f"Label '{name}' must define num_classes for classification tasks")
            output_dim = int(spec.num_classes)
        else:
            output_dim = 1
        specs.append(TaskSpec(name=name, output_dim=output_dim, task_type=spec.task_type))
    return specs
