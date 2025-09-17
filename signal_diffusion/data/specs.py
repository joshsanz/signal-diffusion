"""Label specifications and registries for spectrogram datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


@dataclass(slots=True)
class LabelSpec:
    """Metadata describing how to derive a classifier target from a metadata row."""

    name: str
    num_classes: int
    column: str | None = None
    classes: Mapping[Any, int] | None = None
    encoder: Callable[[Mapping[str, Any]], int] | None = None
    description: str | None = None

    def encode(self, row: Mapping[str, Any]) -> int:
        if self.encoder is not None:
            return int(self.encoder(row))
        if self.column is None:
            raise ValueError(f"LabelSpec '{self.name}' requires encoder or column")
        value = row[self.column]
        if self.classes is None:
            return int(value)
        try:
            return int(self.classes[value])
        except KeyError as exc:
            raise KeyError(f"Unexpected value {value!r} for label '{self.name}'") from exc


class LabelRegistry(dict[str, LabelSpec]):
    """Registry mapping label names to specifications."""

    def register(self, spec: LabelSpec) -> None:
        if spec.name in self:
            raise ValueError(f"Label '{spec.name}' already registered")
        self[spec.name] = spec

    def encode(self, name: str, row: Mapping[str, Any]) -> int:
        return self[name].encode(row)

