"""Label specifications and registries for spectrogram datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

TaskType = str


@dataclass(slots=True)
class LabelSpec:
    """Metadata describing how to derive a classifier target from a metadata row."""

    name: str
    num_classes: int | None = None
    column: str | None = None
    classes: Mapping[Any, int] | None = None
    encoder: Callable[[Mapping[str, Any]], int] | None = None
    description: str | None = None
    task_type: TaskType = "classification"
    decoder: Callable[[Any], Any] | None = None

    def encode(self, row: Mapping[str, Any]) -> int:
        if self.encoder is not None:
            result = self.encoder(row)
        else:
            if self.column is None:
                raise ValueError(f"LabelSpec '{self.name}' requires encoder or column")
            value = row[self.column]
            if self.classes is None:
                result = value
            else:
                try:
                    result = self.classes[value]
                except KeyError as exc:
                    raise KeyError(f"Unexpected value {value!r} for label '{self.name}'") from exc

        if self.task_type == "classification":
            return int(result)
        try:
            return float(result)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"LabelSpec '{self.name}' expected numeric value, got {result!r}") from exc

    def decode(self, value: Any) -> Any:
        if self.task_type != "classification":
            return value

        if self.decoder is not None:
            return self.decoder(value)

        if self.classes is not None:
            int_value = int(value)
            for raw_value, encoded in self.classes.items():
                if int(encoded) == int_value:
                    return raw_value
        raise ValueError(f"LabelSpec '{self.name}' does not define a decoder")


class LabelRegistry(dict[str, LabelSpec]):
    """Registry mapping label names to specifications."""

    def register(self, spec: LabelSpec) -> None:
        if spec.name in self:
            raise ValueError(f"Label '{spec.name}' already registered")
        self[spec.name] = spec

    def encode(self, name: str, row: Mapping[str, Any]) -> int:
        return self[name].encode(row)

    def decode(self, name: str, value: Any) -> Any:
        return self[name].decode(value)
