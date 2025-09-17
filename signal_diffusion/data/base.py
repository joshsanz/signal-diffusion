"""Shared preprocessing scaffolding for EEG spectrogram datasets."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
import csv
import math
import random
import shutil
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from signal_diffusion.config import Settings

try:  # Optional import; subclasses can provide custom writers.
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is expected but not required for the base class.
    Image = None  # type: ignore

DEFAULT_SPLITS: Mapping[str, float] = OrderedDict([("train", 0.8), ("val", 0.1), ("test", 0.1)])


@dataclass(slots=True)
class SpectrogramExample:
    """Represents a single spectrogram sample produced by a preprocessor."""

    subject_id: str
    relative_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    image: Any | None = None
    writer: Callable[[Path], None] | None = None


@dataclass(slots=True)
class MetadataRecord:
    """Metadata row written alongside generated spectrograms."""

    file_name: str
    split: str
    values: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        row = OrderedDict([("file_name", self.file_name), ("split", self.split)])
        for key, value in self.values.items():
            if key in row:
                continue
            row[key] = value
        return row


class BaseSpectrogramPreprocessor(ABC):
    """Base class encapsulating common spectrogram preprocessing steps."""

    def __init__(self, settings: Settings, dataset_name: str) -> None:
        self.settings = settings
        self.dataset_name = dataset_name
        self.dataset_settings = settings.dataset(dataset_name)

    def preprocess(
        self,
        *,
        splits: Mapping[str, float] | None = None,
        seed: int | None = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> dict[str, list[MetadataRecord]]:
        """Generate spectrograms and metadata.

        Parameters
        ----------
        splits:
            Mapping of split names to fractional allocations. Defaults to ``train``/``val``/``test``.
        seed:
            Optional random seed for reproducible split assignments.
        overwrite:
            If ``True``, existing output directory contents will be removed before processing.
        kwargs:
            Additional parameters passed through to :meth:`generate_examples`.
        """

        if overwrite:
            self._reset_output_root()

        split_map = self._assign_splits(splits=splits, seed=seed)
        metadata_by_split: dict[str, list[MetadataRecord]] = {name: [] for name in split_map.values()}

        for subject_id, split in split_map.items():
            for example in self.generate_examples(subject_id=subject_id, split=split, **kwargs):
                record = self._persist_example(example, split)
                metadata_by_split[split].append(record)

        self._write_metadata_files(metadata_by_split)
        return metadata_by_split

    @abstractmethod
    def subjects(self) -> Sequence[str]:
        """Return the list of subject identifiers available for preprocessing."""

    @abstractmethod
    def generate_examples(
        self,
        *,
        subject_id: str,
        split: str,
        **kwargs: Any,
    ) -> Iterable[SpectrogramExample]:
        """Yield :class:`SpectrogramExample` instances for ``subject_id``."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _assign_splits(
        self,
        *,
        splits: Mapping[str, float] | None,
        seed: int | None,
    ) -> dict[str, str]:
        split_spec = self._normalise_splits(splits or DEFAULT_SPLITS)
        subjects = list(self.subjects())
        if not subjects:
            raise RuntimeError(f"Dataset '{self.dataset_name}' has no subjects to preprocess")

        rng = random.Random(seed)
        rng.shuffle(subjects)

        counts = self._compute_split_counts(len(subjects), split_spec)
        assignments: dict[str, str] = {}
        cursor = 0
        for split_name, count in counts.items():
            for subject in subjects[cursor : cursor + count]:
                assignments[subject] = split_name
            cursor += count
        return assignments

    def _normalise_splits(self, splits: Mapping[str, float]) -> OrderedDict[str, float]:
        ordered = OrderedDict()
        total = float(sum(splits.values()))
        if math.isclose(total, 0.0):
            raise ValueError("Split fractions must sum to a positive value")
        for key, value in splits.items():
            if value < 0:
                raise ValueError(f"Split fraction for '{key}' must be non-negative")
            ordered[key] = float(value) / total
        return ordered

    def _compute_split_counts(self, total: int, splits: Mapping[str, float]) -> OrderedDict[str, int]:
        remaining = total
        counts = OrderedDict()
        items = list(splits.items())
        for index, (name, fraction) in enumerate(items):
            if index == len(items) - 1:
                count = remaining
            else:
                count = min(remaining, int(round(fraction * total)))
            counts[name] = count
            remaining -= count
        # Any rounding leftovers go to the last split implicitly (already handled).
        return counts

    def _reset_output_root(self) -> None:
        root = self.dataset_settings.output
        if root.exists():
            for child in root.iterdir():
                if child.is_file():
                    child.unlink()
                else:
                    shutil.rmtree(child)
        root.mkdir(parents=True, exist_ok=True)

    def _persist_example(self, example: SpectrogramExample, split: str) -> MetadataRecord:
        relative_path = example.relative_path
        if relative_path.is_absolute():
            raise ValueError("relative_path must be relative, not absolute")

        output_dir = self.dataset_settings.output / split
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if example.writer is not None:
            example.writer(output_path)
        else:
            self._write_image_payload(example.image, output_path)

        file_name = str(Path(split) / relative_path)
        return MetadataRecord(file_name=file_name, split=split, values=dict(example.metadata))

    def _write_image_payload(self, payload: Any, output_path: Path) -> None:
        if payload is None:
            raise ValueError("SpectrogramExample must define either 'writer' or 'image'")

        if Image is not None and isinstance(payload, Image.Image):  # type: ignore[arg-type]
            payload.save(output_path)
            return

        if isinstance(payload, np.ndarray):
            self._save_numpy_image(payload, output_path)
            return

        if isinstance(payload, bytes):
            output_path.write_bytes(payload)
            return

        raise TypeError(f"Unsupported payload type: {type(payload)!r}. Provide a writer callable instead.")

    def _save_numpy_image(self, array: np.ndarray, output_path: Path) -> None:
        if array.ndim not in (2, 3):
            raise ValueError("NumPy payload must be 2D (grayscale) or 3D (HWC)")
        if Image is None:
            raise RuntimeError("Pillow is required to save numpy arrays as images")
        mode = "F" if array.dtype == np.float32 else None
        image = Image.fromarray(array, mode=mode)  # type: ignore[arg-type]
        image.save(output_path)

    def _write_metadata_files(self, metadata_by_split: Mapping[str, Sequence[MetadataRecord]]) -> None:
        for split, records in metadata_by_split.items():
            if not records:
                continue
            fieldnames = self._collect_fieldnames(records)
            metadata_path = self.dataset_settings.output / f"{split}-metadata.csv"
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with metadata_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for record in records:
                    writer.writerow(record.as_row())

    def _collect_fieldnames(self, records: Sequence[MetadataRecord]) -> list[str]:
        keys: list[str] = ["file_name", "split"]
        seen = set(keys)
        for record in records:
            for key in record.values.keys():
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        return keys

