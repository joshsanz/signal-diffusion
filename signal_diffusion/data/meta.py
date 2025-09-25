"""Meta-dataset for combining multiple EEG spectrogram datasets."""
from __future__ import annotations

import importlib

from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler

from signal_diffusion.config import Settings
from signal_diffusion.data.specs import LabelRegistry, LabelSpec
from signal_diffusion.log_setup import logger

# ---------------------------------------------------------------------------
# Meta-level label specifications
# ---------------------------------------------------------------------------


def _encode_gender(row: Mapping[str, object]) -> int:
    value = str(row.get("gender", "")).strip().upper()
    return 1 if value.startswith("F") else 0


def _encode_health(row: Mapping[str, object]) -> int:
    value = str(row.get("health", "")).strip().upper()
    return 1 if value in {"PD", "PARKINSONS", "1", "TRUE"} else 0


def _encode_age(row: Mapping[str, object]) -> float:
    value = row.get("age", row.get("Age"))
    if value is None:
        raise ValueError("Missing age value in metadata")
    try:
        age = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid age value {value!r}") from exc
    if age < 0:
        raise ValueError(f"Age must be non-negative, received {age}")
    return age


def _decode_gender(value: object) -> str:
    return GENDER_CLASS_LABELS[int(value)]


def _decode_health(value: object) -> str:
    return HEALTH_CLASS_LABELS[int(value)]


META_LABELS = LabelRegistry()
META_LABELS.register(
    LabelSpec(
        name="gender",
        num_classes=2,
        encoder=_encode_gender,
        description="0: male, 1: female",
        decoder=_decode_gender,
    )
)
META_LABELS.register(
    LabelSpec(
        name="health",
        num_classes=2,
        encoder=_encode_health,
        description="0: healthy, 1: parkinsons",
        decoder=_decode_health,
    )
)
META_LABELS.register(
    LabelSpec(
        name="age",
        encoder=_encode_age,
        description="Participant age in years",
        task_type="regression",
    )
)

# Shared label maps used across multiple datasets/utilities.
GENDER_CLASS_LABELS: Mapping[int, str] = {0: "male", 1: "female"}
HEALTH_CLASS_LABELS: Mapping[int, str] = {0: "healthy", 1: "parkinsons"}
GENERAL_DATASET_MAP: Mapping[int, str] = {0: "math", 1: "parkinsons", 2: "SEED"}


_LABEL_REGISTRY_CACHE: dict[tuple[str, ...], LabelRegistry] = {}


def _collect_label_specs(dataset_names: Iterable[str]) -> LabelRegistry:
    """Aggregate label specifications across the requested datasets."""

    key = tuple(dataset_names)
    if key in _LABEL_REGISTRY_CACHE:
        return _LABEL_REGISTRY_CACHE[key]

    registry: MutableMapping[str, LabelSpec] = LabelRegistry()
    for dataset_name in key:
        module = importlib.import_module(f"signal_diffusion.data.{dataset_name}")
        registry_name = f"{dataset_name.upper()}_LABELS"
        dataset_registry = getattr(module, registry_name, None)
        if dataset_registry is None:
            continue
        for label_name, spec in dataset_registry.items():
            registry[label_name] = spec

    for name, spec in META_LABELS.items():
        registry.setdefault(name, spec)

    _LABEL_REGISTRY_CACHE[key] = registry
    return registry


class MetaPreprocessor:
    """Orchestrates preprocessing for multiple datasets."""

    def __init__(self, settings: Settings, dataset_names: Sequence[str], **kwargs: Any) -> None:
        self.settings = settings
        self.dataset_names = dataset_names
        self.preprocessor_kwargs = kwargs

    def preprocess(self, **kwargs: Any) -> None:
        for name in self.dataset_names:
            logger.info(f"Preprocessing {name} dataset...")
            module = importlib.import_module(f"signal_diffusion.data.{name}")
            if name in ("mit", "seed"):
                preprocessor_class_name = f"{name.upper()}Preprocessor"
            else:
                preprocessor_class_name = f"{name.capitalize()}Preprocessor"
            preprocessor_class = getattr(module, preprocessor_class_name)
            preprocessor = preprocessor_class(self.settings, **self.preprocessor_kwargs)
            preprocessor.preprocess(**kwargs)


class MetaDataset(ConcatDataset):
    """Concatenates multiple datasets."""

    def __init__(self, settings: Settings, dataset_names: Sequence[str], split: str, tasks: Sequence[str] = ("gender",), **kwargs: Any) -> None:
        self.settings = settings
        self.dataset_names = dataset_names
        self.split = split
        self.tasks = tasks

        datasets = []
        for name in self.dataset_names:
            module = importlib.import_module(f"signal_diffusion.data.{name}")
            if name in ("mit", "seed"):
                dataset_class_name = f"{name.upper()}Dataset"
            else:
                dataset_class_name = f"{name.capitalize()}Dataset"
            dataset_class = getattr(module, dataset_class_name)
            datasets.append(dataset_class(settings, split=split, tasks=tasks, **kwargs))

        super().__init__(datasets)
        self.label_registry = _collect_label_specs(self.dataset_names)


class MetaSampler(WeightedRandomSampler):
    """Weighted sampler for meta-dataset."""

    def __init__(self, dataset: MetaDataset, num_samples: int, replacement: bool = True, generator=None) -> None:
        self.dataset = dataset
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

        weights = self._generate_weights()
        super().__init__(weights, num_samples, replacement=replacement, generator=generator)

    def _generate_weights(self) -> torch.Tensor:
        if not self.dataset.datasets:
            return torch.empty(0)

        if "gender" not in self.dataset.label_registry:
            total = sum(len(ds) for ds in self.dataset.datasets)
            if total == 0:
                return torch.empty(0)
            return torch.ones(total, dtype=torch.double) / total

        gender_spec = self.dataset.label_registry["gender"]

        metadatas: list[pd.DataFrame] = []
        encoded_genders: list[np.ndarray] = []
        for ds in self.dataset.datasets:
            if hasattr(ds, "metadata"):
                metadata = ds.metadata.copy()
            else:
                metadata_path = ds.dataset_settings.output / f"{self.dataset.split}-metadata.csv"
                if not metadata_path.exists():
                    metadata_path = ds.dataset_settings.output / "metadata.csv"
                metadata = pd.read_csv(metadata_path)
            metadatas.append(metadata)
            encoded_series = metadata.apply(lambda row: gender_spec.encode(row), axis=1).astype(int)
            encoded_genders.append(encoded_series.to_numpy())

        totals, female, male = [], 0, 0

        for encoded in encoded_genders:
            totals.append(len(encoded))
            female += int((encoded == 1).sum())
            male += int((encoded == 0).sum())

        total_samps = sum(totals)
        if total_samps == 0:
            return torch.empty(0)

        gend_weights = [female / total_samps, male / total_samps]
        dataset_weights = [total / total_samps for total in totals]

        summed_weights = []
        weights = []
        for dw in dataset_weights:
            data_gender_w = []
            for gw in gend_weights:
                data_gender_w.append(gw + dw)
            summed_weights.append(sum(data_gender_w))
            weights.append(data_gender_w)

        rankings = {}
        for label in range(len(metadatas)):
            label_weight = summed_weights[label]
            rank = 0
            for weight in summed_weights:
                if label_weight > weight:
                    rank += 1
            rankings[label] = rank

        new_label_weights = [weights[(len(metadatas) - 1) - rankings[i]] for i in range(len(metadatas))]
        output_weights = []

        for i, encoded in enumerate(encoded_genders):
            # Default to 0 weight if new_label_weights is not long enough
            female_weight = new_label_weights[i][0] if i < len(new_label_weights) and len(new_label_weights[i]) > 0 else 0
            male_weight = new_label_weights[i][1] if i < len(new_label_weights) and len(new_label_weights[i]) > 1 else 0

            gender_weights = np.where(encoded == 1, female_weight, male_weight)
            output_weights.extend(gender_weights.tolist())

        if not output_weights:
            return torch.empty(0)

        norm_fact = sum(output_weights)
        if norm_fact == 0:
            return torch.ones(len(output_weights)) / len(output_weights)

        final_weights = [w / norm_fact for w in output_weights]

        return torch.as_tensor(final_weights, dtype=torch.double)


# ---------------------------------------------------------------------------
# Backwards-compatible aliases for legacy General* helpers
# ---------------------------------------------------------------------------

GeneralPreprocessor = MetaPreprocessor
GeneralDataset = MetaDataset
GeneralSampler = MetaSampler
general_class_labels = GENDER_CLASS_LABELS
general_dataset_map = GENERAL_DATASET_MAP
