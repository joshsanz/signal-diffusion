"""Legacy shim for Parkinsons dataset utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from bidict import bidict
from scipy.signal import decimate
from torchvision import transforms

from signal_diffusion.config import DatasetSettings, Settings, load_settings
from signal_diffusion.data.parkinsons import (
    PARKINSONS_CONDITION_CLASSES,
    PARKINSONS_LABELS,
    ParkinsonsDataset as _ParkinsonsDataset,
    ParkinsonsPreprocessor as _ParkinsonsPreprocessor,
)



def _encode_gender(row):
    value = str(row['gender']).strip().upper()
    return 1 if value.startswith('F') else 0


def _encode_health(row):
    value = str(row['health']).strip().upper()
    return 1 if value in {'PD', 'PARKINSONS', '1', 'TRUE'} else 0


def _encode_condition(row):
    return _encode_health(row) * 2 + _encode_gender(row)


parkinsons_class_labels = bidict(PARKINSONS_CONDITION_CLASSES)
health_class_labels = bidict({0: "healthy", 1: "parkinsons"})


def _settings_for_datadir(datadir: str | Path) -> Settings:
    base = load_settings()
    datadir = Path(datadir)
    dataset_settings = DatasetSettings(
        name="parkinsons",
        root=datadir,
        output=datadir / "stfts",
    )
    base.datasets["parkinsons"] = dataset_settings
    base.data_root = datadir
    base.output_root = datadir
    return base


class ParkinsonsPreprocessor(_ParkinsonsPreprocessor):
    """Backward-compatible wrapper around the refactored preprocessor."""

    def __init__(
        self,
        datadir: str | Path,
        nsamps: int,
        ovr_perc: float = 0.0,
        fs: float = 250,
        bin_spacing: str = "linear",
    ) -> None:
        self.datadir = Path(datadir)
        self.bin_spacing = bin_spacing
        settings = _settings_for_datadir(self.datadir)
        super().__init__(
            settings=settings,
            nsamps=nsamps,
            ovr_perc=ovr_perc,
            fs=fs,
            bin_spacing=bin_spacing,
        )

    def preprocess(
        self,
        resolution: int = 512,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int | None = None,
    ):
        splits = {"train": train_frac, "val": val_frac, "test": test_frac}
        return super().preprocess(
            splits=splits,
            seed=seed,
            overwrite=True,
            resolution=resolution,
        )

    def get_num_channels(self) -> int:
        return self.n_channels

    def get_gender(self, subject_dir: str) -> str:
        return self._subject_metadata(subject_dir).gender

    def get_health(self, subject_dir: str) -> str:
        return self._subject_metadata(subject_dir).health

    def get_age(self, subject_dir: str) -> int:
        return self._subject_metadata(subject_dir).age

    def decimate(self, data):  # pragma: no cover - retained for API compatibility
        if self.decimation > 1:
            return decimate(data, self.decimation, axis=1, zero_phase=True)
        return data

    @staticmethod
    def get_caption(gender: str, health: str, age: int) -> str:
        gender_txt = "female" if str(gender).upper().startswith("F") else "male"
        health_txt = "parkinsons disease diagnosed" if str(health).upper().startswith("P") else "healthy"
        return f"an EEG spectrogram of a {age} year old, {health_txt}, {gender_txt} subject"


_TASK_MAP = {
    "gender": "gender",
    "health": "health",
    "parkinsons_condition": "parkinsons_condition",
}


class ParkinsonsDataset(_ParkinsonsDataset):
    """Legacy dataset wrapper preserving the historical tuple interface."""

    name = "Parkinsons"

    def __init__(
        self,
        datadir: str | Path,
        split: str = "train",
        transform=None,
        task: str = "gender",
    ) -> None:
        if task not in _TASK_MAP:
            raise ValueError(f"Invalid task {task!r} for ParkinsonsDataset: {sorted(_TASK_MAP)}")
        settings = _settings_for_datadir(datadir)
        super().__init__(
            settings=settings,
            split=split,
            tasks=(_TASK_MAP[task],),
            transform=transform,
            target_format="tuple",
        )
        self.task = task
        self.dataname = "parkinsons"
        self.datadir = Path(datadir)

    def caption(self, index: int) -> str:
        row = self.metadata.iloc[index]
        return row.get("caption") or row.get("text", "")


class HealthSampler(torch.utils.data.Sampler):
    def __init__(self, datadir, num_samples, split="train", replacement=True, generator=None):
        metadata_path = Path(datadir) / "stfts" / f"{split}-metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No metadata file found for split {split}: {metadata_path}")
        self.metadata = pd.read_csv(metadata_path)
        self.weights = torch.as_tensor(self.generate_weights(self.metadata), dtype=torch.double)
        self.num_samples = num_samples
        self.split = split
        self.replacement = replacement
        self.generator = generator

    def __len__(self):
        return len(self.metadata)

    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, len(self.metadata), self.replacement, generator=self.generator)
        yield from iter(rand_tensor.tolist())

    def generate_weights(self, metadata):
        labels = [_encode_health(row) for _, row in metadata.iterrows()]
        label_weights = [labels.count(i) / len(metadata) for i in range(2)]
        rankings = {label: sum(weight < label_weights[label] for weight in label_weights) for label in range(2)}
        label_weights.sort(reverse=True)
        new_label_weights = [label_weights[rankings[i]] for i in range(2)]
        output_weights = [new_label_weights[label] for label in labels]
        norm = sum(output_weights)
        return [weight / norm for weight in output_weights]


class ParkinsonsSampler(torch.utils.data.Sampler):
    def __init__(self, datadir, num_samples, split="train", replacement=True, generator=None):
        metadata_path = Path(datadir) / "stfts" / f"{split}-metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No metadata file found for split {split}: {metadata_path}")
        self.metadata = pd.read_csv(metadata_path)
        self.weights = torch.as_tensor(self.generate_weights(self.metadata), dtype=torch.double)
        self.num_samples = num_samples
        self.split = split
        self.replacement = replacement
        self.generator = generator

    def __len__(self):
        return len(self.metadata)

    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, len(self.metadata), self.replacement, generator=self.generator)
        yield from iter(rand_tensor.tolist())

    def generate_weights(self, metadata):
        labels = [_encode_condition(row) for _, row in metadata.iterrows()]
        label_weights = [labels.count(i) / len(metadata) for i in range(4)]
        rankings = {label: sum(weight < label_weights[label] for weight in label_weights) for label in range(4)}
        label_weights.sort(reverse=True)
        new_label_weights = [label_weights[rankings[i]] for i in range(4)]
        output_weights = [new_label_weights[label] for label in labels]
        norm = sum(output_weights)
        return [weight / norm for weight in output_weights]
