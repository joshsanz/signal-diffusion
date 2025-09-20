"""Legacy shim for the CHB-MIT EEG dataset.

This module keeps the historical interface intact while delegating the actual
work to the refactored implementations in ``signal_diffusion.data.mit``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from bidict import bidict
from torchvision import transforms

from signal_diffusion.config import DatasetSettings, Settings, load_settings
from signal_diffusion.data.mit import (
    MIT_CONDITION_CLASSES,
    MIT_LABELS,
    MITDataset as _MITDataset,
    MITPreprocessor as _MITPreprocessor,
)

mit_class_labels = bidict(MIT_CONDITION_CLASSES)


def _settings_for_datadir(datadir: str | Path) -> Settings:
    datadir = Path(datadir)
    base = load_settings()
    dataset_settings = DatasetSettings(
        name="mit",
        root=datadir,
        output=datadir / "stfts",
    )
    base.datasets["mit"] = dataset_settings
    base.data_root = datadir
    base.output_root = datadir / "stfts"
    return base


class MITPreprocessor(_MITPreprocessor):
    """Backwards compatible constructor forwarding to the new preprocessor."""

    def __init__(
        self,
        datadir: str | Path,
        nsamps: int,
        ovr_perc: float = 0.0,
        fs: float = 256,
        bin_spacing: str = "linear",
    ) -> None:
        self.datadir = Path(datadir)
        self.eegdir = self.datadir / "files"
        self.stfttdir = self.datadir / "stfts"
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

    def get_label(self, filename: str):
        metadata_path = self.stfttdir / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError("Run preprocess() before calling get_label")
        metadata = pd.read_csv(metadata_path)
        row = metadata.loc[metadata["file_name"] == filename]
        if row.empty:
            raise KeyError(f"Metadata for {filename!r} not found")
        record = row.iloc[0].to_dict()
        gender = record.get("gender", "M")
        seizure = int(record.get("seizure", 0))
        age = record.get("age", "")
        label_idx = seizure * 2 + (1 if str(gender).upper().startswith("F") else 0)
        label = torch.tensor(label_idx, dtype=torch.long)
        return gender, bool(seizure), age, label

    @staticmethod
    def get_caption(gender: str, seizure: bool, age: int | str) -> str:
        gender_text = "female" if str(gender).upper().startswith("F") else "male"
        seizure_text = "during a seizure" if seizure else "without seizure"
        if age not in ("", None):
            return f"an EEG spectrogram of a {age} year old, {gender_text} subject {seizure_text}"
        return f"an EEG spectrogram of a {gender_text} subject {seizure_text}"


_TASK_MAP = {
    "gender": "gender",
    "seizure": "seizure",
    "health": "seizure",
    "mit_condition": "mit_condition",
}


class MITDataset(_MITDataset):
    """Legacy dataset wrapper preserving the original tuple interface."""

    def __init__(
        self,
        datadir: str | Path,
        split: str = "train",
        transform=None,
        task: str = "gender",
    ) -> None:
        if task not in _TASK_MAP:
            raise ValueError(f"Invalid task {task!r} for MITDataset: {sorted(_TASK_MAP)}")
        settings = _settings_for_datadir(datadir)
        transform = transform or transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        super().__init__(
            settings=settings,
            split=split,
            tasks=(_TASK_MAP[task],),
            transform=transform,
            target_format="tuple",
        )
        self.task = task
        self.datadir = Path(datadir)
        self.stfttdir = self.datadir / "stfts"

    def caption(self, index: int) -> str:
        row = self.metadata.iloc[index]
        return row.get("caption") or row.get("text", "")


class MITSampler(torch.utils.data.Sampler[int]):
    """Weighted sampler mirroring the legacy behaviour."""

    def __init__(
        self,
        datadir: str | Path,
        num_samples: int,
        split: str = "train",
        replacement: bool = True,
        generator: torch.Generator | None = None,
        task: str = "seizure",
    ) -> None:
        datadir = Path(datadir)
        metadata_path = datadir / "stfts" / f"{split}-metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No metadata file found for split {split!r}")
        self.metadata = pd.read_csv(metadata_path)
        if task not in _TASK_MAP:
            raise ValueError(f"Unsupported task {task!r} for MITSampler")
        self.task = _TASK_MAP[task]
        self.weights = torch.as_tensor(self._generate_weights(), dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        yield from torch.multinomial(
            self.weights,
            self.num_samples,
            self.replacement,
            generator=self.generator,
        ).tolist()

    def _generate_weights(self) -> Sequence[float]:
        labels = [MIT_LABELS.encode(self.task, row) for _, row in self.metadata.iterrows()]
        class_counts: dict[int, int] = {}
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        total = len(labels)
        weights = []
        for label in labels:
            freq = class_counts[label] / total
            weights.append(1.0 / max(freq, 1e-8))
        weight_sum = sum(weights)
        return [w / weight_sum for w in weights]
