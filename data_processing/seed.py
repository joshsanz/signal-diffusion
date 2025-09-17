"""Legacy shim for SEED dataset utilities."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bidict import bidict
from scipy.signal import decimate
from torchvision import transforms

from signal_diffusion.config import DatasetSettings, Settings, load_settings
from signal_diffusion.data.seed import (
    EMOTION_MAP,
    EMOTION_NAMES,
    SEED_CONDITION_CLASSES,
    SEED_LABELS,
    SeedDataset as _SeedDataset,
    SeedPreprocessor as _SeedPreprocessor,
    START_SECOND,
    TRIAL_EMOTIONS,
)


emotion_map = bidict(EMOTION_MAP)
emotion_class_labels = {i: EMOTION_NAMES[i] for i in range(len(EMOTION_NAMES))}
seed_class_labels = bidict(SEED_CONDITION_CLASSES)

def _settings_for_datadir(datadir: str | Path) -> Settings:
    base = load_settings()
    datadir = Path(datadir)
    dataset_settings = DatasetSettings(
        name="seed",
        root=datadir,
        output=datadir / "stfts",
    )
    base.datasets["seed"] = dataset_settings
    base.data_root = datadir
    base.output_root = datadir
    return base


class SEEDPreprocessor(_SeedPreprocessor):
    """Backward-compatible wrapper around the refactored SEED preprocessor."""

    def __init__(
        self,
        datadir: str | Path,
        nsamps: int,
        ovr_perc: float = 0.0,
        fs: int = 250,
        bin_spacing: str = "linear",
    ) -> None:
        self.datadir = Path(datadir)
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

    def get_gender(self, subject: int) -> str:
        subject_id = f"sub-{subject+1:02d}"
        return self._subject_metadata(subject_id).gender

    def get_age(self, subject: int) -> int:
        subject_id = f"sub-{subject+1:02d}"
        return self._subject_metadata(subject_id).age

    def get_emotion(self, session: int, trial: int, text: bool = False):
        emotion = TRIAL_EMOTIONS[session][trial]
        return EMOTION_NAMES[emotion] if text else emotion

    def get_caption(self, subject: int, session: int, trial: int) -> str:
        subject_id = f"sub-{subject+1:02d}"
        info = self._subject_metadata(subject_id)
        emotion_id = TRIAL_EMOTIONS[session][trial]
        return self.caption(info.age, info.gender, emotion_id)

    def decimate(self, data):  # pragma: no cover - retained for API compatibility
        if self.decimation > 1:
            return decimate(data, self.decimation, axis=1, zero_phase=True)
        return data


_TASK_MAP = {
    "gender": "gender",
    "emotion": "emotion",
    "seed_condition": "seed_condition",
}


class SEEDDataset(_SeedDataset):
    """Legacy dataset wrapper preserving the historical tuple interface."""

    name = "SEED_V"

    def __init__(
        self,
        datadir: str | Path,
        split: str = "train",
        transform=None,
        task: str = "emotion",
    ) -> None:
        if task not in _TASK_MAP:
            raise ValueError(f"Invalid task {task!r} for SEEDDataset: {sorted(_TASK_MAP)}")
        settings = _settings_for_datadir(datadir)
        super().__init__(
            settings=settings,
            split=split,
            tasks=(_TASK_MAP[task],),
            transform=transform,
            target_format="tuple",
        )
        self.task = task
        self.dataname = "seed"
        self.datadir = Path(datadir)

    def caption(self, index: int) -> str:
        row = self.metadata.iloc[index]
        return row.get("caption") or row.get("text", "")


class EmotionSampler(torch.utils.data.Sampler):
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
        emotions = [_encode_emotion(row) for _, row in metadata.iterrows()]
        label_weights = [emotions.count(i) / len(metadata) for i in range(5)]
        rankings = {label: sum(weight < label_weights[label] for weight in label_weights) for label in range(5)}
        label_weights.sort(reverse=True)
        new_label_weights = [label_weights[rankings[i]] for i in range(5)]
        output_weights = [new_label_weights[label] for label in emotions]
        norm = sum(output_weights)
        return [weight / norm for weight in output_weights]


def _encode_emotion(row):
    emotion = row.get("emotion_id", row.get("emotion"))
    if isinstance(emotion, str):
        return EMOTION_MAP[emotion]
    return int(emotion)


class SeedConditionSampler(torch.utils.data.Sampler):
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
        label_weights = [labels.count(i) / len(metadata) for i in range(10)]
        rankings = {label: sum(weight < label_weights[label] for weight in label_weights) for label in range(10)}
        label_weights.sort(reverse=True)
        new_label_weights = [label_weights[rankings[i]] for i in range(10)]
        output_weights = [new_label_weights[label] for label in labels]
        norm = sum(output_weights)
        return [weight / norm for weight in output_weights]


def _encode_condition(row):
    gender = 1 if str(row["gender"]).strip().upper().startswith("F") else 0
    emotion = _encode_emotion(row)
    return emotion * 2 + gender

