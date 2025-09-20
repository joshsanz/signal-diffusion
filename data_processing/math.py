"""Legacy shim for Math dataset utilities.

This module preserves the historical interface while delegating to the
refactored implementations in ``signal_diffusion.data.math``.
"""
from __future__ import annotations

from pathlib import Path
import torch
from bidict import bidict
from scipy.signal import decimate

from signal_diffusion.config import DatasetSettings, Settings, load_settings
from signal_diffusion.data.math import (
    MATH_CONDITION_CLASSES,
    MathDataset as _MathDataset,
    MathPreprocessor as _MathPreprocessor,
)

math_class_labels = bidict(MATH_CONDITION_CLASSES)


def _settings_for_datadir(datadir: str | Path) -> Settings:
    base = load_settings()
    datadir = Path(datadir)
    dataset_settings = DatasetSettings(
        name="math",
        root=datadir,
        output=datadir / "stfts",
    )
    base.datasets["math"] = dataset_settings
    base.data_root = datadir
    base.output_root = datadir
    return base


class MathPreprocessor(_MathPreprocessor):
    """Backwards compatible constructor forwarding to the new preprocessor."""

    def __init__(
        self,
        datadir: str | Path,
        nsamps: int,
        ovr_perc: float = 0.0,
        fs: float = 250,
        bin_spacing: str = "linear",
        include_math_trials: bool = False,
    ) -> None:
        self.datadir = Path(datadir)
        self.eegdir = self.datadir / "raw_eeg"
        self.stfttdir = self.datadir / "stfts"
        settings = _settings_for_datadir(self.datadir)
        super().__init__(
            settings=settings,
            nsamps=nsamps,
            ovr_perc=ovr_perc,
            fs=fs,
            bin_spacing=bin_spacing,
            include_math_trials=include_math_trials,
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
        subject, state_with_ext = filename.split("_")
        state = int(state_with_ext.split(".")[0])
        info = self._subject_metadata(subject)
        doing_math = int(state == 2)
        label_idx = doing_math * 2 + (1 if info.gender.upper().startswith("F") else 0)
        label = torch.tensor(label_idx, dtype=torch.long)
        return info.gender, doing_math, info.age, label

    @staticmethod
    def decimate(self, data):
        if self.decimation > 1:
            return decimate(data, self.decimation, axis=1, zero_phase=True)
        return data

    def get_caption(gender: str, doing_math: int, age: int) -> str:
        gender_text = "female" if str(gender).upper().startswith("F") else "male"
        activity = "doing math" if doing_math else "background"
        return f"an EEG spectrogram of a {age} year old, {activity}, {gender_text} subject"


_TASK_MAP = {
    "gender": "gender",
    "math": "math_activity",
    "math_condition": "math_condition",
}


class MathDataset(_MathDataset):
    """Legacy dataset wrapper preserving the original tuple interface."""

    def __init__(
        self,
        datadir: str | Path,
        split: str = "train",
        transform=None,
        task: str = "gender",
    ) -> None:
        if task not in _TASK_MAP:
            raise ValueError(f"Invalid task {task!r} for MathDataset: {sorted(_TASK_MAP)}")
        settings = _settings_for_datadir(datadir)
        super().__init__(
            settings=settings,
            split=split,
            tasks=(_TASK_MAP[task],),
            transform=transform,
            target_format="tuple",
        )
        self.task = task
        self.dataname = "math"
        self.datadir = datadir

    def caption(self, index: int) -> str:
        row = self.metadata.iloc[index]
        return row.get("caption") or row.get("text", "")
