"""Parkinsons EEG dataset preprocessing and dataset utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import glob
import math as pymath

import mne
import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import decimate
from torchvision import transforms

from common.multichannel_spectrograms import multichannel_spectrogram
from data_processing.channel_map import parkinsons_channels

from signal_diffusion.config import DatasetSettings, Settings
from signal_diffusion.data.base import BaseSpectrogramPreprocessor, SpectrogramExample
from signal_diffusion.data.specs import LabelRegistry, LabelSpec

mne.set_log_level("WARNING")

HEALTH_LABELS = {0: "healthy", 1: "parkinsons"}
GENDER_LABELS = {0: "male", 1: "female"}


def _encode_gender(row: Mapping[str, object]) -> int:
    value = str(row["gender"]).strip().upper()
    return 1 if value.startswith("F") else 0


def _encode_health(row: Mapping[str, object]) -> int:
    value = str(row["health"]).strip().upper()
    return 1 if value in {"PD", "PARKINSONS", "1", "TRUE"} else 0


def _encode_condition(row: Mapping[str, object]) -> int:
    gender = _encode_gender(row)
    health = _encode_health(row)
    return health * 2 + gender


PARKINSONS_CONDITION_CLASSES = {
    0: "healthy_male",
    1: "healthy_female",
    2: "parkinsons_male",
    3: "parkinsons_female",
}


PARKINSONS_LABELS = LabelRegistry()
PARKINSONS_LABELS.register(
    LabelSpec(
        name="gender",
        num_classes=2,
        encoder=_encode_gender,
        description="0: male, 1: female",
    )
)
PARKINSONS_LABELS.register(
    LabelSpec(
        name="health",
        num_classes=2,
        encoder=_encode_health,
        description="0: healthy, 1: parkinsons",
    )
)
PARKINSONS_LABELS.register(
    LabelSpec(
        name="parkinsons_condition",
        num_classes=4,
        encoder=_encode_condition,
        description="Combined health/gender label",
    )
)


@dataclass(slots=True)
class ParkinsonsSubjectInfo:
    subject: str
    gender: str
    health: str
    age: int


class ParkinsonsPreprocessor(BaseSpectrogramPreprocessor):
    """Preprocess Parkinsons EEG recordings into spectrogram datasets."""

    DEFAULT_RESOLUTION = 512

    def __init__(
        self,
        settings: Settings,
        *,
        nsamps: int,
        ovr_perc: float = 0.0,
        fs: int = 250,
        bin_spacing: str = "linear",
    ) -> None:
        super().__init__(settings, dataset_name="parkinsons")
        self.nsamps = nsamps
        self.overlap_fraction = float(ovr_perc)
        self.noverlap = int(np.floor(nsamps * self.overlap_fraction))
        self.bin_spacing = bin_spacing

        self.data_dir = self.dataset_settings.root
        self.participants = self._load_participants()

        orig_fs = 500
        if fs <= 0 or orig_fs % fs != 0:
            raise ValueError("fs must be a positive divisor of 500 Hz")
        self.target_fs = fs
        self.decimation = orig_fs // fs
        self.fs = orig_fs / self.decimation

        self.channel_indices = [idx for _, idx in parkinsons_channels]
        self.n_channels = len(self.channel_indices)
        self._subject_ids: Sequence[str] | None = None

    # ------------------------------------------------------------------
    # BaseSpectrogramPreprocessor hooks
    # ------------------------------------------------------------------
    def subjects(self) -> Sequence[str]:
        if self._subject_ids is None:
            subs = [
                entry.name
                for entry in self.data_dir.iterdir()
                if entry.is_dir() and entry.name.startswith("sub")
            ]
            self._subject_ids = tuple(sorted(subs))
        return self._subject_ids

    def generate_examples(
        self,
        *,
        subject_id: str,
        split: str,
        resolution: int | None = None,
        hop_length: int | None = None,
    ) -> Iterable[SpectrogramExample]:
        resolution = resolution or self.DEFAULT_RESOLUTION
        hop_length = hop_length or self._derive_hop_length(resolution)

        info = self._subject_metadata(subject_id)
        data = self._load_subject_data(subject_id)
        if data is None:
            return

        total_samples = data.shape[1]
        shift = self.nsamps - self.noverlap
        if shift <= 0:
            raise ValueError("Overlap percentage results in non-positive shift size")
        if self.noverlap != 0:
            nblocks = pymath.floor((total_samples - self.nsamps) / shift) + 1
        else:
            nblocks = pymath.floor(total_samples / self.nsamps)
        if nblocks <= 0:
            raise ValueError(f"Recording {subject_id} too short for nsamps={self.nsamps}")

        start = 0
        end = self.nsamps
        counter = 0
        while end <= total_samples:
            blk = data[:, start:end]
            start += shift
            end += shift

            image = multichannel_spectrogram(
                blk,
                resolution=resolution,
                hop_length=hop_length,
                win_length=resolution,
                bin_spacing=self.bin_spacing,
            )

            metadata = {
                "subject": subject_id,
                "gender": info.gender,
                "health": info.health,
                "age": info.age,
                "fs": self.fs,
            }
            metadata["caption"] = self.caption(info.gender, info.health, info.age)
            metadata["parkinsons_condition"] = _encode_condition(metadata)

            relative = Path(subject_id) / f"spectrogram-{counter}.png"
            counter += 1
            yield SpectrogramExample(
                subject_id=subject_id,
                relative_path=relative,
                metadata=metadata,
                image=image,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def caption(self, gender: str, health: str, age: int) -> str:
        gender_text = "female" if str(gender).upper().startswith("F") else "male"
        health_text = "parkinsons" if str(health).upper().startswith("P") else "healthy"
        return f"an EEG spectrogram of a {age} year old, {health_text}, {gender_text} subject"

    def _derive_hop_length(self, resolution: int) -> int:
        max_bins = pymath.floor(resolution / self.n_channels)
        hop_length = 4
        while self.nsamps / hop_length > max_bins:
            hop_length += 4
        return hop_length

    def _load_participants(self) -> pd.DataFrame:
        participants_path = self.data_dir / "participants.tsv"
        if not participants_path.exists():
            raise FileNotFoundError(f"Missing participants.tsv at {participants_path}")
        table = pd.read_csv(participants_path, sep="\t")
        table.columns = [col.strip().upper() for col in table.columns]
        return table

    def _subject_metadata(self, subject_id: str) -> ParkinsonsSubjectInfo:
        sub_num = int(subject_id.split("-")[1]) - 1
        row = self.participants.iloc[sub_num]
        gender = str(row.get("GENDER", "M"))
        health = str(row.get("GROUP", "HC"))
        age = int(row.get("AGE", 0))
        return ParkinsonsSubjectInfo(subject=subject_id, gender=gender, health=health, age=age)

    def _load_subject_data(self, subject_id: str) -> np.ndarray | None:
        eeg_dir = self.data_dir / subject_id / "eeg"
        set_files = glob.glob(str(eeg_dir / "*eeg.set"))
        if not set_files:
            return None
        raw = mne.io.read_raw_eeglab(set_files[0], preload=True, verbose="ERROR")
        data = raw.get_data(picks=self.channel_indices)
        if self.decimation > 1:
            data = decimate(data, self.decimation, axis=1, zero_phase=True)
        return data


class ParkinsonsDataset:
    """Torch-compatible dataset for Parkinsons spectrograms."""

    def __init__(
        self,
        settings: Settings,
        *,
        split: str = "train",
        tasks: Sequence[str] = ("gender",),
        transform=None,
        target_format: str = "dict",
    ) -> None:
        self.settings = settings
        self.dataset_settings: DatasetSettings = settings.dataset("parkinsons")
        self.split = split
        self.transform = transform or transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        self.tasks = tuple(tasks)
        for name in self.tasks:
            if name not in PARKINSONS_LABELS:
                raise KeyError(f"Unknown task '{name}'. Available: {sorted(PARKINSONS_LABELS)}")
        if target_format not in {"dict", "tuple"}:
            raise ValueError("target_format must be 'dict' or 'tuple'")
        self.target_format = target_format

        metadata_path = self.dataset_settings.output / f"{split}-metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
        self.metadata = pd.read_csv(metadata_path)
        self.root = self.dataset_settings.output

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]
        image_path = self.root / row["file_name"]
        image = Image.open(image_path).convert("L")
        image_tensor = self.transform(image) if self.transform else image

        targets = {name: PARKINSONS_LABELS.encode(name, row) for name in self.tasks}
        sample = {
            "image": image_tensor,
            "targets": targets,
            "metadata": row.to_dict(),
        }
        if self.target_format == "tuple":
            if len(self.tasks) == 1:
                return image_tensor, targets[self.tasks[0]]
            return image_tensor, targets
        return sample

    @property
    def available_tasks(self) -> Sequence[str]:
        return tuple(PARKINSONS_LABELS.keys())

