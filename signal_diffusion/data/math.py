"""Math EEG dataset preprocessing and dataset utilities."""
from __future__ import annotations

import logging
import json
import math as pymath
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import coloredlogs
import mne
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.signal import decimate
from torchvision.transforms import v2 as transforms

from signal_diffusion.data.utils.multichannel_spectrograms import multichannel_spectrogram
from signal_diffusion.config import DatasetSettings, Settings
from signal_diffusion.data.base import BaseSpectrogramPreprocessor, SpectrogramExample
from signal_diffusion.data.specs import LabelRegistry, LabelSpec

# Basic configuration for logging
log_level = "DEBUG" if os.environ.get("DEBUG") and os.environ.get("DEBUG") != "0" else "INFO"
coloredlogs.install(level=log_level)

logger = logging.getLogger(__name__)


mne.set_log_level("WARNING")

GENDER_NAMES = {0: "male", 1: "female"}
MATH_ACTIVITY_NAMES = {0: "background", 1: "doing_math"}
MATH_CONDITION_CLASSES = {
    0: "bkgnd_male",
    1: "bkgnd_female",
    2: "math_male",
    3: "math_female",
}


def _encode_gender(row: Mapping[str, object]) -> int:
    value = str(row["gender"]).strip().upper()
    return 1 if value in {"F", "FEMALE", "1", "TRUE"} else 0


def _encode_math_activity(row: Mapping[str, object]) -> int:
    value = row["doingmath"]
    if isinstance(value, str):
        value = value.strip().lower() in {"1", "true", "yes"}
    return int(bool(value))


def _encode_condition(row: Mapping[str, object]) -> int:
    gender = _encode_gender(row)
    math_activity = _encode_math_activity(row)
    return math_activity * 2 + gender


MATH_LABELS = LabelRegistry()


def _decode_gender(value: object) -> str:
    return GENDER_NAMES[int(value)]


def _decode_math_activity(value: object) -> str:
    return MATH_ACTIVITY_NAMES[int(value)]


def _decode_math_condition(value: object) -> str:
    return MATH_CONDITION_CLASSES[int(value)]


MATH_LABELS.register(
    LabelSpec(
        name="gender",
        num_classes=2,
        encoder=_encode_gender,
        description="0: male, 1: female",
        decoder=_decode_gender,
    )
)
MATH_LABELS.register(
    LabelSpec(
        name="math_activity",
        num_classes=2,
        encoder=_encode_math_activity,
        description="0: background, 1: doing math",
        decoder=_decode_math_activity,
    )
)
MATH_LABELS.register(
    LabelSpec(
        name="math_condition",
        num_classes=4,
        encoder=_encode_condition,
        description="Combined math/gender classes",
        decoder=_decode_math_condition,
    )
)


def _encode_health(row: Mapping[str, object]) -> int:
    """Math dataset participants are healthy (no Parkinson's)."""
    return 0


def _decode_health(value: object) -> str:
    return "healthy" if int(value) == 0 else "parkinsons"


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


MATH_LABELS.register(
    LabelSpec(
        name="health",
        num_classes=2,
        encoder=_encode_health,
        description="0: healthy, 1: parkinsons",
        decoder=_decode_health,
    )
)
MATH_LABELS.register(
    LabelSpec(
        name="age",
        encoder=_encode_age,
        description="Participant age in years",
        task_type="regression",
    )
)


def _normalize_gender(value: object) -> str:
    text = str(value).strip().lower()
    return "F" if text in {"f", "female", "1", "true"} else "M"


@dataclass(slots=True)
class MathSubjectInfo:
    subject_id: str
    age: int
    gender: str


class MathPreprocessor(BaseSpectrogramPreprocessor):
    """Preprocess Math EEG recordings into spectrogram datasets."""

    DEFAULT_RESOLUTION = 512

    def __init__(
        self,
        settings: Settings,
        *,
        nsamps: int,
        ovr_perc: float = 0.0,
        fs: float = 250,
        bin_spacing: str = "log",
        include_math_trials: bool = False,
    ) -> None:
        super().__init__(settings, dataset_name="math")
        self.nsamps = nsamps
        self.overlap_fraction = float(ovr_perc)
        self.noverlap = int(np.floor(nsamps * self.overlap_fraction))
        self.bin_spacing = bin_spacing
        self.include_math_trials = include_math_trials

        self.eeg_dir = self.dataset_settings.root / "raw_eeg"
        self.subject_table = self._load_subject_table()

        self.orig_fs = 500
        if fs <= 0:
            raise ValueError("fs must be positive for Math dataset")
        decimation_ratio = self.orig_fs / fs
        if not decimation_ratio.is_integer():
            raise ValueError("fs must be a positive divisor of 500 Hz for Math dataset")
        self.target_fs = fs
        self.decimation = int(decimation_ratio)
        self.fs = self.orig_fs / float(self.decimation)

        self.states: Sequence[int] = (1, 2) if include_math_trials else (1,)
        self._subject_ids: Sequence[str] | None = None
        self.channel_indices = self._determine_channel_indices()
        self.n_channels = len(self.channel_indices)

    # ------------------------------------------------------------------
    # BaseSpectrogramPreprocessor hooks
    # ------------------------------------------------------------------
    def subjects(self) -> Sequence[str]:
        if self._subject_ids is None:
            existing: set[str] = set()
            for edf_path in self.eeg_dir.glob("Subject*_*.edf"):
                subject = edf_path.stem.split("_")[0]
                if all((self.eeg_dir / f"{subject}_{state}.edf").exists() for state in self.states):
                    existing.add(subject)
            self._subject_ids = tuple(sorted(existing))
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
        noise_floor_db = self.dataset_settings.min_db if self.dataset_settings.min_db is not None else -130.0

        subject_info = self._subject_metadata(subject_id)
        gender_code = subject_info.gender
        base_folder = Path(f"sub{subject_id[-2:]}")
        counter = 0

        for state in self.states:
            data = self._load_subject_state(subject_id, state)
            if data is None:
                continue
            total_samples = data.shape[1]
            shift = self.nsamps - self.noverlap
            if shift <= 0:
                raise ValueError("Overlap percentage results in non-positive shift size")
            if self.noverlap != 0:
                nblocks = pymath.floor((total_samples - self.nsamps) / shift) + 1
            else:
                nblocks = pymath.floor(total_samples / self.nsamps)
            if nblocks <= 0:
                raise ValueError(
                    f"Recording {subject_id}_{state} too short for nsamps={self.nsamps}"
                )

            start = 0
            end = self.nsamps
            for _ in range(nblocks):
                blk = data[:, start:end]
                start += shift
                end += shift

                image = multichannel_spectrogram(
                    blk,
                    resolution=resolution,
                    hop_length=hop_length,
                    win_length=resolution,
                    bin_spacing=self.bin_spacing,
                    noise_floor_db=noise_floor_db,
                    output_type=self.settings.output_type,
                    min_db=self.dataset_settings.min_db,
                    max_db=self.dataset_settings.max_db,
                )

                doing_math = int(state == 2)

                metadata = {
                    "state": state,
                    "gender": gender_code,
                    "doingmath": doing_math,
                    "age": subject_info.age,
                }
                metadata["math_condition"] = _encode_condition(metadata)

                relative = base_folder / f"spectrogram-{counter}.png"
                counter += 1
                yield SpectrogramExample(
                    subject_id=subject_id,
                    relative_path=relative,
                    metadata=metadata,
                    image=image,
                )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def caption(self, gender: str, doing_math: int, age: int) -> str:
        gender_text = "female" if str(gender).upper().startswith("F") else "male"
        activity = "doing math" if doing_math else "background"
        return f"an EEG spectrogram of a {age} year old, {activity}, {gender_text} subject"

    def _derive_hop_length(self, resolution: int) -> int:
        max_bins = pymath.floor(resolution / self.n_channels)
        hop_length = 4
        while self.nsamps / hop_length > max_bins:
            hop_length += 4
        return hop_length

    def _load_subject_table(self) -> pd.DataFrame:
        info_path = self.dataset_settings.root / "raw_eeg" / "subject-info.csv"
        logger.debug(f"Loading subject info from {info_path}")
        if not info_path.exists():
            raise FileNotFoundError(f"Missing subject info CSV at {info_path}")
        table = pd.read_csv(info_path)
        table.columns = [col.strip() for col in table.columns]
        return table

    def _subject_metadata(self, subject_id: str) -> MathSubjectInfo:
        idx = int(subject_id[-2:]) - 1
        row = self.subject_table.iloc[idx]
        age = self._extract_value(row, ("Age", "age"), default_index=1)
        gender = self._extract_value(row, ("Gender", "gender", "Sex"), default_index=2)
        gender_code = _normalize_gender(gender)
        return MathSubjectInfo(
            subject_id=subject_id,
            age=int(age),
            gender=gender_code,
        )

    def _extract_value(self, row: pd.Series, candidates: Sequence[str], default_index: int) -> object:
        for name in candidates:
            if name in row.index:
                return row[name]
        return row.iloc[default_index]

    def _determine_channel_indices(self) -> Sequence[int]:
        first_subject = next(iter(self.subjects()))
        sample_path = self.eeg_dir / f"{first_subject}_{self.states[0]}.edf"
        raw = mne.io.read_raw_edf(sample_path, preload=False, verbose="ERROR")
        ch_names = raw.ch_names
        indices = [i for i, name in enumerate(ch_names) if name.upper() != "EKG"]
        if not indices:
            indices = list(range(len(ch_names)))
        return tuple(indices)

    def _load_subject_state(self, subject_id: str, state: int) -> np.ndarray | None:
        edf_path = self.eeg_dir / f"{subject_id}_{state}.edf"
        logger.debug(f"Loading data from {edf_path}")
        if not edf_path.exists():
            return None
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
        data = raw.get_data(picks=self.channel_indices)
        if self.decimation > 1:
            data = decimate(data, self.decimation, axis=1, zero_phase=True)
        return data


class MathTimeSeriesPreprocessor(MathPreprocessor):
    """Preprocess Math EEG recordings into time-domain .npy datasets."""

    DEFAULT_RESOLUTION = 512

    def __init__(
        self,
        settings: Settings,
        *,
        nsamps: int,
        ovr_perc: float = 0.0,
        fs: float = 250,
        include_math_trials: bool = False,
    ) -> None:
        super().__init__(
            settings,
            nsamps=nsamps,
            ovr_perc=ovr_perc,
            fs=fs,
            include_math_trials=include_math_trials,
        )
        self.output_dir = self.dataset_settings.timeseries_output or (
            self.dataset_settings.output.parent / "timeseries"
        )
        self.norm_stats = self._load_or_compute_normalization_stats()

    # ------------------------------------------------------------------
    # BaseSpectrogramPreprocessor hooks
    # ------------------------------------------------------------------
    def subjects(self) -> Sequence[str]:
        if self._subject_ids is None:
            existing: set[str] = set()
            for edf_path in self.eeg_dir.glob("Subject*_*.edf"):
                subject = edf_path.stem.split("_")[0]
                if all((self.eeg_dir / f"{subject}_{state}.edf").exists() for state in self.states):
                    existing.add(subject)
            self._subject_ids = tuple(sorted(existing))
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
        if resolution != self.nsamps:
            logger.warning(
                f"Configured resolution ({resolution}) does not match preprocessing nsamps ({self.nsamps}). "
                f"Using nsamps={self.nsamps} for time-series data."
            )

        subject_info = self._subject_metadata(subject_id)
        gender_code = subject_info.gender
        means = np.asarray(self.norm_stats["channel_means"], dtype=np.float32)[:, None]
        stds = np.asarray(self.norm_stats["channel_stds"], dtype=np.float32)[:, None]

        for state in self.states:
            data = self._load_subject_state(subject_id, state)
            if data is None:
                continue
            total_samples = data.shape[1]
            shift = self.nsamps - self.noverlap
            if shift <= 0:
                raise ValueError("Overlap percentage results in non-positive shift size")
            nblocks = int(pymath.floor((total_samples - self.nsamps) / shift)) + 1
            if nblocks <= 0:
                raise ValueError(
                    f"Recording {subject_id}_{state} too short for nsamps={self.nsamps}"
                )

            base_folder = Path(subject_id) / f"state-{state}"
            for block_idx in range(nblocks):
                block_start = block_idx * shift
                block_end = block_start + self.nsamps
                blk = data[:, block_start:block_end]

                normalized_block = (blk.astype(np.float32) - means) / (stds + 1e-8)

                doing_math = int(state == 2)

                metadata = {
                    "state": state,
                    "gender": gender_code,
                    "doingmath": doing_math,
                    "age": subject_info.age,
                }
                metadata["math_condition"] = _encode_condition(metadata)

                relative = base_folder / f"timeseries-{block_idx}.npy"
                yield SpectrogramExample(
                    subject_id=subject_id,
                    relative_path=relative,
                    metadata=metadata,
                    writer=lambda path, data=normalized_block.copy(): np.save(path, data),
                )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    def _load_or_compute_normalization_stats(self) -> dict:
        """Load existing normalization stats or compute from all data."""
        stats_path = self.output_dir / "math_normalization_stats.json"
        if stats_path.exists():
            logger.info(f"Loading normalization statistics from {stats_path}")
            with stats_path.open("r") as handle:
                return json.load(handle)

        logger.info("Computing normalization statistics from all Math data...")
        return self._compute_normalization_stats(stats_path)

    def _compute_normalization_stats(self, stats_path: Path) -> dict:
        """Compute per-channel mean and std from all subjects and states."""
        n_samples = np.zeros(self.n_channels, dtype=np.int64)
        means = np.zeros(self.n_channels, dtype=np.float64)
        m2s = np.zeros(self.n_channels, dtype=np.float64)

        for subject_id in self.subjects():
            for state in self.states:
                data = self._load_subject_state(subject_id, state)
                if data is None:
                    continue
                for ch_idx in range(self.n_channels):
                    channel_data = data[ch_idx, :]
                    for value in channel_data:
                        n_samples[ch_idx] += 1
                        delta = value - means[ch_idx]
                        means[ch_idx] += delta / n_samples[ch_idx]
                        delta2 = value - means[ch_idx]
                        m2s[ch_idx] += delta * delta2

        stds = np.sqrt(m2s / n_samples)
        stats = {
            "channel_means": means.tolist(),
            "channel_stds": stds.tolist(),
            "n_samples_per_channel": n_samples.tolist(),
        }

        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w") as handle:
            json.dump(stats, handle, indent=2)

        logger.info(f"Saved normalization statistics to {stats_path}")
        return stats


class MathDataset:
    """Torch-compatible dataset for Math spectrograms."""

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
        self.dataset_settings: DatasetSettings = settings.dataset("math")
        self.split = split
        # Use appropriate normalization based on output_type
        if transform is not None:
            self.transform = transform
        elif settings.output_type == "db-only":
            self.transform = transforms.Compose(
                [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True), transforms.Normalize([0.5], [0.5])]
            )
        else:  # db-iq or db-polar (3 channels)
            self.transform = transforms.Compose(
                [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
            )
        self.tasks = tuple(tasks)
        for name in self.tasks:
            if name not in MATH_LABELS:
                raise KeyError(f"Unknown task '{name}'. Available: {sorted(MATH_LABELS)}")
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
        # Convert to appropriate mode based on output_type
        mode = "L" if self.settings.output_type == "db-only" else "RGB"
        image = Image.open(image_path).convert(mode)
        image_tensor = self.transform(image) if self.transform else image

        targets = {name: MATH_LABELS.encode(name, row) for name in self.tasks}
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
        return tuple(MATH_LABELS.keys())


class MathTimeSeriesDataset(torch.utils.data.Dataset):
    """Torch-compatible dataset for Math time-domain windows."""

    def __init__(
        self,
        settings: Settings,
        *,
        split: str = "train",
        tasks: Sequence[str] = ("gender",),
        transform=None,
        target_format: str = "dict",
        expected_length: int | None = None,
    ) -> None:
        self.settings = settings
        self.dataset_settings: DatasetSettings = settings.dataset("math")
        self.split = split
        self.transform = transform
        self.tasks = tuple(tasks)
        self.expected_length = expected_length
        for name in self.tasks:
            if name not in MATH_LABELS:
                raise KeyError(f"Unknown task '{name}'. Available: {sorted(MATH_LABELS)}")
        if target_format not in {"dict", "tuple"}:
            raise ValueError("target_format must be 'dict' or 'tuple'")
        self.target_format = target_format

        self.root = self._resolve_root()
        metadata_path = self.root / f"{split}-metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
        self.metadata = pd.read_csv(metadata_path)

        if self.expected_length is not None:
            self._validate_signal_length()

    def _resolve_root(self) -> Path:
        if self.dataset_settings.timeseries_output is not None:
            return self.dataset_settings.timeseries_output
        if self.dataset_settings.output.name == "stfts":
            return self.dataset_settings.output.parent / "timeseries"
        return self.dataset_settings.output

    def _validate_signal_length(self) -> None:
        """Warn if configured length does not match stored signals."""
        if self.metadata.empty:
            return
        first = self.root / self.metadata.iloc[0]["file_name"]
        try:
            sample = np.load(first)
        except FileNotFoundError:
            logger.warning("Cannot validate signal length; missing sample at %s", first)
            return
        actual_length = sample.shape[1]
        if actual_length != self.expected_length:
            logger.warning(
                "Expected signal length %s but found %s in %s",
                self.expected_length,
                actual_length,
                first.name,
            )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]
        data_path = self.root / row["file_name"]
        signal = torch.from_numpy(np.load(data_path)).float()

        if self.transform:
            signal = self.transform(signal)

        targets = {name: MATH_LABELS.encode(name, row) for name in self.tasks}
        sample = {
            "signal": signal,
            "targets": targets,
            "metadata": row.to_dict(),
        }
        if self.target_format == "tuple":
            if len(self.tasks) == 1:
                return signal, targets[self.tasks[0]]
            return signal, targets
        return sample

    @property
    def available_tasks(self) -> Sequence[str]:
        return tuple(MATH_LABELS.keys())
