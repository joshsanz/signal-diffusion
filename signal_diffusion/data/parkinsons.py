"""Parkinsons EEG dataset preprocessing and dataset utilities."""
from __future__ import annotations


from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import glob
import math as pymath

import mne
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.signal import decimate
from torchvision.transforms import v2 as transforms

from signal_diffusion.data.utils.multichannel_spectrograms import multichannel_spectrogram
from signal_diffusion.data.channel_maps import parkinsons_channels

from signal_diffusion.config import DatasetSettings, Settings
from signal_diffusion.data.base import BaseSpectrogramPreprocessor, SpectrogramExample
from signal_diffusion.data.specs import LabelRegistry, LabelSpec
from signal_diffusion.log_setup import get_logger

logger = get_logger(__name__)


mne.set_log_level("WARNING")

HEALTH_LABELS = {0: "healthy", 1: "parkinsons"}
GENDER_LABELS = {0: "male", 1: "female"}


def _encode_gender(row: Mapping[str, object]) -> int:
    value = str(row["gender"]).strip().upper()
    return 1 if value in {"F", "FEMALE", "1", "TRUE"} else 0


def _encode_health(row: Mapping[str, object]) -> int:
    value = str(row["health"]).strip().upper()
    return 1 if value in {"PD", "PARKINSONS", "1", "TRUE"} else 0


def _encode_condition(row: Mapping[str, object]) -> int:
    gender = _encode_gender(row)
    health = _encode_health(row)
    return health * 2 + gender


def _encode_age(row: Mapping[str, object]) -> float:
    value = row.get("age", row.get("AGE"))
    if value is None:
        raise ValueError("Missing age value in metadata")
    try:
        age = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid age value {value!r}") from exc
    if age < 0:
        raise ValueError(f"Age must be non-negative, received {age}")
    return age


PARKINSONS_CONDITION_CLASSES = {
    0: "healthy_male",
    1: "healthy_female",
    2: "parkinsons_male",
    3: "parkinsons_female",
}

# Backwards-compatible aliases for notebook imports.
health_class_labels = HEALTH_LABELS


PARKINSONS_LABELS = LabelRegistry()


def _decode_gender(value: object) -> str:
    return GENDER_LABELS[int(value)]


def _decode_health(value: object) -> str:
    return HEALTH_LABELS[int(value)]


def _decode_condition(value: object) -> str:
    return PARKINSONS_CONDITION_CLASSES[int(value)]


PARKINSONS_LABELS.register(
    LabelSpec(
        name="gender",
        num_classes=2,
        encoder=_encode_gender,
        description="0: male, 1: female",
        decoder=_decode_gender,
    )
)
PARKINSONS_LABELS.register(
    LabelSpec(
        name="health",
        num_classes=2,
        encoder=_encode_health,
        description="0: healthy, 1: parkinsons",
        decoder=_decode_health,
    )
)
PARKINSONS_LABELS.register(
    LabelSpec(
        name="parkinsons_condition",
        num_classes=4,
        encoder=_encode_condition,
        description="Combined health/gender label",
        decoder=_decode_condition,
    )
)
PARKINSONS_LABELS.register(
    LabelSpec(
        name="age",
        encoder=_encode_age,
        description="Participant age in years",
        task_type="regression",
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
        fs: float = 250,
        bin_spacing: str = "log",
    ) -> None:
        super().__init__(settings, dataset_name="parkinsons")
        self.nsamps = nsamps
        self.overlap_fraction = float(ovr_perc)
        self.noverlap = int(np.floor(nsamps * self.overlap_fraction))
        self.bin_spacing = bin_spacing

        self.data_dir = self.dataset_settings.root
        self.participants = self._load_participants()

        orig_fs = 500
        if fs <= 0:
            raise ValueError("fs must be positive for Parkinsons dataset")
        decimation_ratio = orig_fs / fs
        if not decimation_ratio.is_integer():
            raise ValueError("fs must be a positive divisor of 500 Hz")
        self.target_fs = fs
        self.decimation = int(decimation_ratio)
        self.fs = orig_fs / float(self.decimation)

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
        noise_floor_db = self.dataset_settings.min_db if self.dataset_settings.min_db is not None else -130.0

        info = self._subject_metadata(subject_id)
        data = self._load_subject_data(subject_id)
        if data is None:
            return

        total_samples = data.shape[1]
        gender_code = info.gender
        health_code = info.health
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
                noise_floor_db=noise_floor_db,
                output_type=self.settings.output_type,
                min_db=self.dataset_settings.min_db,
                max_db=self.dataset_settings.max_db,
            )

            metadata = {
                "gender": gender_code,
                "health": health_code,
                "age": info.age,
            }
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
        logger.debug(f"Loading participants from {participants_path}")
        if not participants_path.exists():
            raise FileNotFoundError(f"Missing participants.tsv at {participants_path}")
        table = pd.read_csv(participants_path, sep="\t")
        table.columns = [col.strip().upper() for col in table.columns]
        return table

    def _subject_metadata(self, subject_id: str) -> ParkinsonsSubjectInfo:
        sub_num = int(subject_id.split("-")[1]) - 1
        row = self.participants.iloc[sub_num]
        gender_code = _normalize_gender(row.get("GENDER", "M"))
        health_code = _normalize_health(row.get("GROUP", "HC"))
        age = int(row.get("AGE", 0))
        return ParkinsonsSubjectInfo(
            subject=subject_id,
            gender=gender_code,
            health=health_code,
            age=age,
        )

    def _load_subject_data(self, subject_id: str) -> np.ndarray | None:
        eeg_dir = self.data_dir / subject_id / "eeg"
        set_files = glob.glob(str(eeg_dir / "*eeg.set"))
        if not set_files:
            return None
        logger.debug(f"Loading data from {set_files[0]}")
        raw = mne.io.read_raw_eeglab(set_files[0], preload=True, verbose="ERROR")
        data = raw.get_data(picks=self.channel_indices)
        if self.decimation > 1:
            data = decimate(data, self.decimation, axis=1, zero_phase=True)
        return data


class ParkinsonsTimeSeriesPreprocessor(ParkinsonsPreprocessor):
    """Preprocess Parkinsons EEG recordings into time-domain .npy datasets."""

    DEFAULT_RESOLUTION = 512

    def __init__(
        self,
        settings: Settings,
        *,
        nsamps: int,
        ovr_perc: float = 0.0,
        fs: float = 250,
    ) -> None:
        super().__init__(settings, nsamps=nsamps, ovr_perc=ovr_perc, fs=fs)
        self.output_dir = self.dataset_settings.timeseries_output or (
            self.dataset_settings.output.parent / "timeseries"
        )
        self.norm_stats = self._load_or_compute_normalization_stats()

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
        if resolution != self.nsamps:
            logger.warning(
                f"Configured resolution ({resolution}) does not match preprocessing nsamps ({self.nsamps}). "
                f"Using nsamps={self.nsamps} for time-series data."
            )

        info = self._subject_metadata(subject_id)
        data = self._load_subject_data(subject_id)
        if data is None:
            return

        total_samples = data.shape[1]
        shift = self.nsamps - self.noverlap
        if shift <= 0:
            raise ValueError("Overlap percentage results in non-positive shift size")

        nblocks = int(pymath.floor((total_samples - self.nsamps) / shift)) + 1
        if nblocks <= 0:
            raise ValueError(f"Recording {subject_id} too short for nsamps={self.nsamps}")

        means = np.asarray(self.norm_stats["channel_means"], dtype=np.float32)[:, None]
        stds = np.asarray(self.norm_stats["channel_stds"], dtype=np.float32)[:, None]

        start = 0
        end = self.nsamps
        counter = 0
        for _ in range(nblocks):
            blk = data[:, start:end]
            start += shift
            end += shift

            normalized_block = (blk.astype(np.float32) - means) / (stds + 1e-8)

            metadata = {
                "gender": info.gender,
                "health": info.health,
                "age": info.age,
            }
            metadata["parkinsons_condition"] = _encode_condition(metadata)

            relative = Path(subject_id) / f"timeseries-{counter}.npy"
            counter += 1
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
        stats_path = self.output_dir / "parkinsons_normalization_stats.json"
        if stats_path.exists():
            logger.info(f"Loading normalization statistics from {stats_path}")
            with stats_path.open("r") as handle:
                return json.load(handle)

        logger.info("Computing normalization statistics from all Parkinsons data...")
        return self._compute_normalization_stats(stats_path)

    def _compute_normalization_stats(self, stats_path: Path) -> dict:
        """Compute per-channel mean and std from all subjects."""
        n_samples = np.zeros(self.n_channels, dtype=np.int64)
        means = np.zeros(self.n_channels, dtype=np.float64)
        m2s = np.zeros(self.n_channels, dtype=np.float64)

        for subject_id in self.subjects():
            data = self._load_subject_data(subject_id)
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
        total_samples = int(n_samples.sum())

        stats = {
            "channel_means": means.tolist(),
            "channel_stds": stds.tolist(),
            "n_eeg_channels": self.n_channels,
            "n_samples_total": total_samples,
        }

        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w") as handle:
            json.dump(stats, handle, indent=2)

        logger.info(f"Saved normalization statistics to {stats_path}")
        logger.info(f"  Channels: {self.n_channels}, Total samples: {total_samples:,}")
        return stats


class ParkinsonsDataset:
    """Torch-compatible dataset for Parkinsons spectrograms."""

    def __init__(
        self,
        settings: Settings,
        *,
        split: str = "train",
        tasks: Sequence[str] = ("health",),
        transform=None,
        target_format: str = "dict",
    ) -> None:
        self.settings = settings
        self.dataset_settings: DatasetSettings = settings.dataset("parkinsons")
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
        # Convert to appropriate mode based on output_type
        mode = "L" if self.settings.output_type == "db-only" else "RGB"
        image = Image.open(image_path).convert(mode)
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


class ParkinsonsTimeSeriesDataset(torch.utils.data.Dataset):
    """Torch-compatible dataset for Parkinsons time-domain windows."""

    def __init__(
        self,
        settings: Settings,
        *,
        split: str = "train",
        tasks: Sequence[str] = ("health",),
        transform=None,
        target_format: str = "dict",
        expected_length: int | None = None,
    ) -> None:
        self.settings = settings
        self.dataset_settings: DatasetSettings = settings.dataset("parkinsons")
        self.split = split
        self.transform = transform
        self.tasks = tuple(tasks)
        self.expected_length = expected_length
        for name in self.tasks:
            if name not in PARKINSONS_LABELS:
                raise KeyError(f"Unknown task '{name}'. Available: {sorted(PARKINSONS_LABELS)}")
        if target_format not in {"dict", "tuple"}:
            raise ValueError("target_format must be 'dict' or 'tuple'")
        self.target_format = target_format

        self.root = self._resolve_root()
        metadata_path = self.root / f"{split}-metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
        self.metadata = pd.read_csv(metadata_path)

        # Load normalization stats to get n_eeg_channels
        stats_path = self.dataset_settings.output / "parkinsons_normalization_stats.json"
        if stats_path.exists():
            with stats_path.open() as f:
                norm_stats = json.load(f)
            self.n_eeg_channels = norm_stats.get("n_eeg_channels")
        else:
            # Fallback: infer from first sample if stats don't exist
            if not self.metadata.empty:
                first_file = self.root / self.metadata.iloc[0]["file_name"]
                try:
                    sample = np.load(first_file)
                    self.n_eeg_channels = sample.shape[0]
                except FileNotFoundError:
                    logger.warning(f"Cannot infer n_eeg_channels; missing sample at {first_file}")
                    self.n_eeg_channels = None
            else:
                self.n_eeg_channels = None

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

        targets = {name: PARKINSONS_LABELS.encode(name, row) for name in self.tasks}
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
        return tuple(PARKINSONS_LABELS.keys())


def _load_metadata_frame(datadir: str | Path, split: str) -> pd.DataFrame:
    base = Path(datadir)
    metadata_path = base / f"{split}-metadata.csv"
    if not metadata_path.exists():
        metadata_path = base / "stfts" / f"{split}-metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file for split '{split}': {metadata_path}")
    return pd.read_csv(metadata_path)


class HealthSampler(torch.utils.data.Sampler[int]):
    """Weighted sampler balancing the Parkinsons health label."""

    def __init__(
        self,
        datadir: str | Path,
        num_samples: int,
        split: str = "train",
        replacement: bool = True,
        generator=None,
    ) -> None:
        self.metadata = _load_metadata_frame(datadir, split)
        self.weights = torch.as_tensor(self._generate_weights(self.metadata), dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __len__(self) -> int:
        return len(self.metadata)

    def __iter__(self):
        draws = self.num_samples if self.replacement else len(self.metadata)
        rand_tensor = torch.multinomial(self.weights, draws, self.replacement, generator=self.generator)
        yield from rand_tensor.tolist()

    @staticmethod
    def _generate_weights(metadata: pd.DataFrame) -> list[float]:
        labels = metadata.apply(_encode_health, axis=1).tolist()
        total = len(labels)
        if total == 0:
            return []
        counts = [labels.count(i) for i in range(2)]
        if sum(counts) == 0:
            return [1 / total] * total
        # Larger weight for minority class
        norm = sum(counts)
        label_weights = [count / norm for count in counts]
        rankings = {idx: sum(weight < label_weights[idx] for weight in label_weights) for idx in range(2)}
        ordered = sorted(label_weights, reverse=True)
        remapped = [ordered[rankings[idx]] for idx in range(2)]
        output = [remapped[label] for label in labels]
        normaliser = sum(output)
        return [weight / normaliser for weight in output]


class ParkinsonsSampler(torch.utils.data.Sampler[int]):
    """Weighted sampler balancing combined health/gender condition labels."""

    def __init__(
        self,
        datadir: str | Path,
        num_samples: int,
        split: str = "train",
        replacement: bool = True,
        generator=None,
    ) -> None:
        self.metadata = _load_metadata_frame(datadir, split)
        self.weights = torch.as_tensor(self._generate_weights(self.metadata), dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __len__(self) -> int:
        return len(self.metadata)

    def __iter__(self):
        draws = self.num_samples if self.replacement else len(self.metadata)
        rand_tensor = torch.multinomial(self.weights, draws, self.replacement, generator=self.generator)
        yield from rand_tensor.tolist()

    @staticmethod
    def _generate_weights(metadata: pd.DataFrame) -> list[float]:
        labels = metadata.apply(_encode_condition, axis=1).tolist()
        total = len(labels)
        if total == 0:
            return []
        counts = [labels.count(i) for i in range(4)]
        if sum(counts) == 0:
            return [1 / total] * total
        norm = sum(counts)
        label_weights = [count / norm for count in counts]
        rankings = {idx: sum(weight < label_weights[idx] for weight in label_weights) for idx in range(4)}
        ordered = sorted(label_weights, reverse=True)
        remapped = [ordered[rankings[idx]] for idx in range(4)]
        output = [remapped[label] for label in labels]
        normaliser = sum(output)
        return [weight / normaliser for weight in output]
def _normalize_gender(value: object) -> str:
    text = str(value).strip().lower()
    return "F" if text in {"f", "female", "1", "true"} else "M"


def _normalize_health(value: object) -> str:
    text = str(value).strip().upper()
    return "PD" if text in {"PD", "PARKINSONS", "1", "TRUE"} else "H"
