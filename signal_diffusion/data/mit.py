"""CHB-MIT seizure dataset preprocessing and dataset utilities."""
from __future__ import annotations

import math as pymath
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import mne
import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import decimate
from torchvision import transforms

from common.multichannel_spectrograms import multichannel_spectrogram
from data_processing.channel_map import mit_channels

from signal_diffusion.config import DatasetSettings, Settings
from signal_diffusion.data.base import BaseSpectrogramPreprocessor, SpectrogramExample
from signal_diffusion.data.specs import LabelRegistry, LabelSpec

mne.set_log_level("WARNING")

MIT_CONDITION_CLASSES = {
    0: "baseline_male",
    1: "baseline_female",
    2: "seizure_male",
    3: "seizure_female",
}


@dataclass(slots=True)
class MITSubjectInfo:
    """Lookup information for a CHB-MIT subject."""

    subject_id: str
    index: int
    gender: str
    age: int | None = None


def _encode_gender(row: Mapping[str, object]) -> int:
    gender = str(row.get("gender", "M")).strip().upper()
    return 1 if gender.startswith("F") else 0


def _encode_seizure(row: Mapping[str, object]) -> int:
    return int(bool(row.get("seizure", 0)))


def _encode_condition(row: Mapping[str, object]) -> int:
    gender = _encode_gender(row)
    seizure = _encode_seizure(row)
    return seizure * 2 + gender


def _encode_health(_: Mapping[str, object]) -> int:
    return 0


MIT_LABELS = LabelRegistry()
MIT_LABELS.register(
    LabelSpec(
        name="gender",
        num_classes=2,
        encoder=_encode_gender,
        description="0: male, 1: female",
    )
)
MIT_LABELS.register(
    LabelSpec(
        name="health",
        num_classes=2,
        encoder=_encode_health,
        description="0: healthy, 1: ill",
    )
)
MIT_LABELS.register(
    LabelSpec(
        name="seizure",
        num_classes=2,
        encoder=_encode_seizure,
        description="0: non-seizure chunk, 1: seizure chunk",
    )
)
MIT_LABELS.register(
    LabelSpec(
        name="mit_condition",
        num_classes=4,
        encoder=_encode_condition,
        description="Combined gender/seizure class",
    )
)


class MITPreprocessor(BaseSpectrogramPreprocessor):
    """Preprocess CHB-MIT recordings into spectrogram datasets."""

    DEFAULT_RESOLUTION = 512

    def __init__(
        self,
        settings: Settings,
        *,
        nsamps: int,
        ovr_perc: float = 0.0,
        fs: float = 256,
        bin_spacing: str = "linear",
    ) -> None:
        super().__init__(settings, dataset_name="mit")
        self.nsamps = nsamps
        self.overlap_fraction = float(ovr_perc)
        self.noverlap = int(np.floor(nsamps * self.overlap_fraction))
        self.bin_spacing = bin_spacing

        self.dataset_root = self.dataset_settings.root
        self.files_dir = self.dataset_root / "files"
        if not self.files_dir.exists():
            raise FileNotFoundError(f"Expected CHB-MIT files directory at {self.files_dir}")

        self.subject_table = self._load_subject_table()
        self.channel_indices = [idx for _, idx in mit_channels]
        self.n_channels = len(self.channel_indices)

        self.orig_fs = 256
        if fs <= 0 or self.orig_fs % fs != 0:
            raise ValueError("fs must be a positive divisor of 256 Hz for MIT dataset")
        self.target_fs = fs
        self.decimation = self.orig_fs // fs
        self.fs = self.orig_fs / self.decimation

        self.recordings = self._discover_recordings()
        self._subject_ids = tuple(sorted(self.recordings))

    # ------------------------------------------------------------------
    # Base hooks
    # ------------------------------------------------------------------
    def subjects(self) -> Sequence[str]:
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

        for recording_path in self.recordings.get(subject_id, []):
            raw = mne.io.read_raw_edf(recording_path, preload=True, verbose="ERROR")
            data = raw.get_data()[self.channel_indices, :]
            if self.decimation > 1:
                data = decimate(data, self.decimation, axis=1, zero_phase=True)

            seizure_intervals = self._seizure_intervals(raw, recording_path)
            if self.decimation > 1 and seizure_intervals:
                seizure_intervals = [
                    (int(start // self.decimation), int(end // self.decimation)) for start, end in seizure_intervals
                ]

            total_samples = data.shape[1]
            shift = self.nsamps - self.noverlap
            if shift <= 0:
                raise ValueError("Overlap percentage results in non-positive shift size")
            if self.noverlap != 0:
                nblocks = pymath.floor((total_samples - self.nsamps) / shift) + 1
            else:
                nblocks = pymath.floor(total_samples / self.nsamps)
            if nblocks <= 0:
                continue

            start = 0
            end = self.nsamps
            block_index = 0
            while end <= total_samples:
                block = data[:, start:end]
                start += shift
                end += shift

                image = multichannel_spectrogram(
                    block,
                    resolution=resolution,
                    hop_length=hop_length,
                    win_length=resolution,
                    bin_spacing=self.bin_spacing,
                )

                chunk_start = start - shift
                chunk_end = chunk_start + self.nsamps
                active_seizure = self._chunk_has_seizure(chunk_start, chunk_end, seizure_intervals)

                metadata = {
                    "subject": subject_id,
                    "recording": recording_path.stem,
                    "gender": info.gender,
                    "health": "healthy",
                    "age": info.age if info.age is not None else "",
                    "seizure": int(active_seizure),
                    "fs": self.fs,
                }
                metadata["caption"] = self.caption(info, bool(active_seizure))
                metadata["mit_condition"] = _encode_condition(metadata)

                relative = Path(subject_id) / recording_path.stem / f"spectrogram-{block_index}.png"
                block_index += 1
                yield SpectrogramExample(
                    subject_id=subject_id,
                    relative_path=relative,
                    metadata=metadata,
                    image=image,
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def caption(self, info: MITSubjectInfo, active_seizure: bool) -> str:
        gender_text = "female" if info.gender.upper().startswith("F") else "male"
        seizure_text = "during a seizure" if active_seizure else "without seizure"
        if info.age is not None:
            return f"an EEG spectrogram of a {info.age} year old, {gender_text} subject {seizure_text}"
        return f"an EEG spectrogram of a {gender_text} subject {seizure_text}"

    def _derive_hop_length(self, resolution: int) -> int:
        max_bins = pymath.floor(resolution / self.n_channels)
        hop_length = 8
        while self.nsamps / hop_length > max_bins:
            hop_length += 8
        return hop_length

    def _load_subject_table(self) -> pd.DataFrame:
        info_path = self.files_dir / "SUBJECT-INFO.csv"
        if not info_path.exists():
            raise FileNotFoundError(f"Missing SUBJECT-INFO.csv at {info_path}")
        table = pd.read_csv(info_path)
        # Normalise column names for predictable access
        table.columns = [col.strip().lower() for col in table.columns]
        return table

    def _subject_metadata(self, subject_id: str) -> MITSubjectInfo:
        index = int(subject_id.replace("chb", "")) - 1
        if 0 <= index < len(self.subject_table):
            row = self.subject_table.iloc[index]
            gender = str(row.get("gender", "M")).upper()
            age_value = row.get("age")
        else:
            gender = "M"
            age_value = None
        try:
            age = int(age_value) if age_value is not None and not pd.isna(age_value) else None
        except (TypeError, ValueError):
            age = None
        return MITSubjectInfo(subject_id=subject_id, index=index, gender=gender, age=age)

    def _discover_recordings(self) -> dict[str, list[Path]]:
        recordings: dict[str, list[Path]] = {}
        for subject_dir in sorted(self.files_dir.glob("chb*/")):
            if not subject_dir.is_dir():
                continue
            subject_id = subject_dir.name
            edf_files = sorted(subject_dir.glob("*.edf"))
            if not edf_files:
                continue
            recordings[subject_id] = edf_files
        return recordings

    def _seizure_intervals(self, raw: mne.io.BaseRaw, recording_path: Path) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        annotations = getattr(raw, "annotations", None)
        if annotations is not None and len(annotations) > 0:
            for onset, duration, description in zip(annotations.onset, annotations.duration, annotations.description):
                if "seiz" in description.lower():
                    start_sample = int(round((onset - raw.first_time) * raw.info["sfreq"]))
                    end_sample = start_sample + int(round(duration * raw.info["sfreq"]))
                    spans.append((max(0, start_sample), max(start_sample, end_sample)))
        if spans:
            return spans

        try:
            import pyedflib
        except ImportError:
            return spans

        try:
            reader = pyedflib.EdfReader(str(recording_path))
        except Exception:  # pragma: no cover - dataset specific edge cases
            return spans

        try:
            onsets, durations, descriptions = reader.readAnnotations()
        except Exception:  # pragma: no cover - fall back when annotations missing
            reader.close()
            return spans

        sfreq = reader.getSampleFrequency(0)
        for onset, duration, description in zip(onsets, durations, descriptions):
            if isinstance(description, bytes):
                description = description.decode("utf-8", "ignore")
            if "seiz" not in str(description).lower():
                continue
            start_sample = int(round(onset * sfreq))
            end_sample = start_sample + int(round(duration * sfreq))
            spans.append((max(0, start_sample), max(start_sample, end_sample)))

        reader.close()
        return spans

    def _chunk_has_seizure(
        self,
        chunk_start: int,
        chunk_end: int,
        intervals: Sequence[tuple[int, int]],
    ) -> bool:
        if not intervals:
            return False
        min_overlap = max(1, int(self.nsamps * 0.1))
        for start, end in intervals:
            overlap_start = max(chunk_start, start)
            overlap_end = min(chunk_end, end)
            if overlap_end - overlap_start >= min_overlap:
                return True
        return False


class MITDataset:
    """Torch-compatible dataset for CHB-MIT spectrograms."""

    def __init__(
        self,
        settings: Settings,
        *,
        split: str = "train",
        tasks: Sequence[str] = ("seizure",),
        transform=None,
        target_format: str = "dict",
    ) -> None:
        self.settings = settings
        self.dataset_settings: DatasetSettings = settings.dataset("mit")
        self.split = split
        self.transform = transform or transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        self.tasks = tuple(tasks)
        for name in self.tasks:
            if name not in MIT_LABELS:
                raise KeyError(f"Unknown task '{name}'. Available: {sorted(MIT_LABELS)}")
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

        targets = {name: MIT_LABELS.encode(name, row) for name in self.tasks}
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
        return tuple(MIT_LABELS.keys())
