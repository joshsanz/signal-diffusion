"""Longitudinal EEG dataset preprocessing and dataset utilities."""
from __future__ import annotations

import math as pymath
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import mne
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.signal import decimate
from torchvision.transforms import v2 as transforms

from signal_diffusion.config import DatasetSettings, Settings
from signal_diffusion.data.base import BaseSpectrogramPreprocessor, SpectrogramExample
from signal_diffusion.data.channel_maps import longitudinal_channels
from signal_diffusion.data.specs import LabelRegistry, LabelSpec
from signal_diffusion.data.utils.multichannel_spectrograms import multichannel_spectrogram
from signal_diffusion.log_setup import get_logger

logger = get_logger(__name__)

mne.set_log_level("WARNING")

GENDER_LABELS = {0: "male", 1: "female"}
ACQUISITION_LABELS = {0: "pre", 1: "post"}


def _encode_gender(row: Mapping[str, object]) -> int:
    value = str(row["gender"]).strip().upper()
    return 1 if value in {"F", "FEMALE", "1", "TRUE"} else 0


def _encode_acquisition(row: Mapping[str, object]) -> int:
    value = str(row["acquisition"]).strip().lower()
    return 1 if value in {"post", "1", "true"} else 0


def _encode_age(row: Mapping[str, object]) -> float:
    value = row.get("age")
    if value is None:
        raise ValueError("Missing age value in metadata")
    try:
        age = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid age value {value!r}") from exc
    if age < 0:
        raise ValueError(f"Age must be non-negative, received {age}")
    return age


def _encode_handedness(row: Mapping[str, object]) -> int:
    value = str(row.get("handedness", "right")).strip().lower()
    return 1 if value in {"left", "l", "1", "true"} else 0


def _encode_health(row: Mapping[str, object]) -> int:
    """Longitudinal dataset participants are healthy (no Parkinson's)."""
    return 0


LONGITUDINAL_LABELS = LabelRegistry()


def _decode_gender(value: object) -> str:
    return GENDER_LABELS[int(value)]


def _decode_acquisition(value: object) -> str:
    return ACQUISITION_LABELS[int(value)]


def _decode_handedness(value: object) -> str:
    return "left" if int(value) == 1 else "right"


def _decode_health(value: object) -> str:
    return "healthy" if int(value) == 0 else "parkinsons"


LONGITUDINAL_LABELS.register(
    LabelSpec(
        name="gender",
        num_classes=2,
        encoder=_encode_gender,
        description="0: male, 1: female",
        decoder=_decode_gender,
    )
)
LONGITUDINAL_LABELS.register(
    LabelSpec(
        name="age",
        encoder=_encode_age,
        description="Participant age in years (adjusted for follow-up)",
        task_type="regression",
    )
)
LONGITUDINAL_LABELS.register(
    LabelSpec(
        name="acquisition",
        num_classes=2,
        encoder=_encode_acquisition,
        description="0: pre (before cognitive tasks), 1: post (after cognitive tasks)",
        decoder=_decode_acquisition,
    )
)
LONGITUDINAL_LABELS.register(
    LabelSpec(
        name="handedness",
        num_classes=2,
        encoder=_encode_handedness,
        description="0: right, 1: left",
        decoder=_decode_handedness,
    )
)
LONGITUDINAL_LABELS.register(
    LabelSpec(
        name="health",
        num_classes=2,
        encoder=_encode_health,
        description="0: healthy, 1: parkinsons",
        decoder=_decode_health,
    )
)


def _normalize_gender(value: object) -> str:
    text = str(value).strip().lower()
    return "F" if text in {"f", "female", "1", "true"} else "M"


@dataclass(slots=True)
class LongitudinalSubjectInfo:
    subject_id: str
    baseline_age: int
    gender: str
    handedness: str
    has_session1: bool
    has_session2: bool


class LongitudinalPreprocessor(BaseSpectrogramPreprocessor):
    """Preprocess Longitudinal EEG recordings into spectrogram datasets."""

    DEFAULT_RESOLUTION = 512

    def __init__(
        self,
        settings: Settings,
        *,
        nsamps: int,
        ovr_perc: float = 0.0,
        fs: float = 125,
        bin_spacing: str = "log",
    ) -> None:
        super().__init__(settings, dataset_name="longitudinal")
        self.nsamps = nsamps
        self.overlap_fraction = float(ovr_perc)
        self.noverlap = int(np.floor(nsamps * self.overlap_fraction))
        self.bin_spacing = bin_spacing

        self.data_dir = self.dataset_settings.root
        self.participants = self._load_participants()

        # Original sampling rate is 1000 Hz
        orig_fs = 1000
        if fs <= 0 or orig_fs % fs != 0:
            raise ValueError("fs must be a positive divisor of 1000 Hz for longitudinal dataset")
        self.target_fs = fs
        self.decimation = orig_fs // fs
        self.fs = orig_fs / self.decimation

        # Use the standard 20-channel selection
        self.channel_indices = [idx for _, idx in longitudinal_channels]
        self.n_channels = len(self.channel_indices)
        self._subject_ids: Sequence[str] | None = None

        logger.info(
            f"Initialized LongitudinalPreprocessor: {self.n_channels} channels, "
            f"{orig_fs}Hz → {self.fs}Hz (decimation={self.decimation})"
        )

    # ------------------------------------------------------------------
    # BaseSpectrogramPreprocessor hooks
    # ------------------------------------------------------------------
    def subjects(self) -> Sequence[str]:
        if self._subject_ids is None:
            subs = [
                entry.name
                for entry in self.data_dir.iterdir()
                if entry.is_dir() and entry.name.startswith("sub-")
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

        # Determine which sessions to process
        sessions_to_process = []
        if info.has_session1:
            sessions_to_process.append("ses-1")
        if info.has_session2:
            sessions_to_process.append("ses-2")

        counter = 0
        for session in sessions_to_process:
            # Get adjusted age for this session
            age = self._get_adjusted_age(subject_id, session, info.baseline_age)

            # Get recording year for metadata
            recording_year = self._get_recording_year(subject_id, session)

            # Process both pre and post acquisitions for EyesOpen task
            for acquisition in ["pre", "post"]:
                data = self._load_subject_data(subject_id, session, "EyesOpen", acquisition)
                if data is None:
                    logger.debug(
                        f"Skipping {subject_id}/{session}/EyesOpen/{acquisition} - file not found"
                    )
                    continue

                # Segment into windows
                total_samples = data.shape[1]
                shift = self.nsamps - self.noverlap
                if shift <= 0:
                    raise ValueError("Overlap percentage results in non-positive shift size")

                if self.noverlap != 0:
                    nblocks = pymath.floor((total_samples - self.nsamps) / shift) + 1
                else:
                    nblocks = pymath.floor(total_samples / self.nsamps)

                if nblocks <= 0:
                    logger.warning(
                        f"Recording {subject_id}/{session}/{acquisition} too short "
                        f"for nsamps={self.nsamps}"
                    )
                    continue

                start = 0
                end = self.nsamps
                for _ in range(nblocks):
                    blk = data[:, start:end]
                    start += shift
                    end += shift

                    # Generate spectrogram
                    image = multichannel_spectrogram(
                        blk,
                        resolution=resolution,
                        hop_length=hop_length,
                        win_length=resolution,
                        bin_spacing=self.bin_spacing,
                    )

                    # Prepare metadata
                    metadata = {
                        "gender": info.gender,
                        "age": age,
                        "acquisition": acquisition,
                        "handedness": info.handedness,
                        "health": "H",  # All longitudinal participants are healthy
                        "session": session,
                        "recording_year": recording_year,
                    }

                    relative = Path(subject_id) / f"spectrogram-{counter}.png"
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
    def caption(self, gender: str, age: int, acquisition: str) -> str:
        gender_text = "female" if str(gender).upper().startswith("F") else "male"
        acq_text = f"{acquisition} cognitive tasks"
        return f"an EEG spectrogram of a {age} year old {gender_text} subject, {acq_text}"

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
        table.columns = [col.strip() for col in table.columns]
        return table

    def _subject_metadata(self, subject_id: str) -> LongitudinalSubjectInfo:
        row = self.participants[self.participants["participant_id"] == subject_id]
        if row.empty:
            raise ValueError(f"Subject {subject_id} not found in participants.tsv")

        row = row.iloc[0]
        gender_code = _normalize_gender(row.get("sex", "M"))
        handedness = row.get("handedness", "right")
        age = int(row.get("age", 0))
        has_ses1 = str(row.get("session1", "no")).lower() == "yes"
        has_ses2 = str(row.get("session2", "no")).lower() == "yes"

        return LongitudinalSubjectInfo(
            subject_id=subject_id,
            baseline_age=age,
            gender=gender_code,
            handedness=handedness,
            has_session1=has_ses1,
            has_session2=has_ses2,
        )

    def _get_recording_year(self, subject_id: str, session: str) -> int | None:
        sessions_file = self.data_dir / subject_id / f"{subject_id}_sessions.tsv"
        if not sessions_file.exists():
            return None

        sessions_df = pd.read_csv(sessions_file, sep="\t")
        year_row = sessions_df[sessions_df["session_id"] == session]
        if year_row.empty:
            return None

        return int(year_row.iloc[0]["recording_year"])

    def _get_adjusted_age(self, subject_id: str, session: str, baseline_age: int) -> int:
        """Calculate adjusted age for a given session."""
        if session == "ses-1":
            return baseline_age

        # Get recording years for both sessions
        year_ses1 = self._get_recording_year(subject_id, "ses-1")
        year_ses2 = self._get_recording_year(subject_id, "ses-2")

        if year_ses1 and year_ses2:
            years_diff = year_ses2 - year_ses1
            return baseline_age + years_diff

        # Fallback: assume 5 years
        logger.warning(
            f"Could not determine exact age for {subject_id}/{session}, assuming 5-year gap"
        )
        return baseline_age + 5

    def _load_subject_data(
        self, subject_id: str, session: str, task: str, acquisition: str
    ) -> np.ndarray | None:
        """Load EDF data for a specific subject, session, task, and acquisition."""
        eeg_dir = self.data_dir / subject_id / session / "eeg"
        # Pattern: sub-XXX_ses-Y_task-EyesOpen_acq-pre_eeg.edf
        edf_pattern = f"{subject_id}_{session}_task-{task}_acq-{acquisition}_eeg.edf"
        edf_path = eeg_dir / edf_pattern

        if not edf_path.exists():
            return None

        logger.debug(f"Loading data from {edf_path}")
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

        # Extract the 20 standard channels in the correct order
        data = raw.get_data(picks=self.channel_indices)

        # Decimate if needed
        if self.decimation > 1:
            data = decimate(data, self.decimation, axis=1, zero_phase=True)

        return data


class LongitudinalDataset:
    """Torch-compatible dataset for Longitudinal spectrograms."""

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
        self.dataset_settings: DatasetSettings = settings.dataset("longitudinal")
        self.split = split
        self.transform = transform or transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.tasks = tuple(tasks)
        for name in self.tasks:
            if name not in LONGITUDINAL_LABELS:
                raise KeyError(f"Unknown task '{name}'. Available: {sorted(LONGITUDINAL_LABELS)}")
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

        targets = {name: LONGITUDINAL_LABELS.encode(name, row) for name in self.tasks}
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
        return tuple(LONGITUDINAL_LABELS.keys())
