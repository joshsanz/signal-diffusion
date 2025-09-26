"""SEED EEG dataset preprocessing and dataset utilities."""
from __future__ import annotations


from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import math as pymath

import mne
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.signal import decimate
from torchvision import transforms

from common.multichannel_spectrograms import multichannel_spectrogram
from signal_diffusion.data.channel_maps import seed_channels

from signal_diffusion.config import DatasetSettings, Settings
from signal_diffusion.data.base import BaseSpectrogramPreprocessor, SpectrogramExample
from signal_diffusion.data.specs import LabelRegistry, LabelSpec
from signal_diffusion.log_setup import get_logger

logger = get_logger(__name__)

mne.set_log_level("WARNING")

EMOTION_MAP = {
    "disgust": 0,
    "fear": 1,
    "sad": 2,
    "neutral": 3,
    "happy": 4,
}

EMOTION_NAMES = {v: k for k, v in EMOTION_MAP.items()}
GENDER_NAMES = {0: "male", 1: "female"}
SEED_CONDITION_CLASSES = {
    0: "disgust_male",
    1: "disgust_female",
    2: "fear_male",
    3: "fear_female",
    4: "sad_male",
    5: "sad_female",
    6: "neutral_male",
    7: "neutral_female",
    8: "happy_male",
    9: "happy_female",
}

TRIAL_EMOTIONS = {
    0: [4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0],
    1: [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
    2: [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
}

START_SECOND = {
    0: [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
    1: [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
    2: [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888],
}

END_SECOND = {
    0: [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359],
    1: [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817],
    2: [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066],
}

emotion_class_labels = {idx: EMOTION_NAMES[idx] for idx in sorted(EMOTION_NAMES)}


def _encode_gender(row: Mapping[str, object]) -> int:
    value = str(row["gender"]).strip().upper()
    return 1 if value in {"F", "FEMALE", "1", "TRUE"} else 0


def _encode_emotion(row: Mapping[str, object]) -> int:
    emotion = row["emotion_id"] if "emotion_id" in row else row.get("emotion")
    if isinstance(emotion, str):
        emotion = EMOTION_MAP[emotion]
    return int(emotion)


def _encode_condition(row: Mapping[str, object]) -> int:
    gender = _encode_gender(row)
    emotion = _encode_emotion(row)
    return emotion * 2 + gender


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


def _encode_health(row: Mapping[str, object]) -> int:
    value = str(row.get("health", "H")).strip().upper()
    return 0 if value in {"H", "HEALTHY", "0", "FALSE"} else 1


SEED_LABELS = LabelRegistry()


def _decode_gender(value: object) -> str:
    return GENDER_NAMES[int(value)]


def _decode_health(value: object) -> str:
    return "healthy" if int(value) == 0 else "ill"


def _decode_emotion(value: object) -> str:
    return EMOTION_NAMES[int(value)]


def _decode_condition(value: object) -> str:
    return SEED_CONDITION_CLASSES[int(value)]


SEED_LABELS.register(
    LabelSpec(
        name="gender",
        num_classes=2,
        encoder=_encode_gender,
        description="0: male, 1: female",
        decoder=_decode_gender,
    )
)
SEED_LABELS.register(
    LabelSpec(
        name="health",
        num_classes=2,
        encoder=_encode_health,
        description="0: healthy, 1: ill",
        decoder=_decode_health,
    )
)
SEED_LABELS.register(
    LabelSpec(
        name="emotion",
        num_classes=5,
        encoder=_encode_emotion,
        description="Emotion class index",
        decoder=_decode_emotion,
    )
)
SEED_LABELS.register(
    LabelSpec(
        name="seed_condition",
        num_classes=10,
        encoder=_encode_condition,
        description="Combined emotion/gender class",
        decoder=_decode_condition,
    )
)
SEED_LABELS.register(
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
class SeedSubjectInfo:
    subject_id: str
    index: int
    gender: str
    age: int


class SEEDPreprocessor(BaseSpectrogramPreprocessor):
    """Preprocess SEED EEG recordings into spectrogram datasets."""

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
        super().__init__(settings, dataset_name="seed")
        self.nsamps = nsamps
        self.overlap_fraction = float(ovr_perc)
        self.noverlap = int(np.floor(nsamps * self.overlap_fraction))
        self.bin_spacing = bin_spacing

        self.data_dir = self.dataset_settings.root
        self.raw_dir = self.data_dir / "EEG_raw"
        self.participants = self._load_participants()

        self.orig_fs = 1000
        if fs <= 0 or self.orig_fs % fs != 0:
            raise ValueError("fs must be a positive divisor of 1000 Hz")
        self.target_fs = fs
        self.decimation = self.orig_fs // fs
        self.fs = self.orig_fs / self.decimation

        self.channel_indices = [idx for _, idx in seed_channels]
        self.n_channels = len(self.channel_indices)
        self.n_sessions = 3
        self.n_trials = len(START_SECOND[0])

        self.session_files = self._discover_sessions()
        self._subject_ids = tuple(sorted(self.session_files.keys()))

    # ------------------------------------------------------------------
    # BaseSpectrogramPreprocessor hooks
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
        for session_index, cnt_path in self.session_files.get(subject_id, []):
            logger.debug(f"Loading data from {cnt_path}")
            raw = mne.io.read_raw_cnt(cnt_path, preload=True, verbose="WARNING")
            data = raw.get_data()[self.channel_indices, :]
            if self.decimation > 1:
                data = decimate(data, self.decimation, axis=1, zero_phase=True)

            start_points = START_SECOND[session_index]
            end_points = END_SECOND[session_index]
            emotions = TRIAL_EMOTIONS[session_index]

            for trial_index in range(self.n_trials):
                start = int(start_points[trial_index] * self.fs)
                end = int(end_points[trial_index] * self.fs)
                trial_data = data[:, start:end]

                shift = self.nsamps - self.noverlap
                if shift <= 0:
                    raise ValueError("Overlap percentage results in non-positive shift size")
                if self.noverlap != 0:
                    nblocks = pymath.floor((trial_data.shape[1] - self.nsamps) / shift) + 1
                else:
                    nblocks = pymath.floor(trial_data.shape[1] / self.nsamps)
                if nblocks <= 0:
                    continue

                block_start = 0
                block_end = self.nsamps
                block_idx = 0
                while block_end <= trial_data.shape[1]:
                    block = trial_data[:, block_start:block_end]
                    block_start += shift
                    block_end += shift

                    image = multichannel_spectrogram(
                        block,
                        resolution=resolution,
                        hop_length=hop_length,
                        win_length=resolution,
                        bin_spacing=self.bin_spacing,
                    )

                    emotion_id = emotions[trial_index]
                    metadata = {
                        "session": session_index,
                        "trial": trial_index,
                        "gender": info.gender,
                        "health": "H",
                        "age": info.age,
                        "emotion_id": emotion_id,
                        "emotion": emotion_id,
                        "emotion_label": EMOTION_NAMES[emotion_id],
                    }
                    metadata["seed_condition"] = _encode_condition(metadata)

                    relative = Path(subject_id) / f"spectrogram-s{session_index+1}-t{trial_index}-{block_idx}.png"
                    block_idx += 1
                    yield SpectrogramExample(
                        subject_id=subject_id,
                        relative_path=relative,
                        metadata=metadata,
                        image=image,
                    )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def caption(self, age: int, gender: str, emotion_id: int) -> str:
        gender_text = "female" if str(gender).upper().startswith("F") else "male"
        emotion_text = EMOTION_NAMES[emotion_id]
        return f"an EEG spectrogram of a {age} year old, {gender_text} subject feeling {emotion_text}"

    def _derive_hop_length(self, resolution: int) -> int:
        max_bins = pymath.floor(resolution / self.n_channels)
        hop_length = 8
        while self.nsamps / hop_length > max_bins:
            hop_length += 8
        return hop_length

    def _load_participants(self) -> pd.DataFrame:
        info_path = self.data_dir / "participants_info.csv"
        logger.debug(f"Loading participants from {info_path}")
        if not info_path.exists():
            raise FileNotFoundError(f"Missing participants_info.csv at {info_path}")
        table = pd.read_csv(info_path)
        table.columns = [col.strip().capitalize() for col in table.columns]
        return table

    def _subject_metadata(self, subject_id: str) -> SeedSubjectInfo:
        index = int(subject_id.split("-")[1]) - 1
        row = self.participants.iloc[index]
        age = int(row.get("Age", 0))
        gender_code = _normalize_gender(row.get("Sex", "M"))
        return SeedSubjectInfo(
            subject_id=subject_id,
            index=index,
            gender=gender_code,
            age=age,
        )

    def _discover_sessions(self) -> dict[str, list[tuple[int, Path]]]:
        sessions: dict[str, list[tuple[int, Path]]] = {}
        for cnt_path in sorted(self.raw_dir.glob("*.cnt")):
            stem = cnt_path.stem
            parts = stem.split("_")
            if len(parts) < 2:
                continue
            subject_num = int(parts[0])
            session_num = int(parts[1])
            subject_id = f"sub-{subject_num:02d}"
            sessions.setdefault(subject_id, []).append((session_num - 1, cnt_path))
        return sessions


class SEEDDataset:
    """Torch-compatible dataset for SEED spectrograms."""

    def __init__(
        self,
        settings: Settings,
        *,
        split: str = "train",
        tasks: Sequence[str] = ("emotion",),
        transform=None,
        target_format: str = "dict",
    ) -> None:
        self.settings = settings
        self.dataset_settings: DatasetSettings = settings.dataset("seed")
        self.split = split
        self.transform = transform or transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        self.tasks = tuple(tasks)
        for name in self.tasks:
            if name not in SEED_LABELS:
                raise KeyError(f"Unknown task '{name}'. Available: {sorted(SEED_LABELS)}")
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

        targets = {name: SEED_LABELS.encode(name, row) for name in self.tasks}
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
        return tuple(SEED_LABELS.keys())


def _load_metadata_frame(datadir: str | Path, split: str) -> pd.DataFrame:
    base = Path(datadir)
    metadata_path = base / f"{split}-metadata.csv"
    if not metadata_path.exists():
        metadata_path = base / "stfts" / f"{split}-metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file for split '{split}': {metadata_path}")
    return pd.read_csv(metadata_path)


class EmotionSampler(torch.utils.data.Sampler[int]):
    """Weighted sampler balancing SEED gender distribution.

    Retains the historical class name for notebook compatibility while the
    weighting strategy now targets the gender label, matching the broader
    data layer convention.
    """

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
        labels = metadata.apply(_encode_gender, axis=1).tolist()
        total = len(labels)
        if total == 0:
            return []
        counts = [labels.count(i) for i in range(2)]
        if sum(counts) == 0:
            return [1 / total] * total
        norm = sum(counts)
        label_weights = [count / norm for count in counts]
        rankings = {idx: sum(weight < label_weights[idx] for weight in label_weights) for idx in range(2)}
        ordered = sorted(label_weights, reverse=True)
        remapped = [ordered[rankings[idx]] for idx in range(2)]
        output = [remapped[label] for label in labels]
        normaliser = sum(output)
        return [weight / normaliser for weight in output]
