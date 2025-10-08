"""Dataset helpers for classifier training."""
from __future__ import annotations

from typing import Mapping, Sequence

from torchvision.transforms import v2 as transforms

from signal_diffusion.config import Settings
from signal_diffusion.data import (
    MathDataset,
    MITDataset,
    ParkinsonsDataset,
    SEEDDataset,
)

_DATASET_CLS: Mapping[str, type] = {
    "math": MathDataset,
    "parkinsons": ParkinsonsDataset,
    "seed": SEEDDataset,
    "mit": MITDataset,
}

_DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def default_transform():
    """Return the default spectrogram transform used for classification."""
    return _DEFAULT_TRANSFORM


def build_dataset(
    settings: Settings,
    dataset_name: str,
    *,
    split: str,
    tasks: Sequence[str],
    transform=None,
    target_format: str = "dict",
):
    """Construct a dataset instance compatible with the shared data layer."""

    try:
        dataset_cls = _DATASET_CLS[dataset_name]
    except KeyError as exc:  # pragma: no cover - validated upstream
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. Available: {sorted(_DATASET_CLS)}"
        ) from exc

    return dataset_cls(
        settings=settings,
        split=split,
        tasks=tuple(tasks),
        transform=transform or default_transform(),
        target_format=target_format,
    )

