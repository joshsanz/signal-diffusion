"""Dataset helpers for classifier training."""
from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import torch
from datasets import load_dataset as hf_load_dataset, Features, Image, Value, ClassLabel
from torchvision.transforms import v2 as transforms

from signal_diffusion.config import Settings
from signal_diffusion.data import (
    LongitudinalTimeSeriesDataset,
    MathDataset,
    MathTimeSeriesDataset,
    ParkinsonsDataset,
    ParkinsonsTimeSeriesDataset,
    SEEDDataset,
    SEEDTimeSeriesDataset,
)
from signal_diffusion.data.meta import META_LABELS

_DATASET_CLS: Mapping[str, type] = {
    "math": MathDataset,
    "parkinsons": ParkinsonsDataset,
    "seed": SEEDDataset,
    "math_timeseries": MathTimeSeriesDataset,
    "parkinsons_timeseries": ParkinsonsTimeSeriesDataset,
    "seed_timeseries": SEEDTimeSeriesDataset,
    "longitudinal_timeseries": LongitudinalTimeSeriesDataset,
}

def default_transform(output_type: str = "db-only"):
    """Return the default spectrogram transform used for classification.

    Parameters:
    - output_type: "db-only" for 1-channel, "db-iq" or "db-polar" for 3-channel
    """
    if output_type == "db-only":
        return transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    else:  # db-iq or db-polar (3 channels)
        return transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )


def _preprocess_hf_dataset(
    examples: Mapping[str, Any],
    *,
    transform: transforms.Compose,
    tasks: Sequence[str],
    output_type: str = "db-only",
) -> Mapping[str, Any]:
    """Preprocess examples from HuggingFace dataset."""
    images = examples.get("image") or examples.get("pixel_values")
    if images is None:
        raise KeyError("Dataset must provide an 'image' column")

    # Convert to appropriate mode based on output_type
    mode = "L" if output_type == "db-only" else "RGB"
    processed_images = [transform(image.convert(mode)) for image in images]

    # Build targets dict from metadata columns
    batch: MutableMapping[str, Any] = {"image": processed_images}

    # Extract task labels from the examples
    targets_list = []
    for i in range(len(processed_images)):
        row_dict = {key: examples[key][i] for key in examples.keys()}
        targets = {task: META_LABELS.encode(task, row_dict) for task in tasks}
        targets_list.append(targets)

    batch["targets"] = targets_list
    # Exclude 'image' and 'pixel_values' from metadata to avoid PIL image issues in collate
    metadata_keys = [key for key in examples.keys() if key not in ("image", "pixel_values")]
    batch["metadata"] = [{key: examples[key][i] for key in metadata_keys} for i in range(len(processed_images))]

    return batch


def build_dataset(
    settings: Settings,
    dataset_name: str,
    *,
    split: str,
    tasks: Sequence[str],
    transform=None,
    target_format: str = "dict",
    extras: dict[str, Any] | None = None,
):
    """Construct a dataset instance compatible with the shared data layer.

    For time-series datasets, extras dict is used bidirectionally:
    - Input: Can contain 'expected_length' to validate signal length
    - Output: Populated with 'n_eeg_channels' and 'sequence_length' from dataset
    """

    # Check if dataset_name is a path (for reweighted meta datasets)
    dataset_path = Path(dataset_name).expanduser()
    if dataset_path.exists() or "/" in dataset_name or "~" in dataset_name:
        # Load from path using HuggingFace datasets
        ds = hf_load_dataset(str(dataset_path), split=split)

        # Apply preprocessing transform
        transform_fn = transform or default_transform(settings.output_type)
        preprocess = partial(
            _preprocess_hf_dataset,
            transform=transform_fn,
            tasks=tasks,
            output_type=settings.output_type,
        )
        return ds.with_transform(preprocess)

    # Otherwise, use named dataset classes
    try:
        dataset_cls = _DATASET_CLS[dataset_name]
    except KeyError as exc:  # pragma: no cover - validated upstream
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. Available: {sorted(_DATASET_CLS)}"
        ) from exc

    is_timeseries = dataset_name.endswith("_timeseries") or "timeseries" in dataset_cls.__name__.lower()
    resolved_transform = transform
    if resolved_transform is None and not is_timeseries:
        resolved_transform = default_transform(settings.output_type)

    # Prepare constructor kwargs
    kwargs = {
        "settings": settings,
        "split": split,
        "tasks": tuple(tasks),
        "transform": resolved_transform,
        "target_format": target_format,
    }

    # For time-series datasets, add expected_length from extras if provided
    if is_timeseries and extras:
        if "expected_length" in extras:
            kwargs["expected_length"] = extras["expected_length"]

    # Instantiate dataset
    dataset = dataset_cls(**kwargs)

    # For time-series datasets, populate extras with metadata
    if is_timeseries and extras is not None:
        if hasattr(dataset, 'n_eeg_channels') and dataset.n_eeg_channels is not None:
            extras["n_eeg_channels"] = dataset.n_eeg_channels
        # Also store sequence_length from expected_length if available
        if "sequence_length" not in extras and hasattr(dataset, 'expected_length'):
            if dataset.expected_length is not None:
                extras["sequence_length"] = dataset.expected_length

    return dataset
