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
    MathDataset,
    ParkinsonsDataset,
    SEEDDataset,
)
from signal_diffusion.data.meta import META_LABELS

_DATASET_CLS: Mapping[str, type] = {
    "math": MathDataset,
    "parkinsons": ParkinsonsDataset,
    "seed": SEEDDataset,
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
):
    """Construct a dataset instance compatible with the shared data layer."""

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

    return dataset_cls(
        settings=settings,
        split=split,
        tasks=tuple(tasks),
        transform=transform or default_transform(settings.output_type),
        target_format=target_format,
    )
