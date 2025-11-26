"""Dataset and dataloader builders for diffusion training."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping

import numpy as np
import torch
from datasets import (
    Dataset, DatasetDict, IterableDataset, IterableDatasetDict,
    load_dataset
)
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from transformers import PreTrainedTokenizerBase

from signal_diffusion.config import Settings, load_settings

from .config import DatasetConfig


@dataclass(slots=True)
class DiffusionBatch:
    """Structured batch returned by diffusion dataloaders."""

    pixel_values: torch.Tensor
    captions: torch.Tensor | None = None
    class_labels: torch.Tensor | None = None


class DiffusionCollator:
    """Collate function converting preprocessed examples to tensors."""

    def __call__(self, examples: list[Mapping[str, Any]]) -> DiffusionBatch:
        if "signal" in examples[0]:
            signals = [torch.as_tensor(example["signal"]) for example in examples]
            pixel_values = torch.stack(signals).unsqueeze(1)
        elif "pixel_values" in examples[0]:
            pixel_values = torch.stack([torch.as_tensor(example["pixel_values"]) for example in examples])
        elif "image" in examples[0]:
            pixel_values = torch.stack([torch.as_tensor(example["image"]) for example in examples])
        else:
            raise KeyError("Examples must contain 'signal', 'pixel_values', or 'image' key")
        captions = None
        class_labels = None
        if "captions" in examples[0]:
            captions = torch.stack([torch.as_tensor(example["captions"]) for example in examples])
        if "class_labels" in examples[0]:
            class_labels = torch.as_tensor([example["class_labels"] for example in examples])
        return DiffusionBatch(pixel_values=pixel_values, captions=captions, class_labels=class_labels)


def _maybe_load_settings(cfg_path: Path | None) -> Settings | None:
    if cfg_path is None:
        return None
    return load_settings(cfg_path)


def _resolve_dataset(cfg: DatasetConfig, settings: Settings | None) -> tuple[str | None, Path | None]:
    """Return (huggingface_dataset_id, local_path)."""
    if settings is not None and cfg.identifier in settings.datasets:
        ds_settings = settings.dataset(cfg.identifier)
        return None, ds_settings.root

    if cfg.dataset_type == "imagefolder" or Path(cfg.identifier).exists():
        return None, Path(cfg.identifier)

    return cfg.identifier, None


def _build_transforms(cfg: DatasetConfig, *, train: bool, data_type: str = "spectrogram") -> transforms.Compose | None:
    if data_type == "timeseries":
        return _build_timeseries_transforms(cfg, train=train)
    return _build_image_transforms(cfg, train=train)


class _ConvertToTensor(torch.nn.Module):
    """Convert numpy arrays to tensors."""

    def forward(self, x: Any) -> torch.Tensor:
        """Convert numpy array to tensor or pass through tensor."""
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.as_tensor(x, dtype=torch.float32)


class _GaussianNoiseTransform(torch.nn.Module):
    """Add gaussian noise to tensor input."""

    def __init__(self, noise_std: float):
        super().__init__()
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        return x + torch.randn_like(x) * self.noise_std


def _build_image_transforms(cfg: DatasetConfig, *, train: bool) -> transforms.Compose:
    size = cfg.resolution
    ops: list[Any] = [transforms.ToImage()]
    ops.append(transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR))
    if cfg.center_crop:
        ops.append(transforms.CenterCrop(size))
    else:
        if train:
            ops.append(transforms.RandomCrop(size))
        else:
            ops.append(transforms.CenterCrop(size))
    if train and cfg.random_flip:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend(
        [
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return transforms.Compose(ops)


def _build_timeseries_transforms(cfg: DatasetConfig, *, train: bool) -> transforms.Compose | None:
    ops: list[Any] = [_ConvertToTensor()]

    if train:
        noise_std = 0.0
        if isinstance(cfg.extras, Mapping):
            noise_std = float(cfg.extras.get("gaussian_noise_std", 0.0) or 0.0)
        if noise_std > 0:
            ops.append(_GaussianNoiseTransform(noise_std))

    return transforms.Compose(ops)


def _tokenize_captions(
    tokenizer: PreTrainedTokenizerBase,
    captions: Iterable[Any],
    *,
    is_train: bool,
) -> torch.Tensor:
    processed: list[str] = []
    for caption in captions:
        if isinstance(caption, str):
            processed.append(caption)
        elif isinstance(caption, (list, tuple, np.ndarray)) and caption:
            if is_train:
                idx = np.random.randint(len(caption))
                processed.append(str(caption[idx]))
            else:
                processed.append(str(caption[0]))
        else:
            raise ValueError("Caption column must contain strings or non-empty sequences of strings")
    tokenized = tokenizer(
        processed,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return tokenized.input_ids


def _preprocess(
    examples: Mapping[str, Any],
    *,
    transform: transforms.Compose | None,
    tokenizer: PreTrainedTokenizerBase | None,
    caption_column: str | None,
    class_column: str | None,
    is_train: bool,
    data_type: str = "spectrogram",
) -> Mapping[str, Any]:
    if data_type == "timeseries":
        signals = examples.get("signal")
        if signals is None:
            raise KeyError("Time-series dataset must provide a 'signal' column")

        if transform:
            processed_data = [transform(signal) for signal in signals]
        else:
            processed_data = [
                torch.from_numpy(signal).float() if isinstance(signal, np.ndarray) else signal for signal in signals
            ]
        batch: MutableMapping[str, Any] = {"signal": processed_data}
    else:
        images = examples.get("image") or examples.get("pixel_values")
        if images is None:
            raise KeyError("Dataset must provide an 'image' column")

        processed_images = [transform(image.convert("RGB")) for image in images]
        batch = {"pixel_values": processed_images}

    if tokenizer is not None and caption_column:
        captions = examples.get(caption_column)
        if captions is None:
            raise KeyError(f"Dataset missing caption column '{caption_column}'")
        tokens = _tokenize_captions(tokenizer, captions, is_train=is_train)
        batch["captions"] = [token for token in tokens]

    if class_column:
        labels = examples.get(class_column)
        if labels is None:
            raise KeyError(f"Dataset missing class column '{class_column}'")
        batch["class_labels"] = [int(label) for label in labels]

    return batch


def _prepare_dataset(
    dataset: Dataset,
    cfg: DatasetConfig,
    *,
    tokenizer: PreTrainedTokenizerBase | None,
    caption_column: str | None,
    class_column: str | None,
    is_train: bool,
    data_type: str = "spectrogram",
) -> Dataset:
    transform = _build_transforms(cfg, train=is_train, data_type=data_type)
    preprocess = partial(
        _preprocess,
        transform=transform,
        tokenizer=tokenizer,
        caption_column=caption_column,
        class_column=class_column,
        is_train=is_train,
        data_type=data_type,
    )
    return dataset.with_transform(preprocess)


def build_dataloaders(
    cfg: DatasetConfig,
    *,
    tokenizer: PreTrainedTokenizerBase | None,
    settings_path: Path | None,
    data_type: str = "spectrogram",
) -> tuple[DataLoader, DataLoader | None]:
    """Construct training (and optional validation) dataloaders."""

    settings = _maybe_load_settings(settings_path)
    if settings and hasattr(settings, "data_type"):
        data_type = settings.data_type
    hf_dataset, local_path = _resolve_dataset(cfg, settings)

    ds: Dataset | DatasetDict | IterableDataset | IterableDatasetDict
    if hf_dataset is not None:
        cache_dir = None
        if cfg.cache_dir:
            cache_dir = str(cache_dir)
        ds = load_dataset(hf_dataset, cache_dir=cache_dir, split=None)
    else:
        if local_path is None:
            raise FileNotFoundError(
                f"Unable to resolve dataset path for identifier '{cfg.identifier}'"
            )
        ds = load_dataset(path=str(local_path))

    if isinstance(ds, DatasetDict):
        train_dataset = ds[cfg.train_split]
        val_dataset = ds[cfg.val_split] if cfg.val_split else None
    elif isinstance(ds, Dataset):
        train_dataset = ds
        val_dataset = None
    else:  # IterableDataset is unsupported for now
        raise TypeError("Iterable datasets are not currently supported for diffusion training")

    if cfg.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(cfg.max_train_samples, len(train_dataset))))
    if cfg.max_eval_samples is not None and val_dataset is not None:
        val_dataset = val_dataset.select(range(min(cfg.max_eval_samples, len(val_dataset))))

    train_dataset = _prepare_dataset(
        train_dataset,
        cfg,
        tokenizer=tokenizer,
        caption_column=cfg.caption_column,
        class_column=cfg.class_column,
        is_train=True,
        data_type=data_type,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=DiffusionCollator(),
    )

    val_loader: DataLoader | None = None
    if val_dataset is not None:
        val_dataset = _prepare_dataset(
            val_dataset,
            cfg,
            tokenizer=tokenizer,
            caption_column=cfg.caption_column,
            class_column=cfg.class_column,
            is_train=False,
            data_type=data_type,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size * 4,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=DiffusionCollator(),
        )

    return train_loader, val_loader
