"""Dataset helpers for metrics scripts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, cast

import numpy as np
from datasets import DatasetDict, Features, Image, IterableDatasetDict, load_dataset
from torch.utils.data import Dataset


DEFAULT_SPLIT = "train"


@dataclass(frozen=True)
class ImageFolderConfig:
    """Configuration describing how to load an imagefolder dataset."""

    data_dir: str
    split: str = DEFAULT_SPLIT
    image_mode: str = "RGB"


class _IndexableDataset(Protocol):
    def __len__(self) -> int:
        ...

    def __getitem__(self, idx: int) -> Any:
        ...


class RandomSubsetDataset(Dataset[Any]):
    """Wrap a dataset to return a random, non-repeating subset."""

    def __init__(
        self,
        dataset: _IndexableDataset,
        subset_size: int | None = None,
        *,
        seed: int | None = None,
        image_key: str = "image",
    ) -> None:
        self.dataset = dataset
        self.subset_size = subset_size
        self.seed = seed
        self.image_key = image_key

        if subset_size is not None and subset_size > 0:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(dataset), subset_size, replace=False)
            self._indices = indices.tolist()
        else:
            self._indices = None

    def __len__(self) -> int:
        if self._indices is not None:
            return len(self._indices)
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Any:
        if self._indices is not None:
            idx = self._indices[idx]
        example = self.dataset[idx]
        if self.image_key:
            return example[self.image_key]
        return example


def load_imagefolder_dataset(
    config: ImageFolderConfig,
    *,
    transform: Optional[Callable] = None,
) -> Dataset[Any]:
    """Load an imagefolder dataset with HuggingFace datasets."""

    dataset_dict = cast(
        DatasetDict | IterableDatasetDict,
        load_dataset(
            "imagefolder",
            data_dir=config.data_dir,
            features=Features({"image": Image(mode=config.image_mode)}),
        ).with_format("torch"),
    )

    if transform is not None:
        dataset_dict = dataset_dict.with_transform(transform)

    dataset = dataset_dict[config.split]
    return cast(Dataset[Any], dataset)
