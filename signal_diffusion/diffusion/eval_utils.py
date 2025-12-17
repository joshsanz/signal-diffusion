"""Evaluation helpers for diffusion training."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, TensorDataset
from torch_fidelity import calculate_metrics
from torchvision.utils import make_grid


def _to_uint8(images: torch.Tensor) -> torch.Tensor:
    images = images.detach().cpu().clamp(-1.0, 1.0)
    images = (images + 1.0) / 2.0
    return (images * 255.0).clamp(0, 255).to(torch.uint8)


def save_image_grid(images: torch.Tensor, output_path: Path, cols: int = 4) -> None:
    if images.ndim != 4:
        raise ValueError("Expected images tensor with shape (N, C, H, W)")
    grid = make_grid(images, nrow=cols, normalize=True, value_range=(-1, 1))
    array = _to_uint8(grid.unsqueeze(0))[0].permute(1, 2, 0).numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(output_path, format="JPEG")


class _WrappedDataset(Dataset):
    def __init__(self, dataset: Dataset, image_column: str | int = "image"):
        self._dataset = dataset
        self._image_column = image_column

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        item = self._dataset[idx]
        if isinstance(item, (tuple, list)):
            image = item[self._image_column]
        elif isinstance(item, dict):
            image = item[self._image_column]
        else:
            image = item

        # From [-1, 1] float to [0, 255] uint8
        image = (image.clamp(-1.0, 1.0) + 1.0) / 2.0
        image = (image * 255.0).clamp(0, 255).to(torch.uint8)
        return image


def compute_kid_score(
    generated_samples: torch.Tensor,
    ref_dataset: Dataset,
) -> Tuple[float, float]:
    generated_dataset = TensorDataset(generated_samples.cpu())

    # KID requires subset_size <= min(num_gen, num_ref)
    # Use the smaller of the two, but clamp to reasonable range
    num_gen = len(generated_dataset)
    num_ref = len(ref_dataset)
    kid_subset_size = min(num_gen, num_ref, 1000)

    metrics = calculate_metrics(
        input1=_WrappedDataset(generated_dataset, image_column=0),
        input2=_WrappedDataset(ref_dataset, image_column="pixel_values"),
        kid=True,
        kid_subset_size=kid_subset_size,
        feature_extractor="dinov2-vit-s-14",
        feature_extractor_internal_dtype="float32",
        verbose=False,
    )
    kid_mean = metrics["kernel_inception_distance_mean"]
    kid_std = metrics["kernel_inception_distance_std"]

    return kid_mean, kid_std
