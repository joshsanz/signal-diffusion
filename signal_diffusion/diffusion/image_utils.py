"""Image tensor conversion utilities for diffusion models."""

from __future__ import annotations

import torch
from PIL import Image
from torchvision import transforms

from .eval_utils import _to_uint8


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a tensor in [-1, 1] range to a PIL Image.

    Args:
        tensor: Image tensor shaped (C, H, W) in range [-1, 1]

    Returns:
        PIL Image in RGB format

    Raises:
        ValueError: If tensor is not 3D (C, H, W)
    """
    if tensor.ndim != 3:
        raise ValueError(f"Expected image tensor shaped (C, H, W), got {tensor.shape}")

    # Use _to_uint8 to convert [-1, 1] → uint8
    uint8 = _to_uint8(tensor.unsqueeze(0))[0].permute(1, 2, 0).numpy()
    return Image.fromarray(uint8)


def tensor_to_pil_simple(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a tensor in [-1, 1] range to a PIL Image (simple version).

    This is an alternative implementation that doesn't use _to_uint8.

    Args:
        tensor: Image tensor shaped (C, H, W) or (B, C, H, W) in range [-1, 1]

    Returns:
        PIL Image in RGB format
    """
    tensor = tensor.detach().cpu().clamp(-1, 1)
    tensor = (tensor + 1) / 2  # Scale to [0, 1]
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    return transforms.ToPILImage()(tensor)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL Image to a tensor in [-1, 1] range.

    Args:
        image: PIL Image in any mode

    Returns:
        Tensor shaped (1, C, H, W) in range [-1, 1]
    """
    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Convert to tensor in [0, 1]
    tensor = transforms.ToTensor()(image)

    # Scale to [-1, 1]
    tensor = tensor * 2 - 1

    # Add batch dimension
    return tensor.unsqueeze(0)


def scale_to_unit_range(tensor: torch.Tensor) -> torch.Tensor:
    """
    Scale tensor from [-1, 1] to [0, 1] range.

    Args:
        tensor: Tensor in range [-1, 1]

    Returns:
        Tensor in range [0, 1]
    """
    return (tensor + 1) / 2


def scale_to_signed_range(tensor: torch.Tensor) -> torch.Tensor:
    """
    Scale tensor from [0, 1] to [-1, 1] range.

    Args:
        tensor: Tensor in range [0, 1]

    Returns:
        Tensor in range [-1, 1]
    """
    return tensor * 2 - 1


def ensure_rgb(image: Image.Image) -> Image.Image:
    """
    Ensure PIL Image is in RGB mode.

    Args:
        image: PIL Image in any mode

    Returns:
        PIL Image in RGB mode
    """
    if image.mode != "RGB":
        return image.convert("RGB")
    return image
