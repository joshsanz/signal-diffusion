"""Evaluation helpers for diffusion training."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch_fidelity import create_feature_extractor, metric_kid
from torchvision.utils import make_grid


_FEATURE_EXTRACTOR_CACHE: dict[str, torch.nn.Module] = {}


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


def collect_real_samples(
    data_iterable: Iterable,
    num_samples: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    collected: list[torch.Tensor] = []
    total = 0
    for batch in data_iterable:
        batch_pixels = getattr(batch, "pixel_values", None)
        if batch_pixels is None:
            raise ValueError("Batch missing 'pixel_values' tensor")
        batch_pixels = batch_pixels.to(device=device)
        collected.append(batch_pixels)
        total += batch_pixels.shape[0]
        if total >= num_samples:
            break
    if not collected:
        raise ValueError("Failed to gather real samples for KID evaluation")
    stacked = torch.cat(collected, dim=0)
    return stacked[:num_samples]


def compute_kid_score(
    generated: torch.Tensor,
    reference: torch.Tensor,
    *,
    device: torch.device,
) -> float:
    if generated.ndim != 4 or reference.ndim != 4:
        raise ValueError("Expected image tensors with shape (N, C, H, W)")
    generated = generated.to(device=device)
    reference = reference.to(device=device)

    feature_extractor = _get_dino_feature_extractor(device)

    with torch.no_grad():
        normalized_gen = ((generated.clamp(-1.0, 1.0) + 1.0) / 2.0).to(dtype=torch.float32)
        normalized_ref = ((reference.clamp(-1.0, 1.0) + 1.0) / 2.0).to(dtype=torch.float32)
        gen_features = feature_extractor(normalized_gen)
        ref_features = feature_extractor(normalized_ref)

    kid = metric_kid(gen_features, ref_features)
    if isinstance(kid, torch.Tensor):
        return float(kid.mean().item())
    return float(kid)


def _get_dino_feature_extractor(device: torch.device) -> torch.nn.Module:
    key_parts = [device.type]
    if device.type == "cuda":
        index = getattr(device, "index", None)
        key_parts.append("none" if index is None else str(index))
    cache_key = ":".join(key_parts)

    extractor = _FEATURE_EXTRACTOR_CACHE.get(cache_key)
    if extractor is None:
        extractor = create_feature_extractor(
            "dinov2-vit-s-14",
            use_gpu=device.type == "cuda",
            fuse_forward=True,
        )
        extractor = extractor.to(device)
        extractor.eval()
        extractor.requires_grad_(False)
        _FEATURE_EXTRACTOR_CACHE[cache_key] = extractor
    else:
        extractor = extractor.to(device)
        extractor.eval()

    return extractor
