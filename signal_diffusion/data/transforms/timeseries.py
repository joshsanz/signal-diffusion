"""Transforms for time-domain EEG signals."""

from __future__ import annotations


import torch
import torch.nn as nn


class ZScoreNormalize(nn.Module):
    """Z-score normalization using precomputed statistics."""

    def __init__(self, means: torch.Tensor, stds: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("means", means.view(-1, 1))
        self.register_buffer("stds", stds.view(-1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return (x - self.means) / (self.stds + 1e-8)


class GaussianNoise(nn.Module):
    """Add Gaussian noise for augmentation."""

    def __init__(self, std: float = 0.01) -> None:
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.training and self.std > 0:
            return x + torch.randn_like(x) * self.std
        return x


class ChannelDropout(nn.Module):
    """Randomly zero out channels."""

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.training and self.p > 0:
            mask = torch.rand((x.shape[0], 1), device=x.device, dtype=x.dtype) > self.p
            return x * mask
        return x


class TemporalCrop(nn.Module):
    """Extract a fixed-length window from a signal."""

    def __init__(self, length: int, center: bool = False) -> None:
        super().__init__()
        self.length = length
        self.center = center

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if x.shape[-1] == self.length:
            return x

        max_start = x.shape[-1] - self.length
        if max_start < 0:
            raise ValueError(
                f"Cannot crop length {self.length} from signal with length {x.shape[-1]}"
            )

        if self.center:
            start = max_start // 2
        else:
            start = torch.randint(0, max_start + 1, (1,), device=x.device).item()

        return x[..., start : start + self.length]
