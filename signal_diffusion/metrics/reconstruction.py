"""Reconstruction quality metrics."""
from __future__ import annotations

import torch


def compute_psnr(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    max_value: float = 1.0,
) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = torch.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float("inf")

    return 20 * torch.log10(torch.tensor(max_value) / torch.sqrt(mse)).item()


def compute_batch_psnr(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    max_value: float = 1.0,
) -> tuple[float, float]:
    """Compute mean and std PSNR across a batch."""
    psnrs = []
    for orig, recon in zip(originals, reconstructions):
        psnrs.append(compute_psnr(orig, recon, max_value))

    psnrs_tensor = torch.tensor(psnrs)
    return psnrs_tensor.mean().item(), psnrs_tensor.std().item()
