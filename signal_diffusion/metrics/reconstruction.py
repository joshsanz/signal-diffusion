"""Reconstruction quality metrics."""
from __future__ import annotations

import math

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
    """Compute mean and std PSNR across a batch.

    Averages in linear space (MSE) then converts to dB for correct statistics.
    """
    # Compute MSE for each sample
    mses = []
    for orig, recon in zip(originals, reconstructions):
        mse = torch.mean((orig - recon) ** 2).item()
        mses.append(mse)

    # Filter out perfect reconstructions for std calculation
    finite_mses = [mse for mse in mses if mse > 0]

    if not finite_mses:
        # All perfect reconstructions
        return float("inf"), 0.0

    # Average in linear space
    mean_mse = sum(finite_mses) / len(finite_mses)

    # Convert to dB
    mean_psnr = 20 * math.log10(max_value / math.sqrt(mean_mse))

    # Compute standard deviation in dB space
    psnrs = []
    for mse in finite_mses:
        psnr = 20 * math.log10(max_value / math.sqrt(mse))
        psnrs.append(psnr)

    if len(psnrs) > 1:
        psnr_mean = sum(psnrs) / len(psnrs)
        variance = sum((p - psnr_mean) ** 2 for p in psnrs) / len(psnrs)
        std_psnr = math.sqrt(variance)
    else:
        std_psnr = 0.0

    return mean_psnr, std_psnr
