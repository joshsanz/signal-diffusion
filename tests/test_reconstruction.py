"""Tests for reconstruction quality metrics."""
import math

import pytest
import torch

from signal_diffusion.metrics import compute_batch_psnr, compute_psnr


def test_compute_psnr_perfect_reconstruction():
    """Test PSNR for perfect reconstruction returns infinity."""
    original = torch.rand(3, 32, 32)
    reconstructed = original.clone()

    psnr = compute_psnr(original, reconstructed)

    assert psnr == float("inf")


def test_compute_psnr_known_mse():
    """Test PSNR computation with known MSE value."""
    # Create tensors with known difference
    original = torch.ones(3, 32, 32)
    reconstructed = original + 0.1  # MSE = 0.01

    psnr = compute_psnr(original, reconstructed, max_value=1.0)

    # PSNR = 20 * log10(max_value / sqrt(MSE))
    # PSNR = 20 * log10(1.0 / sqrt(0.01))
    # PSNR = 20 * log10(1.0 / 0.1)
    # PSNR = 20 * log10(10) = 20 dB
    expected_psnr = 20.0

    assert abs(psnr - expected_psnr) < 0.01


def test_compute_batch_psnr_averaging():
    """Test that PSNR averaging uses linear space correctly."""
    # Create known MSE values
    originals = torch.tensor([
        [[1.0, 1.0], [1.0, 1.0]],
        [[1.0, 1.0], [1.0, 1.0]],
    ]).unsqueeze(1)

    # Sample 1: difference = 0.1 -> MSE = 0.01
    recon1 = originals[0] + 0.1
    # Sample 2: difference = 0.2 -> MSE = 0.04
    recon2 = originals[1] + 0.2

    reconstructions = torch.stack([recon1, recon2])

    mean_psnr, std_psnr = compute_batch_psnr(originals, reconstructions)

    # Mean MSE = (0.01 + 0.04) / 2 = 0.025
    # Mean PSNR = 20*log10(1/sqrt(0.025)) = 20*log10(1/0.158...) ≈ 16.02 dB
    expected_mean = 20 * math.log10(1 / math.sqrt(0.025))

    assert abs(mean_psnr - expected_mean) < 0.1
    assert std_psnr > 0  # Should have non-zero std deviation


def test_compute_batch_psnr_all_perfect():
    """Test batch PSNR with all perfect reconstructions."""
    originals = torch.rand(4, 3, 32, 32)
    reconstructions = originals.clone()

    mean_psnr, std_psnr = compute_batch_psnr(originals, reconstructions)

    assert mean_psnr == float("inf")
    assert std_psnr == 0.0


def test_compute_batch_psnr_single_sample():
    """Test batch PSNR with single sample."""
    original = torch.ones(1, 3, 32, 32)
    reconstructed = original + 0.1

    mean_psnr, std_psnr = compute_batch_psnr(original, reconstructed)

    # Should return PSNR for single sample
    expected_psnr = 20.0
    assert abs(mean_psnr - expected_psnr) < 0.01
    assert std_psnr == 0.0  # No variation with single sample


def test_compute_batch_psnr_different_max_value():
    """Test PSNR computation with different max_value."""
    original = torch.ones(2, 3, 32, 32)
    reconstructed = original + 0.1

    # With max_value=2.0
    psnr_max2, _ = compute_batch_psnr(original, reconstructed, max_value=2.0)
    # With max_value=1.0
    psnr_max1, _ = compute_batch_psnr(original, reconstructed, max_value=1.0)

    # PSNR should be higher with larger max_value
    # PSNR(max=2) = 20*log10(2/sqrt(0.01)) = 20*log10(20) ≈ 26 dB
    # PSNR(max=1) = 20*log10(1/sqrt(0.01)) = 20*log10(10) = 20 dB
    assert psnr_max2 > psnr_max1
    assert abs(psnr_max2 - 26.02) < 0.1
    assert abs(psnr_max1 - 20.0) < 0.1
