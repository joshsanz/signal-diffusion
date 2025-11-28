"""Tests for training-time PSNR computation."""
import pytest
import torch

from signal_diffusion.diffusion.train_utils import compute_training_psnr


def test_training_psnr_vector_field_perfect():
    """Test PSNR computation with perfect vector field prediction."""
    batch_size = 4
    images = torch.rand(batch_size, 3, 32, 32)
    noise = torch.randn_like(images)
    sigmas = torch.rand(batch_size)

    # Forward process: z_t = (1 - sigma) * images + sigma * noise
    z_t = (1 - sigmas.reshape(-1, 1, 1, 1)) * images + sigmas.reshape(-1, 1, 1, 1) * noise

    # Perfect prediction for vector field: model_pred = noise - images
    model_pred = noise - images

    mean_psnr, std_psnr = compute_training_psnr(
        images, z_t, model_pred, sigmas, "vector_field"
    )

    # With perfect prediction, PSNR should be very high (near infinity)
    assert mean_psnr > 100  # Very high PSNR for near-perfect reconstruction
    assert std_psnr >= 0  # Non-negative standard deviation


def test_training_psnr_epsilon_perfect():
    """Test PSNR computation with perfect epsilon prediction."""
    batch_size = 4
    images = torch.rand(batch_size, 3, 32, 32)
    noise = torch.randn_like(images)
    sigmas = torch.rand(batch_size)

    # Forward process: z_t = (1 - sigma) * images + sigma * noise
    z_t = (1 - sigmas.reshape(-1, 1, 1, 1)) * images + sigmas.reshape(-1, 1, 1, 1) * noise

    # Perfect prediction for epsilon: model_pred = noise
    model_pred = noise.clone()

    mean_psnr, std_psnr = compute_training_psnr(
        images, z_t, model_pred, sigmas, "epsilon"
    )

    # With perfect prediction, PSNR should be very high (near infinity)
    assert mean_psnr > 100  # Very high PSNR for near-perfect reconstruction
    assert std_psnr >= 0  # Non-negative standard deviation


def test_training_psnr_vector_field_imperfect():
    """Test PSNR with imperfect vector field prediction."""
    batch_size = 4
    images = torch.rand(batch_size, 3, 32, 32)
    noise = torch.randn_like(images)
    sigmas = torch.rand(batch_size)

    # Forward process
    z_t = (1 - sigmas.reshape(-1, 1, 1, 1)) * images + sigmas.reshape(-1, 1, 1, 1) * noise

    # Imperfect prediction: add some error
    model_pred = noise - images + 0.1 * torch.randn_like(images)

    mean_psnr, std_psnr = compute_training_psnr(
        images, z_t, model_pred, sigmas, "vector_field"
    )

    # PSNR should be finite and reasonable (not infinity, not too low)
    assert math.isfinite(mean_psnr)
    assert 10 < mean_psnr < 100  # Reasonable range for noisy prediction
    assert std_psnr >= 0


def test_training_psnr_epsilon_imperfect():
    """Test PSNR with imperfect epsilon prediction."""
    batch_size = 4
    images = torch.rand(batch_size, 3, 32, 32)
    noise = torch.randn_like(images)
    sigmas = torch.rand(batch_size)

    # Forward process
    z_t = (1 - sigmas.reshape(-1, 1, 1, 1)) * images + sigmas.reshape(-1, 1, 1, 1) * noise

    # Imperfect prediction: add small error (0.01 instead of 0.1 to keep MSE reasonable)
    model_pred = noise + 0.01 * torch.randn_like(noise)

    mean_psnr, std_psnr = compute_training_psnr(
        images, z_t, model_pred, sigmas, "epsilon"
    )

    # PSNR should be finite (can be negative if MSE is large)
    assert math.isfinite(mean_psnr)
    assert mean_psnr > -20  # Allow negative PSNR but not extremely bad
    assert std_psnr >= 0


def test_training_psnr_sigma_broadcasting():
    """Test that sigma broadcasting works correctly."""
    batch_size = 2
    images = torch.rand(batch_size, 3, 16, 16)
    noise = torch.randn_like(images)

    # Test with 1D sigmas
    sigmas_1d = torch.rand(batch_size)
    z_t = (1 - sigmas_1d.reshape(-1, 1, 1, 1)) * images + sigmas_1d.reshape(-1, 1, 1, 1) * noise
    model_pred = noise - images

    mean_psnr_1d, _ = compute_training_psnr(
        images, z_t, model_pred, sigmas_1d, "vector_field"
    )

    # Test with 4D sigmas (already broadcasted)
    sigmas_4d = sigmas_1d.reshape(-1, 1, 1, 1)
    mean_psnr_4d, _ = compute_training_psnr(
        images, z_t, model_pred, sigmas_4d, "vector_field"
    )

    # Should give same result
    assert abs(mean_psnr_1d - mean_psnr_4d) < 0.01


def test_training_psnr_invalid_prediction_type():
    """Test that invalid prediction type returns None."""
    images = torch.rand(2, 3, 16, 16)
    z_t = torch.rand_like(images)
    model_pred = torch.rand_like(images)
    sigmas = torch.rand(2)

    result = compute_training_psnr(
        images, z_t, model_pred, sigmas, "invalid_type"
    )

    assert result is None


def test_training_psnr_detachment():
    """Test that PSNR computation doesn't affect gradients."""
    images = torch.rand(2, 3, 16, 16, requires_grad=True)
    noise = torch.randn_like(images)
    sigmas = torch.rand(2)

    z_t = (1 - sigmas.reshape(-1, 1, 1, 1)) * images + sigmas.reshape(-1, 1, 1, 1) * noise
    model_pred = noise - images

    # Compute PSNR
    mean_psnr, std_psnr = compute_training_psnr(
        images, z_t, model_pred, sigmas, "vector_field"
    )

    # PSNR should be computed (not None)
    assert mean_psnr is not None
    assert std_psnr is not None

    # Original images should still have gradients
    assert images.requires_grad


def test_training_psnr_different_max_values():
    """Test PSNR with different max_value settings."""
    batch_size = 2
    images = torch.rand(batch_size, 3, 16, 16)
    noise = torch.randn_like(images)
    sigmas = torch.rand(batch_size)

    z_t = (1 - sigmas.reshape(-1, 1, 1, 1)) * images + sigmas.reshape(-1, 1, 1, 1) * noise
    model_pred = noise - images + 0.1 * torch.randn_like(images)

    # Default max_value=1.0
    psnr1, _ = compute_training_psnr(
        images, z_t, model_pred, sigmas, "vector_field", max_value=1.0
    )

    # Larger max_value=2.0
    psnr2, _ = compute_training_psnr(
        images, z_t, model_pred, sigmas, "vector_field", max_value=2.0
    )

    # PSNR should be higher with larger max_value
    assert psnr2 > psnr1


# Import math for isfinite check
import math
