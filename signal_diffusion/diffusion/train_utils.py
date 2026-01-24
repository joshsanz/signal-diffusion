"""Training objective helpers for diffusion models."""
import warnings

import torch


def sample_timestep_logitnorm(
    batch_size: int,
    *,
    num_train_timesteps: int,
    timesteps: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Sample timesteps using a logit-normal distribution approximation."""
    z = torch.randn(batch_size, device=device)
    ln = torch.nn.functional.sigmoid(z)
    t = torch.ceil(num_train_timesteps * ln)
    scheduler_timesteps = timesteps.to(device)
    indices = torch.argmin(torch.abs(t.unsqueeze(0) - scheduler_timesteps.unsqueeze(1)), dim=0)
    return scheduler_timesteps[indices]


def get_sigmas_from_timesteps(scheduler, timesteps: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    # Force computation on CPU because small batch sizes incur more overhead on GPU than slowdown from CPU computation
    # ~4x faster for batch size 128 (i.e. 128 timesteps)
    schedule_timesteps = scheduler.timesteps.detach().cpu()
    sigmas = scheduler.sigmas.detach().cpu()
    timesteps_cpu = timesteps.detach().cpu()

    schedule_timesteps_f = schedule_timesteps.float()
    timesteps_f = timesteps_cpu.float()
    neg_schedule = -schedule_timesteps_f
    neg_timesteps = -timesteps_f

    indices = torch.searchsorted(neg_schedule, neg_timesteps, right=False)
    max_index = schedule_timesteps_f.numel()

    lower_idx = torch.clamp(indices - 1, 0, max_index - 1)
    upper_idx = torch.clamp(indices, 0, max_index - 1)

    sigmas_out = torch.empty_like(timesteps_f, dtype=sigmas.dtype)
    mask_high = indices == 0
    mask_low = indices == max_index
    mask_mid = ~(mask_high | mask_low)

    if mask_high.any():
        sigmas_out[mask_high] = sigmas[0]
    if mask_low.any():
        sigmas_out[mask_low] = sigmas[-1]
    if mask_mid.any():
        t0 = schedule_timesteps_f[lower_idx[mask_mid]]
        t1 = schedule_timesteps_f[upper_idx[mask_mid]]
        s0 = sigmas[lower_idx[mask_mid]]
        s1 = sigmas[upper_idx[mask_mid]]
        weight = (timesteps_f[mask_mid] - t0) / (t1 - t0)
        sigmas_out[mask_mid] = s0 + weight * (s1 - s0)
    return sigmas_out.flatten().to(device)


def get_snr(scheduler, timesteps: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    sigma = get_sigmas_from_timesteps(scheduler, timesteps, device=device)
    snr = (1 - sigma) ** 2 / sigma ** 2
    return snr


def apply_min_gamma_snr(snr: torch.Tensor, *, gamma: float | None, prediction_type: str) -> torch.Tensor:
    # Hang et al., "Efficient Diffusion Training via Min-SNR Weighting", NeurIPS 2023
    # https://arxiv.org/pdf/2303.09556
    if gamma is None:
        return torch.ones_like(snr)
    # min(1/sigma^2, gamma)
    mse_loss_weights = torch.minimum(snr, gamma * torch.ones_like(snr))
    if prediction_type == "epsilon":
        return mse_loss_weights / snr
    if prediction_type == "vector_field":
        return mse_loss_weights / (snr + 1)
    raise ValueError(f"Unsupported prediction type '{prediction_type}'")


def apply_soft_min_gamma_snr(snr: torch.Tensor, *, gamma: float | None, prediction_type: str) -> torch.Tensor:
    # Soft min variant of Hang et al., "Efficient Diffusion Training via Min-SNR Weighting", NeurIPS 2023
    # Used by Hourglass Diffusion Models paper
    # https://openreview.net/pdf?id=WRIn2HmtBS
    if gamma is None:
        return torch.ones_like(snr)
    # 1 / (sigma^2 + gamma^-1)
    mse_loss_weights = 1 / (1 / snr + 1 / gamma)
    if prediction_type == "epsilon":
        return mse_loss_weights / snr
    if prediction_type == "vector_field":
        return mse_loss_weights / (snr + 1)
    raise ValueError(f"Unsupported prediction type '{prediction_type}'")


def verify_scheduler(scheduler) -> None:
    """Basic sanity checks for FlowMatch schedulers."""
    if max(scheduler.sigmas) != 1.0:  # pragma: no cover - depends on scheduler config
        warnings.warn("Scheduler maximum sigma is not 1.0; FlowMatch assumptions may not hold")


def compute_training_psnr(
    images: torch.Tensor,
    z_t: torch.Tensor,
    model_pred: torch.Tensor,
    sigmas: torch.Tensor,
    prediction_type: str,
    max_value: float = 1.0,
) -> tuple[float, float] | None:
    """Compute PSNR by algebraically inverting diffusion predictions.

    Args:
        images: Original clean images (B, C, H, W)
        z_t: Noisy images at timestep t (B, C, H, W)
        model_pred: Model predictions (B, C, H, W)
        sigmas: Noise levels (B,) or (B, 1, 1, 1)
        prediction_type: "epsilon" or "vector_field"
        max_value: Maximum pixel value for PSNR calculation

    Returns:
        (mean_psnr, std_psnr) or None if computation fails
    """
    try:
        # Ensure all tensors are detached and on same device
        images = images.detach()
        z_t = z_t.detach()
        model_pred = model_pred.detach()
        sigmas = sigmas.detach()

        # Reshape sigmas to broadcast correctly
        if sigmas.ndim == 1:
            sigmas = sigmas.reshape(-1, 1, 1, 1)

        # Algebraic inversion (prediction-type dependent)
        if prediction_type == "vector_field":
            # z_t = images + sigma * model_pred
            # Therefore: predicted_images = z_t - sigma * model_pred
            predicted_images = z_t - sigmas * model_pred
        elif prediction_type == "epsilon":
            # z_t = (1 - sigma) * images + sigma * model_pred
            # Therefore: predicted_images = (z_t - sigma * model_pred) / (1 - sigma)
            predicted_images = (z_t - sigmas * model_pred) / (1 - sigmas)
        else:
            return None

        # Compute PSNR using the corrected averaging function
        from signal_diffusion.metrics import compute_batch_psnr
        mean_psnr, std_psnr = compute_batch_psnr(
            images, predicted_images, max_value=max_value
        )

        return mean_psnr, std_psnr

    except Exception:
        # Return None on any error to avoid crashing training
        return None
