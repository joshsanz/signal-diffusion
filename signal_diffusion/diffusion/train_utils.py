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
    schedule_timesteps = scheduler.timesteps.to(device)
    sigmas = scheduler.sigmas.to(device)
    step_indices = [scheduler.index_for_timestep(t.item(), schedule_timesteps) for t in timesteps]
    sigmas = sigmas[step_indices].flatten()
    return sigmas


def get_snr(scheduler, timesteps: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    sigma = get_sigmas_from_timesteps(scheduler, timesteps, device=device)
    snr = (1 - sigma) ** 2 / sigma ** 2
    return snr


def apply_min_gamma_snr(snr: torch.Tensor, *, timesteps: torch.Tensor, gamma: float | None, prediction_type: str) -> torch.Tensor:
    if gamma is None:
        return torch.ones_like(snr)
    mse_loss_weights = torch.minimum(snr, gamma * torch.ones_like(snr))
    if prediction_type == "epsilon":
        return mse_loss_weights / snr
    if prediction_type == "vector_field":
        return mse_loss_weights / (snr + 1)
    raise ValueError(f"Unsupported prediction type '{prediction_type}'")


def verify_scheduler(scheduler) -> None:
    """Basic sanity checks for FlowMatch schedulers."""
    if max(scheduler.sigmas) != 1.0:  # pragma: no cover - depends on scheduler config
        warnings.warn("Scheduler maximum sigma is not 1.0; FlowMatch assumptions may not hold")
