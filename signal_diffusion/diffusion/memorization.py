"""Per-sample initial noise adjustment to mitigate memorization in diffusion models.

Implements the per-sample mitigation from:
  Adjusting Initial Noise to Mitigate Memorization in Text-to-Image Diffusion Models
  https://arxiv.org/abs/2510.08625

Minimizes the L2 norm of conditional guidance (pred_cond - pred_uncond) at the
initial timestep via backpropagation until below a target threshold.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor


def adjust_initial_latent_per_sample(
    latent: Tensor,
    timestep: float | Tensor,
    model_eval_fn: Callable[[Tensor, Any, Any], Tensor],
    cond_vector: Any,
    null_cond_vector: Any,
    *,
    target_loss: float,
    lr: float = 0.01,
    max_steps: int = 100,
) -> Tensor:
    """Adjust initial noise to reduce conditional guidance magnitude (per-sample mitigation).

    Minimizes ||pred_cond - pred_uncond||_2 until below target_loss via AdamW.
    See arxiv.org/abs/2510.08625 (Adjusting Initial Noise to Mitigate Memorization).

    Args:
        latent: Initial noise tensor (B, C, H, W), will be cloned and optimized.
        timestep: Timestep for the initial denoising step (scalar or tensor).
        model_eval_fn: Callback to evaluate model with signature
            model_eval_fn(model_input, timestep, conditioning) -> output.
            The conditioning is a dict with keys like 'class' and 'mapping'.
        cond_vector: Conditional conditioning dict (e.g. class_labels, mapping_cond).
        null_cond_vector: Unconditional conditioning dict (null/empty conditioning).
        target_loss: Target L2 norm threshold; stop when loss < target_loss.
        lr: Learning rate for AdamW optimizer.
        max_steps: Maximum adjustment steps.

    Returns:
        Adjusted latent tensor, detached.
    """
    latent = latent.clone().detach().requires_grad_(True)
    optim = torch.optim.AdamW([latent], lr=lr, weight_decay=0.01)

    for _ in range(max_steps):
        pred_uncond = model_eval_fn(latent, timestep, null_cond_vector)
        pred_cond = model_eval_fn(latent, timestep, cond_vector)
        loss = (pred_cond - pred_uncond).norm(p=2)
        if loss.item() < target_loss:
            break
        optim.zero_grad()
        loss.backward()
        optim.step()

    return latent.detach()
