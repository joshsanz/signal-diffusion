"""Classifier-free guidance utilities for diffusion sampling."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor


def apply_cfg_guidance(
    x_t: Tensor,
    timestep: Tensor | float,
    delta_t: float,
    delta_T: float,
    model_eval_fn: Callable[[Tensor, Any, Any], Tensor],
    cond_vector: Any,
    null_cond_vector: Any,
    cfg_scale: float,
    prediction_type: str,
    **kwargs,
) -> Tensor:
    """Apply classifier-free guidance by orchestrating model evaluations.

    This function handles both regular CFG (for epsilon prediction) and
    rectified-CFG++ (for vector_field prediction).

    Args:
        x_t: Current noisy sample (B, C, H, W)
        timestep: Current timestep (scalar or tensor)
        delta_t: [0, 1] timestep difference for Rectified-CFG++ (only for vector_field)
        delta_T: [1, T] timestep difference for Rectified-CFG++ (only for vector_field)
        model_eval_fn: Callback to evaluate model with signature:
            model_eval_fn(model_input, timestep, conditioning) -> output
            The callback takes model input, timestep, and a single conditioning tensor/dict,
            then routes the conditioning appropriately to the model.
        cond_vector: Conditional input (tensor, dict, or other format)
        null_cond_vector: Unconditional input (tensor, dict, or other format)
        cfg_scale: Guidance scale (1.0 = no guidance, 7.5 = strong)
        prediction_type: "epsilon" for noise prediction, "vector_field" for flow matching
        **kwargs: Additional parameters for rectified-cfg++ (e.g., phi, intermediate_steps)

    Returns:
        Guided model output (B, C, H, W)

    Raises:
        ValueError: If prediction_type is invalid

    Notes:
        - prediction_type="epsilon": Uses regular CFG (concatenated batch, single call)
        - prediction_type="vector_field": Uses rectified-CFG++ (separate calls at intermediate points)
        - This function handles all input batching/concatenation logic
    """
    if prediction_type == "epsilon":
        # Regular CFG for noise prediction (SD v1.5)
        # Single model evaluation with concatenated inputs
        model_input = torch.cat([x_t, x_t], dim=0)

        # Prepare timestep (double for concatenated batch)
        if isinstance(timestep, Tensor):
            timestep_input = torch.cat([timestep, timestep], dim=0)
        else:
            timestep_input = timestep

        # Concatenate conditioning (null first, cond second)
        if isinstance(cond_vector, dict):
            # Dict conditioning: concatenate each key
            batched_cond = {
                key: torch.cat([null_cond_vector[key], cond_vector[key]], dim=0)
                for key in cond_vector.keys()
            }
        elif isinstance(cond_vector, Tensor):
            # Tensor conditioning: simple concatenation
            batched_cond = torch.cat([null_cond_vector, cond_vector], dim=0)
        else:
            raise TypeError(f"Unsupported conditioning type: {type(cond_vector)}")

        # Single model evaluation
        model_output = model_eval_fn(model_input, timestep_input, batched_cond)

        # Split and apply guidance
        output_uncond, output_cond = model_output.chunk(2)
        return output_uncond + cfg_scale * (output_cond - output_uncond)

    elif prediction_type == "vector_field":
        # Rectified-CFG++ for flow matching (Hourglass, LocalMamba, DiT, SD 3.5)
        # From arxiv.org/pdf/2510.07631
        v_t_cond = model_eval_fn(x_t, timestep, cond_vector)
        x_t_halfdt = x_t + 0.5 * delta_t * v_t_cond
        # TODO: optional additive noise here for stochasticity
        # Prepare half-timestep
        assert isinstance(timestep, Tensor), "timestep must be a Tensor for vector_field prediction with Rectified-CFG++"
        halfdt_timestep = torch.cat([timestep, timestep], dim=0) - int(delta_T / 2)
        halfdt_input = torch.cat([x_t_halfdt, x_t_halfdt], dim=0)
        if isinstance(cond_vector, dict):
            halfdt_cond_vector = {
                key: torch.cat([null_cond_vector[key], cond_vector[key]], dim=0)
                for key in cond_vector.keys()
            }
        elif isinstance(cond_vector, Tensor):
            halfdt_cond_vector = torch.cat([null_cond_vector, cond_vector], dim=0)
        else:
            raise TypeError(f"Unsupported conditioning type: {type(cond_vector)}")
        # Evaluate at half-timestep for both cond and uncond
        v_t_uncond_halfdt, v_t_cond_halfdt = model_eval_fn(halfdt_input, halfdt_timestep, halfdt_cond_vector)
        # alpha_t in the paper is computed as below, but cfg_scale is used directly in the implementation
        # alpha_t = cfg_scale * (1 - t) ** gamma
        v_lamda_t = v_t_cond + cfg_scale * (v_t_cond_halfdt - v_t_uncond_halfdt)
        return v_lamda_t
    else:
        raise ValueError(
            f"Unknown prediction_type: {prediction_type}. Expected 'epsilon' or 'vector_field'."
        )
