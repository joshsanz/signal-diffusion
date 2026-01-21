"""Classifier-free guidance utilities for diffusion sampling."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import Tensor


def apply_cfg_guidance(
    x_t: Tensor,
    timestep: Tensor | float,
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
        # Separate model evaluations (not concatenated)
        # TODO: Implement rectified-CFG++ algorithm from arxiv.org/pdf/2510.07631
        # For now, use regular CFG formula with separate calls
        output_uncond = model_eval_fn(x_t, timestep, null_cond_vector)
        output_cond = model_eval_fn(x_t, timestep, cond_vector)

        # TODO: Replace with rectified-CFG++ when implemented
        # Rectified-CFG++ will require additional model_eval_fn calls at intermediate points
        # between x_t and the final sample, using interpolated conditioning
        return output_uncond + cfg_scale * (output_cond - output_uncond)

    else:
        raise ValueError(
            f"Unknown prediction_type: {prediction_type}. Expected 'epsilon' or 'vector_field'."
        )
