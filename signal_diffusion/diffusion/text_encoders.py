"""Text encoder loading and embedding utilities for diffusion models."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from signal_diffusion.log_setup import get_logger

LOGGER = get_logger(__name__)


class DualCLIPTextEncoder(nn.Module):
    """Dual CLIP text encoder for Hourglass/LocalMamba caption conditioning.

    Loads CLIP-L (768D) and CLIP-G (1280D) text encoders and concatenates
    their pooled outputs to produce a 2048D embedding per caption.

    This follows the SD 3.5 architecture pattern but only uses the two CLIP
    encoders, skipping T5XXL for efficiency.
    """

    def __init__(
        self,
        sd_model_id: str = "stabilityai/stable-diffusion-3.5-medium",
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

        self._device = device
        self._dtype = dtype

        LOGGER.info("Loading CLIP-L text encoder from %s", sd_model_id)
        self.tokenizer_l = CLIPTokenizer.from_pretrained(sd_model_id, subfolder="tokenizer")
        self.text_encoder_l = CLIPTextModel.from_pretrained(
            sd_model_id, subfolder="text_encoder", torch_dtype=dtype
        )

        LOGGER.info("Loading CLIP-G text encoder from %s", sd_model_id)
        self.tokenizer_g = CLIPTokenizer.from_pretrained(sd_model_id, subfolder="tokenizer_2")
        self.text_encoder_g = CLIPTextModelWithProjection.from_pretrained(
            sd_model_id, subfolder="text_encoder_2", torch_dtype=dtype
        )

        # Freeze encoders
        self.text_encoder_l.requires_grad_(False)
        self.text_encoder_g.requires_grad_(False)

        self.output_dim = 768 + 1280  # 2048D total

        if device is not None:
            self.to(device)

        LOGGER.info(
            "DualCLIPTextEncoder initialized: CLIP-L (768D) + CLIP-G (1280D) = %dD output",
            self.output_dim,
        )

    def encode(self, captions: Sequence[str]) -> torch.Tensor:
        """Encode captions to pooled embeddings.

        Args:
            captions: List/sequence of caption strings
        Returns:
            (B, 2048) pooled embedding tensor
        """
        device = next(self.parameters()).device
        captions = list(captions)

        # CLIP-L encoding
        inputs_l = self.tokenizer_l(
            captions,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs_l = self.text_encoder_l(inputs_l.input_ids.to(device))
            pooled_l = outputs_l.pooler_output  # (B, 768)

        # CLIP-G encoding
        inputs_g = self.tokenizer_g(
            captions,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs_g = self.text_encoder_g(inputs_g.input_ids.to(device))
            pooled_g = outputs_g.text_embeds  # (B, 1280)

        # Concatenate pooled outputs
        return torch.cat([pooled_l, pooled_g], dim=-1)  # (B, 2048)

    def forward(self, captions: Sequence[str]) -> torch.Tensor:
        """Forward pass - same as encode()."""
        return self.encode(captions)


def load_sd35_vae(
    model_id: str = "stabilityai/stable-diffusion-3.5-medium",
    dtype: torch.dtype = torch.float16,
):
    """Load VAE from SD 3.5 medium for latent space training.

    Args:
        model_id: HuggingFace model ID for SD 3.5
        dtype: Data type for the VAE

    Returns:
        AutoencoderKL VAE model
    """
    from diffusers import AutoencoderKL

    LOGGER.info("Loading VAE from %s", model_id)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    return vae


def create_text_encoder_for_adapter(
    adapter_name: str,
    settings: dict | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float16,
) -> DualCLIPTextEncoder | None:
    """Factory function to create appropriate text encoder for an adapter.

    Args:
        adapter_name: Name of the diffusion adapter ("hourglass", "localmamba", etc.)
        settings: Optional settings dict containing model paths
        device: Target device
        dtype: Data type

    Returns:
        DualCLIPTextEncoder for compatible adapters, None otherwise
    """
    # Only Hourglass and LocalMamba support caption conditioning with CLIP
    if adapter_name not in {"hourglass", "localmamba"}:
        return None

    sd_model_id = "stabilityai/stable-diffusion-3.5-medium"
    if settings and "stable_diffusion_model_id" in settings:
        sd_model_id = settings["stable_diffusion_model_id"]

    return DualCLIPTextEncoder(
        sd_model_id=sd_model_id,
        device=device,
        dtype=dtype,
    )
