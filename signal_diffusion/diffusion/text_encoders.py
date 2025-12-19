"""Text encoder loading and embedding utilities for diffusion models."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from signal_diffusion.log_setup import get_logger

LOGGER = get_logger(__name__)


class DualCLIPTextEncoder(nn.Module):
    """Dual CLIP text encoder for caption conditioning in diffusion models.

    This module combines two CLIP text encoders (CLIP-L and CLIP-G) from Stable
    Diffusion 3.5 to produce rich 2048-dimensional caption embeddings for conditioning
    diffusion models on text descriptions. Both encoders are frozen during training for
    memory efficiency and stability.

    Architecture:
        The encoder processes captions through two parallel pathways:

        1. CLIP-L (OpenAI CLIP): Tokenizes to 77 tokens, encodes to 768-dimensional
           pooler_output using CLIPTextModel
        2. CLIP-G (OpenCLIP): Tokenizes to 77 tokens, encodes to 1280-dimensional
           text_embeds using CLIPTextModelWithProjection
        3. Concatenates both outputs along feature dimension: 768 + 1280 = 2048D

        Both encoders are loaded from the Stable Diffusion 3.5 checkpoint and frozen
        (requires_grad=False) to prevent gradient updates during training.

    Supported Adapters:
        - Hourglass: Uses embeddings via mapping_cond parameter
        - LocalMamba: Uses embeddings via mapping_cond parameter
        - Stable Diffusion 3.5: Uses embeddings via pooled_projections parameter

    Tokenization:
        - Max length: 77 tokens (standard CLIP)
        - Padding: "max_length"
        - Truncation: Enabled for longer captions

    Example:
        Basic usage with single caption:

        >>> from signal_diffusion.diffusion.text_encoders import DualCLIPTextEncoder
        >>> import torch
        >>>
        >>> text_encoder = DualCLIPTextEncoder(device="cuda", dtype=torch.float16)
        >>> captions = ["healthy EEG signal from a 25-year-old patient"]
        >>> embeddings = text_encoder.encode(captions)
        >>> embeddings.shape
        torch.Size([1, 2048])

        Batch encoding multiple captions:

        >>> captions = [
        ...     "healthy EEG signal",
        ...     "parkinsons tremor pattern in elderly patient",
        ...     "emotional arousal during positive stimulus"
        ... ]
        >>> embeddings = text_encoder.encode(captions)
        >>> embeddings.shape
        torch.Size([3, 2048])

    Note:
        This encoder skips the T5XXL encoder from the full SD 3.5 pipeline for
        efficiency. The dual CLIP encoders provide sufficient caption conditioning
        for most EEG spectrogram generation tasks.
    """

    def __init__(
        self,
        sd_model_id: str = "stabilityai/stable-diffusion-3.5-medium",
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        """Initialize the dual CLIP text encoder.

        Args:
            sd_model_id: HuggingFace model ID for Stable Diffusion 3.5 checkpoint.
                Must contain both CLIP-L and CLIP-G text encoders in subfolders
                "text_encoder" and "text_encoder_2" respectively.
                Default: "stabilityai/stable-diffusion-3.5-medium"
            device: Target device for the encoders ("cuda", "cpu", or torch.device).
                If None, encoders remain on CPU until moved explicitly.
            dtype: Data type for encoder weights (torch.float16, torch.float32, etc.).
                Default: torch.float16 for memory efficiency.

        Note:
            Both CLIP encoders are automatically frozen (requires_grad=False) after
            loading to prevent gradient updates during training.
        """
        super().__init__()
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

        self._device = device
        self._dtype = dtype

        LOGGER.info("Loading CLIP-L text encoder from %s", sd_model_id)
        self.tokenizer_l = CLIPTokenizer.from_pretrained(sd_model_id, subfolder="tokenizer")
        self.text_encoder_l = CLIPTextModel.from_pretrained(
            sd_model_id, subfolder="text_encoder", dtype=dtype
        )

        LOGGER.info("Loading CLIP-G text encoder from %s", sd_model_id)
        self.tokenizer_g = CLIPTokenizer.from_pretrained(sd_model_id, subfolder="tokenizer_2")
        self.text_encoder_g = CLIPTextModelWithProjection.from_pretrained(
            sd_model_id, subfolder="text_encoder_2", dtype=dtype
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
        """Encode text captions to pooled 2048-dimensional embeddings.

        This method processes captions through both CLIP-L and CLIP-G encoders in parallel,
        extracts their pooled outputs, and concatenates them along the feature dimension.
        All encoding is performed under torch.no_grad() context for efficiency.

        Processing Pipeline:
            1. Convert input to list of strings
            2. CLIP-L: Tokenize → Encode → Extract pooler_output (768D)
            3. CLIP-G: Tokenize → Encode → Extract text_embeds (1280D)
            4. Concatenate along dim=-1 → (B, 2048) output

        Args:
            captions: Sequence of caption strings to encode. Can be list, tuple, or any
                sequence type. Empty strings are valid and will produce zero-like embeddings.

        Returns:
            torch.Tensor: Concatenated embeddings of shape (B, 2048) where B is the
                batch size (number of captions). Output dtype matches the encoder dtype
                (typically float16), and device matches the encoder device.

        Example:
            >>> text_encoder = DualCLIPTextEncoder(device="cuda")
            >>> captions = ["healthy EEG", "parkinsons pattern"]
            >>> emb = text_encoder.encode(captions)
            >>> emb.shape, emb.dtype, emb.device
            (torch.Size([2, 2048]), torch.float16, device(type='cuda', index=0))

        Note:
            - Captions longer than 77 tokens are automatically truncated
            - Shorter captions are padded to 77 tokens
            - Empty captions ("") produce valid embeddings (useful for CFG)
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
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", dtype=dtype)
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
    if settings and "stable_diffusion_model_id" in settings.hf_models:
        sd_model_id = settings.hf_models["stable_diffusion_model_id"]

    return DualCLIPTextEncoder(
        sd_model_id=sd_model_id,
        device=device,
        dtype=dtype,
    )
