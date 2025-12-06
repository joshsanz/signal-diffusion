"""Conditioning utilities for multi-attribute diffusion training."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from signal_diffusion.diffusion.data import DiffusionBatch


class FourierFeatures(nn.Module):
    """Random Fourier features for encoding continuous values.

    Maps low-dimensional continuous inputs to high-dimensional periodic features
    using random projections with sine/cosine activations.
    """

    def __init__(self, in_features: int, out_features: int, std: float = 1.0):
        super().__init__()
        if out_features % 2 != 0:
            raise ValueError("out_features must be even for FourierFeatures")
        self.register_buffer(
            "weight", torch.randn([out_features // 2, in_features]) * std
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to Fourier features.

        Args:
            x: (B, in_features) or (B,) tensor
        Returns:
            (B, out_features) tensor with sin/cos features
        """
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        f = 2 * math.pi * x @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class AgeEmbedding(nn.Module):
    """Fourier feature embedding for continuous age values.

    Designed for age values normalized to [0, 1] range (e.g., age / 100).
    Missing values (NaN) are replaced with 0 for embedding purposes.
    """

    def __init__(self, out_features: int, std: float = 1.0):
        super().__init__()
        self.fourier = FourierFeatures(1, out_features, std=std)

    def forward(self, age: torch.Tensor) -> torch.Tensor:
        """Encode normalized age to embedding.

        Args:
            age: (B,) tensor of normalized ages in [0,1], NaN for missing
        Returns:
            (B, out_features) embedding
        """
        # Replace NaN with 0 for embedding (will be masked by CFG dropout)
        age_clean = torch.where(torch.isnan(age), torch.zeros_like(age), age)
        return self.fourier(age_clean)


class MultiAttributeEmbedding(nn.Module):
    """Combined embedding for gender, health, and age conditioning.

    Creates separate embeddings for discrete attributes (gender, health) and
    continuous attributes (age), then sums them together.

    Args:
        embedding_dim: Output dimension for all embeddings
        num_genders: Number of gender classes (default 3: M, F, dropout)
        num_health: Number of health classes (default 3: H, PD, dropout)
        age_fourier_dim: Fourier feature dimension for age (default: embedding_dim)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_genders: int = 3,
        num_health: int = 3,
        age_fourier_dim: int | None = None,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        age_fourier_dim = age_fourier_dim or embedding_dim

        self.gender_emb = nn.Embedding(num_genders, embedding_dim)
        self.health_emb = nn.Embedding(num_health, embedding_dim)
        self.age_emb = AgeEmbedding(age_fourier_dim)

        # Project age embedding to match embedding_dim if different
        if age_fourier_dim != embedding_dim:
            self.age_proj = nn.Linear(age_fourier_dim, embedding_dim, bias=False)
        else:
            self.age_proj = nn.Identity()

    def forward(
        self,
        gender_labels: torch.Tensor,
        health_labels: torch.Tensor,
        age_values: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute combined multi-attribute embedding.

        Args:
            gender_labels: (B,) int tensor of gender class indices
            health_labels: (B,) int tensor of health class indices
            age_values: (B,) float tensor of normalized ages, or None
        Returns:
            (B, embedding_dim) combined embedding
        """
        gender_emb = self.gender_emb(gender_labels)
        health_emb = self.health_emb(health_labels)

        if age_values is not None:
            age_emb = self.age_proj(self.age_emb(age_values))
        else:
            age_emb = torch.zeros_like(gender_emb)

        return gender_emb + health_emb + age_emb


def prepare_multi_attribute_labels(
    batch: "DiffusionBatch",
    *,
    device: torch.device,
    cfg_dropout: float = 0.0,
    dropout_token: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Prepare multi-attribute labels with classifier-free guidance dropout.

    Args:
        batch: DiffusionBatch containing gender_labels, health_labels, age_values
        device: Target device for tensors
        cfg_dropout: Probability of replacing labels with dropout token
        dropout_token: Index used for unconditional/dropout class

    Returns:
        Tuple of (gender_labels, health_labels, age_values) with CFG dropout applied.
        Gender and health labels are (B,) int tensors.
        Age values are (B,) float tensor or None, with NaN marking dropout.
    """
    if batch.gender_labels is None:
        raise ValueError("batch.gender_labels is required for multi-attribute conditioning")
    if batch.health_labels is None:
        raise ValueError("batch.health_labels is required for multi-attribute conditioning")

    gender_labels = batch.gender_labels.to(device)
    health_labels = batch.health_labels.to(device)
    age_values = batch.age_values.to(device) if batch.age_values is not None else None

    if cfg_dropout > 0:
        mask = torch.rand(gender_labels.shape, device=device) < cfg_dropout
        gender_labels = torch.where(
            mask, torch.full_like(gender_labels, dropout_token), gender_labels
        )
        health_labels = torch.where(
            mask, torch.full_like(health_labels, dropout_token), health_labels
        )
        if age_values is not None:
            age_values = torch.where(
                mask, torch.full_like(age_values, float("nan")), age_values
            )

    return gender_labels, health_labels, age_values


def compute_combined_class(
    gender_labels: torch.Tensor,
    health_labels: torch.Tensor,
    *,
    num_health_classes: int = 2,
    dropout_token: int = 4,
) -> torch.Tensor:
    """Compute combined class index from gender and health labels.

    Maps (gender, health) pairs to a single class index:
    - Valid combinations: gender * num_health_classes + health
    - If either is dropout (index 2), returns dropout_token

    Args:
        gender_labels: (B,) int tensor (0=M, 1=F, 2=dropout)
        health_labels: (B,) int tensor (0=H, 1=PD, 2=dropout)
        num_health_classes: Number of valid health classes (excluding dropout)
        dropout_token: Index for combined dropout class

    Returns:
        (B,) int tensor of combined class indices
    """
    is_dropout = (gender_labels == 2) | (health_labels == 2)
    combined = gender_labels * num_health_classes + health_labels
    return torch.where(is_dropout, torch.full_like(combined, dropout_token), combined)
