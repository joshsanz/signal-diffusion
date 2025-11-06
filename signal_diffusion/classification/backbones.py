"""Reusable classifier backbones for EEG spectrogram models."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "lrelu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "prelu": nn.PReLU,
}


def _make_activation(name: str) -> nn.Module:
    try:
        return _ACTIVATIONS[name.lower()]()
    except KeyError as exc:  # pragma: no cover - validated upstream
        raise ValueError(f"Unsupported activation '{name}'. Available: {sorted(_ACTIVATIONS)}") from exc


class CNNBackbone(nn.Module):
    """Convolutional feature extractor with configurable block-based architecture."""

    def __init__(
        self,
        in_channels: int,
        *,
        depth: int = 3,
        layer_repeats: int = 2,
        activation: str = "gelu",
        dropout: float = 0.3,
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        if layer_repeats < 1:
            raise ValueError("layer_repeats must be at least 1")

        self.activation_name = activation.lower()
        self.embedding_dim = embedding_dim

        # Determine channel growth factor and input conv channels
        # The first block should have input_conv_channels * growth_factor
        # The final block should have embedding_dim
        # So: input_conv_channels * growth_factor^depth = embedding_dim
        growth_4x_start = embedding_dim / (4 ** depth)
        if growth_4x_start >= 8:
            growth_factor = 4
            input_conv_channels = int(growth_4x_start)
        else:
            growth_factor = 2
            input_conv_channels = int(embedding_dim / (2 ** depth))

        # Build channel list for each block (starting from growth_factor × input_conv_channels)
        channels = [input_conv_channels * (growth_factor ** (i + 1)) for i in range(depth)]

        # Input 5x5 convolution (separate from blocks)
        input_conv_layers: list[nn.Module] = [
            nn.Conv2d(in_channels, input_conv_channels, kernel_size=5, stride=1, padding=2),
            _make_activation(self.activation_name),
            nn.GroupNorm(num_groups=1, num_channels=input_conv_channels),
            nn.Dropout(dropout),
            nn.AvgPool2d(kernel_size=2, stride=2),
        ]
        self.input_conv = nn.Sequential(*input_conv_layers)

        # Build identical blocks
        blocks: list[nn.Module] = []
        current_in_channels = input_conv_channels

        for block_idx, out_channels in enumerate(channels):
            block_layers: list[nn.Module] = []

            # GroupNorm at block start
            block_layers.append(nn.GroupNorm(num_groups=1, num_channels=current_in_channels))

            # Add layer_repeats 3x3 convolutions (first one may change channels)
            for repeat_idx in range(layer_repeats):
                conv_in = current_in_channels if repeat_idx == 0 else out_channels
                block_layers.append(
                    nn.Conv2d(conv_in, out_channels, kernel_size=3, stride=1, padding=1)
                )
                block_layers.append(_make_activation(self.activation_name))

            # Dropout at end of block
            block_layers.append(nn.Dropout(dropout))

            # Pooling at end of block
            block_layers.append(nn.AvgPool2d(kernel_size=2, stride=2))

            blocks.append(nn.Sequential(*block_layers))
            current_in_channels = out_channels

        self.blocks = nn.Sequential(*blocks)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        x = self.blocks(x)
        x = self.global_avgpool(x)
        x = self.flatten(x)
        return x


class TransformerBackbone(nn.Module):
    """Transformer encoder backbone that emits a pooled embedding."""

    def __init__(
        self,
        input_dim: int,
        *,
        seq_length: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        batch_first: bool = True,
        pooling: str = "mean",
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be 'mean' or 'cls'")
        self.batch_first = batch_first
        self.pooling = pooling
        self.seq_length = seq_length

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim,
            num_heads,
            ff_dim,
            dropout=dropout,
            norm_first=True,
            batch_first=batch_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.batch_first:
            if x.shape[0] != self.seq_length:
                raise ValueError(f"Expected sequence length {self.seq_length}, got {x.shape[0]}")
        else:
            if x.shape[1] != self.seq_length:
                raise ValueError(f"Expected sequence length {self.seq_length}, got {x.shape[1]}")

        x = self.input_proj(x)
        encoded = self.encoder(x)
        if self.pooling == "mean":
            dim = 1 if self.batch_first else 0
            pooled = encoded.mean(dim=dim)
        else:  # cls token pooling
            pooled = encoded[:, 0] if self.batch_first else encoded[0]
        return self.output_proj(pooled)
