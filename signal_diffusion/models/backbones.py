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
    """Convolutional feature extractor used by spectrogram classifiers."""

    def __init__(
        self,
        in_channels: int,
        *,
        conv_kernels: Sequence[tuple[int, int]] | None = None,
        conv_channels: Sequence[int] | None = None,
        conv_strides: Sequence[int] | None = None,
        conv_padding: Sequence[tuple[int, int]] | None = None,
        pool_kernels: Sequence[tuple[int, int]] | None = None,
        pool_strides: Sequence[tuple[int, int]] | None = None,
        linear_dims: Sequence[int] | None = None,
        activation: str = "gelu",
        dropout: float = 0.3,
        embedding_dim: int = 256,
    ) -> None:
        super().__init__()
        conv_kernels = conv_kernels or [(5, 5), (3, 3), (3, 3)]
        conv_channels = conv_channels or [8, 16, 32]
        conv_strides = conv_strides or [1, 1, 1]
        conv_padding = conv_padding or [(2, 2), (1, 1), (1, 1)]
        pool_kernels = pool_kernels or [(2, 2), (2, 2), (2, 2)]
        pool_strides = pool_strides or [(2, 2), (2, 2), (2, 2)]
        linear_dims = tuple(linear_dims or (512, 128))

        if not (len(conv_kernels) == len(conv_channels) == len(conv_strides) == len(conv_padding) == len(pool_kernels) == len(pool_strides)):
            raise ValueError("Convolution and pooling configuration lengths must match")

        self.activation_name = activation.lower()
        activation_fn = _make_activation(self.activation_name)

        conv_in_channels = [in_channels] + list(conv_channels[:-1])
        conv_layers: list[nn.Module] = []
        for idx, (kernel, out_channels, stride, padding, pool_kernel, pool_stride) in enumerate(
            zip(conv_kernels, conv_channels, conv_strides, conv_padding, pool_kernels, pool_strides)
        ):
            conv_layers.append(
                nn.Conv2d(conv_in_channels[idx], out_channels, kernel_size=kernel, stride=stride, padding=padding)
            )
            conv_layers.append(_make_activation(self.activation_name))
            conv_layers.append(nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride))
        self.convs = nn.Sequential(*conv_layers)

        linear_layers: list[nn.Module] = [nn.Flatten(), nn.Dropout(dropout)]
        if linear_dims:
            linear_layers.append(nn.LazyLinear(linear_dims[0]))
            for idx in range(len(linear_dims) - 1):
                linear_layers.append(_make_activation(self.activation_name))
                linear_layers.append(nn.Dropout(dropout))
                linear_layers.append(nn.Linear(linear_dims[idx], linear_dims[idx + 1]))
            last_dim = linear_dims[-1]
        else:
            linear_layers.append(_make_activation(self.activation_name))
            last_dim = None
        self.linear_stack = nn.Sequential(*linear_layers)
        if last_dim is None:
            self.projection = nn.LazyLinear(embedding_dim)
        else:
            self.projection = nn.Linear(last_dim, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        x = self.linear_stack(x)
        x = self.projection(x)
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

