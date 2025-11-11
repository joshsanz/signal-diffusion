"""
VMamba building blocks for LocalMamba.

Implements MultiScanVSSM (directional scanning + fusion), VSSBlock (SS2D +
MLP + residual), and PatchMerging2D/Token ops that provide the hierarchical
encoder/decoder plumbing for `LocalMamba2DModel`.

Ported from LocalVMamba: https://github.com/OpenGVLab/LocalMamba
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from functools import partial
from einops import rearrange

from .ss2d import SS2D, BiAttn
from .local_scan import LocalScanTriton, LocalReverseTriton, local_scan, local_scan_bchw, local_reverse


class MultiScanVSSM(nn.Module):
    """Multi-directional scanning module for LocalMamba.

    Supports various scan directions:
    - h, h_flip: horizontal scanning
    - v, v_flip: vertical scanning
    - w2, w2_flip, w7, w7_flip: windowed scanning with window sizes 2 and 7
    """

    ALL_CHOICES = ('h', 'h_flip', 'v', 'v_flip', 'w2', 'w2_flip', 'w7', 'w7_flip')

    def __init__(self, dim, choices=None, token_size=None):
        super().__init__()
        self.token_size = token_size
        if choices is None:
            self.choices = MultiScanVSSM.ALL_CHOICES
        else:
            self.choices = choices
        self.attn = BiAttn(dim)

    def merge(self, xs):
        """Merge multi-directional scan results.

        Args:
            xs: [B, K, D, L] tensor from multiple scan directions

        Returns:
            [B, D, L] merged tensor
        """
        # xs: [B, K, D, L]
        # return: [B, D, L]

        # remove the padded tokens
        xs = [xs[:, i, :, :l] for i, l in enumerate(self.scan_lengths)]
        xs = self.multi_reverse(xs)
        xs = [self.attn(x.transpose(-2, -1)) for x in xs]
        x = self.forward(xs)
        return x

    def forward(self, xs):
        """Forward pass for merging scan results.

        Args:
            xs: List of [B, L, D] tensors from different scan directions

        Returns:
            [B, L, D] merged tensor
        """
        # Simply sum all directions
        x = torch.stack(xs).sum(0)
        return x

    def multi_scan(self, x):
        """Apply multi-directional scanning to input.

        Args:
            x: [B, C, H, W] tensor

        Returns:
            [B, K, C, max_length] tensor with padded scan results
        """
        # x: [B, C, H, W]
        # return: [B, K, C, H * W]
        B, C, H, W = x.shape
        self.token_size = (H, W)

        xs = [self.scan(x, direction) for direction in self.choices]  # [[B, C, H, W], ...]

        self.scan_lengths = [x.shape[2] for x in xs]
        max_length = max(self.scan_lengths)

        # pad the tokens into the same length as VMamba compute all directions together
        new_xs = []
        for x in xs:
            if x.shape[2] < max_length:
                x = F.pad(x, (0, max_length - x.shape[2]))
            new_xs.append(x)
        return torch.stack(new_xs, 1)

    def multi_reverse(self, xs):
        """Reverse multi-directional scans.

        Args:
            xs: List of [B, D, L] tensors from different scan directions

        Returns:
            List of [B, D, L] tensors after reversing
        """
        new_xs = []
        for x, direction in zip(xs, self.choices):
            new_xs.append(self.reverse(x, direction))
        return new_xs

    def scan(self, x, direction='h'):
        """Apply a single scan direction to input.

        Args:
            x: [B, L, D] or [B, C, H, W] tensor
            direction: Scan direction (h, v, w2, w7, etc.)

        Returns:
            [B, D, L] tensor after scanning
        """
        H, W = self.token_size
        if len(x.shape) == 3:
            if direction == 'h':
                return x.transpose(-2, -1)
            elif direction == 'h_flip':
                return x.transpose(-2, -1).flip([-1])
            elif direction == 'v':
                return rearrange(x, 'b (h w) d -> b d (w h)', h=H, w=W)
            elif direction == 'v_flip':
                return rearrange(x, 'b (h w) d -> b d (w h)', h=H, w=W).flip([-1])
            elif direction.startswith('w'):
                K = int(direction[1:].split('_')[0])
                flip = direction.endswith('flip')
                return local_scan(x, K, H, W, flip=flip)
            else:
                raise RuntimeError(f'Direction {direction} not found.')
        elif len(x.shape) == 4:
            if direction == 'h':
                return x.flatten(2)
            elif direction == 'h_flip':
                return x.flatten(2).flip([-1])
            elif direction == 'v':
                return rearrange(x, 'b d h w -> b d (w h)', h=H, w=W)
            elif direction == 'v_flip':
                return rearrange(x, 'b d h w -> b d (w h)', h=H, w=W).flip([-1])
            elif direction.startswith('w'):
                K = int(direction[1:].split('_')[0])
                flip = direction.endswith('flip')
                return local_scan_bchw(x, K, H, W, flip=flip)
            else:
                raise RuntimeError(f'Direction {direction} not found.')

    def reverse(self, x, direction='h'):
        """Reverse a single scan direction.

        Args:
            x: [B, D, L] tensor
            direction: Scan direction to reverse

        Returns:
            [B, D, L] tensor after reversing
        """
        H, W = self.token_size
        if direction == 'h':
            return x
        elif direction == 'h_flip':
            return x.flip([-1])
        elif direction == 'v':
            return rearrange(x, 'b d (h w) -> b d (w h)', h=H, w=W)
        elif direction == 'v_flip':
            return rearrange(x.flip([-1]), 'b d (h w) -> b d (w h)', h=H, w=W)
        elif direction.startswith('w'):
            K = int(direction[1:].split('_')[0])
            flip = direction.endswith('flip')
            return local_reverse(x, K, H, W, flip=flip)
        else:
            raise RuntimeError(f'Direction {direction} not found.')

    def __repr__(self):
        scans = ', '.join(self.choices)
        return super().__repr__().replace('MultiScanVSSM', f'MultiScanVSSM[{scans}]')


class PatchMerging2D(nn.Module):
    """Patch merging layer for downsampling.

    Downsamples by a factor of 2 using 2x2 patch merging.
    """

    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class Permute(nn.Module):
    """Helper module for permuting tensor dimensions."""

    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class VSSBlock(nn.Module):
    """Vision State Space Block.

    Basic building block of LocalVMamba consisting of:
    - LayerNorm
    - SS2D (Selective Scan 2D)
    - Residual connection with DropPath
    """

    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: str = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_simple_init=False,
        # =============================
        use_checkpoint: bool = False,
        directions=None,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm = norm_layer(hidden_dim)
        self.op = SS2D(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            # ==========================
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            # ==========================
            dropout=ssm_drop_rate,
            # ==========================
            simple_init=ssm_simple_init,
            # ==========================
            directions=directions
        )
        # Import DropPath from timm if available, otherwise use Identity
        try:
            from timm.models.layers import DropPath
            self.drop_path = DropPath(drop_path)
        except ImportError:
            # Fallback: simple DropPath implementation
            if drop_path > 0:
                self.drop_path = StochasticDepth(drop_path)
            else:
                self.drop_path = nn.Identity()

    def _forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.op(self.norm(input)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            import torch.utils.checkpoint as checkpoint
            return checkpoint.checkpoint(self._forward, input, use_reentrant=False)
        else:
            return self._forward(input)


class StochasticDepth(nn.Module):
    """Simple implementation of stochastic depth (DropPath)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output
