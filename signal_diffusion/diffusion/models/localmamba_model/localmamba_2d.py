"""LocalMamba 2D diffusion model.

Hierarchical LocalVMamba architecture for diffusion models, similar to Hourglass
but using selective scan (Mamba) blocks instead of transformer attention.

Architecture:
- Patch embedding (TokenMerge)
- Time/noise embedding (FourierFeatures + MappingNetwork)
- Hierarchical encoder-decoder with VSSBlocks
- U-Net skip connections
- Class/caption conditioning support

Ported from LocalVMamba and adapted for diffusion modeling.
"""

from dataclasses import dataclass
from functools import reduce
import math

from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F


# ============================================================================
# Helper functions (from Hourglass)
# ============================================================================

def zero_init(layer):
    """Initialize layer weights and bias to zero."""
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def apply_wd(module):
    """Tag weights for weight decay."""
    for name, param in module.named_parameters():
        if name.endswith("weight"):
            if not hasattr(param, "_tags"):
                param._tags = set(["wd"])
            else:
                param._tags.add("wd")
    return module


def tag_module(module, tag):
    """Tag all parameters in module."""
    for param in module.parameters():
        if not hasattr(param, "_tags"):
            param._tags = set([tag])
        else:
            param._tags.add(tag)
    return module


def filter_params(function, module):
    """Filter parameters by tag predicate."""
    for param in module.parameters():
        tags = getattr(param, "_tags", set())
        if function(tags):
            yield param


def downscale_pos(pos):
    """Downscale positional embeddings by 2x2."""
    pos = rearrange(pos, "... (h nh) (w nw) e -> ... h w (nh nw) e", nh=2, nw=2)
    return torch.mean(pos, dim=-2)


# ============================================================================
# Normalization layers
# ============================================================================

def rms_norm(x, scale, eps):
    """RMS normalization kernel."""
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype)**2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))

    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)


class AdaRMSNorm(nn.Module):
    """Adaptive RMS normalization with conditioning."""

    def __init__(self, features, cond_features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.linear = apply_wd(zero_init(nn.Linear(cond_features, features, bias=False)))
        tag_module(self.linear, "mapping")

    def extra_repr(self):
        return f"eps={self.eps},"

    def forward(self, x, cond):
        # x: (B, H, W, C), cond: (B, cond_features)
        return rms_norm(x, self.linear(cond)[:, None, None, :] + 1, self.eps)


# ============================================================================
# Fourier Features & Positional Embeddings
# ============================================================================

class FourierFeatures(nn.Module):
    """Random Fourier feature embeddings."""

    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer('weight', torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


def centers(start, stop, num, dtype=None, device=None):
    """Get centers of evenly spaced bins."""
    edges = torch.linspace(start, stop, num + 1, dtype=dtype, device=device)
    return (edges[:-1] + edges[1:]) / 2


def make_grid(h_pos, w_pos):
    """Create 2D grid from height and width positions."""
    grid = torch.stack(torch.meshgrid(h_pos, w_pos, indexing='ij'), dim=-1)
    h, w, d = grid.shape
    return grid.view(h * w, d)


def bounding_box(h, w, pixel_aspect_ratio=1.0):
    """Compute bounding box for positional embeddings."""
    w_adj = w
    h_adj = h * pixel_aspect_ratio
    ar_adj = w_adj / h_adj

    y_min, y_max, x_min, x_max = -1.0, 1.0, -1.0, 1.0
    if ar_adj > 1:
        y_min, y_max = -1 / ar_adj, 1 / ar_adj
    elif ar_adj < 1:
        x_min, x_max = -ar_adj, ar_adj

    return y_min, y_max, x_min, x_max


def make_axial_pos(h, w, pixel_aspect_ratio=1.0, align_corners=False, dtype=None, device=None):
    """Create axial positional embeddings."""
    y_min, y_max, x_min, x_max = bounding_box(h, w, pixel_aspect_ratio)
    if align_corners:
        h_pos = torch.linspace(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = torch.linspace(x_min, x_max, w, dtype=dtype, device=device)
    else:
        h_pos = centers(y_min, y_max, h, dtype=dtype, device=device)
        w_pos = centers(x_min, x_max, w, dtype=dtype, device=device)
    return make_grid(h_pos, w_pos)


# ============================================================================
# Mapping Network
# ============================================================================

class LinearGEGLU(nn.Linear):
    """Linear layer with GELU gating."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features

    def forward(self, input):
        x = super().forward(input)
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class MappingFeedForwardBlock(nn.Module):
    """Feed-forward block for mapping network."""

    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.up_proj = apply_wd(LinearGEGLU(d_model, d_ff, bias=False))
        self.dropout = nn.Dropout(dropout)
        self.down_proj = apply_wd(zero_init(nn.Linear(d_ff, d_model, bias=False)))

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


class MappingNetwork(nn.Module):
    """Mapping network for conditioning signals."""

    def __init__(self, n_layers, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.in_norm = RMSNorm(d_model)
        self.blocks = nn.ModuleList([
            MappingFeedForwardBlock(d_model, d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.out_norm = RMSNorm(d_model)

    def forward(self, x):
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x


# ============================================================================
# Token Merging and Splitting
# ============================================================================

class TokenMerge(nn.Module):
    """Merge tokens (patch embedding) with learnable projection."""

    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = apply_wd(nn.Linear(in_features * self.h * self.w, out_features, bias=False))

    def forward(self, x):
        # x: (B, H, W, C) -> (B, H/h, W/w, out_features)
        x = rearrange(x, "... (h nh) (w nw) e -> ... h w (nh nw e)", nh=self.h, nw=self.w)
        return self.proj(x)


class TokenSplitWithoutSkip(nn.Module):
    """Split tokens (patch unprojection) without skip connection."""

    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = apply_wd(nn.Linear(in_features, out_features * self.h * self.w, bias=False))

    def forward(self, x):
        # x: (B, H, W, C) -> (B, H*h, W*w, out_features)
        x = self.proj(x)
        return rearrange(x, "... h w (nh nw e) -> ... (h nh) (w nw) e", nh=self.h, nw=self.w)


class TokenSplit(nn.Module):
    """Split tokens with learnable skip connection mixing."""

    def __init__(self, in_features, out_features, patch_size=(2, 2)):
        super().__init__()
        self.h = patch_size[0]
        self.w = patch_size[1]
        self.proj = apply_wd(nn.Linear(in_features, out_features * self.h * self.w, bias=False))
        self.fac = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x, skip):
        # x: (B, H, W, C_in), skip: (B, H*h, W*w, C_out)
        x = self.proj(x)
        x = rearrange(x, "... h w (nh nw e) -> ... (h nh) (w nw) e", nh=self.h, nw=self.w)
        return torch.lerp(skip, x, self.fac.to(x.dtype))


# ============================================================================
# Conditioned VSSBlock
# ============================================================================

class ConditionedVSSBlock(nn.Module):
    """VSSBlock with AdaRMSNorm conditioning.

    Wraps VSSBlock to replace LayerNorm with AdaRMSNorm for time/class conditioning.
    """

    def __init__(
        self,
        hidden_dim: int,
        cond_features: int,
        drop_path: float = 0,
        ssm_d_state: int = 16,
        ssm_ratio: float = 2.0,
        ssm_dt_rank: str = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_simple_init=False,
        use_checkpoint: bool = False,
        directions=None,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # Use AdaRMSNorm instead of LayerNorm
        self.norm = AdaRMSNorm(hidden_dim, cond_features)

        # Import SS2D here to avoid circular import
        from .ss2d import SS2D

        self.op = SS2D(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            dropout=ssm_drop_rate,
            simple_init=ssm_simple_init,
            directions=directions,
        )

        # DropPath (Stochastic Depth)
        try:
            from timm.models.layers import DropPath
            self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        except ImportError:
            from .vmamba_blocks import StochasticDepth
            self.drop_path = StochasticDepth(drop_path) if drop_path > 0 else nn.Identity()

    def _forward(self, input: torch.Tensor, cond: torch.Tensor):
        # input: (B, H, W, C), cond: (B, cond_features)
        x = input + self.drop_path(self.op(self.norm(input, cond)))
        return x

    def forward(self, input: torch.Tensor, pos: torch.Tensor, cond: torch.Tensor):
        # pos is not used by Mamba but kept for interface compatibility
        if self.use_checkpoint:
            import torch.utils.checkpoint as checkpoint
            return checkpoint.checkpoint(self._forward, input, cond, use_reentrant=False)
        else:
            return self._forward(input, cond)


class Level(nn.ModuleList):
    """Sequential container for layers at a hierarchical level."""

    def forward(self, x, pos, cond):
        for layer in self:
            x = layer(x, pos, cond)
        return x


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class LevelSpec:
    """Specification for a hierarchical level."""
    depth: int       # Number of VSSBlocks
    width: int       # Hidden dimension
    d_state: int     # SSM state dimension
    ssm_ratio: float # SSM expansion ratio
    ssm_dt_rank: str # dt rank ("auto" or int)
    drop_path: float # Stochastic depth probability
    dropout: float   # Dropout rate
    directions: list[str] = None  # Scan directions (e.g., ["h", "v", "w7"])


@dataclass
class MappingSpec:
    """Specification for mapping network."""
    depth: int    # Number of layers
    width: int    # Hidden dimension
    d_ff: int     # Feed-forward dimension
    dropout: float


# ============================================================================
# LocalMamba2D Model
# ============================================================================

class LocalMamba2DModel(nn.Module):
    """Hierarchical LocalMamba model for diffusion.

    U-Net-style architecture with:
    - Patch-based input/output
    - Multi-directional selective scanning (Mamba blocks)
    - Time/class/caption conditioning
    - Skip connections between encoder and decoder

    Args:
        levels: List of LevelSpec defining hierarchical architecture
        mapping: MappingSpec for conditioning network
        in_channels: Input image channels
        out_channels: Output image channels
        patch_size: Patch size for token merging/splitting
        num_classes: Number of classes for class conditioning (0 = no class conditioning)
        mapping_cond_dim: Additional conditioning dimension (e.g., for captions)
    """

    def __init__(
        self,
        levels: list[LevelSpec],
        mapping: MappingSpec,
        in_channels: int,
        out_channels: int,
        patch_size: tuple[int, int] = (2, 2),
        num_classes: int = 0,
        mapping_cond_dim: int = 0,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Patch embedding
        self.patch_in = TokenMerge(in_channels, levels[0].width, patch_size)

        # Time embedding (sigma/noise level)
        self.time_emb = FourierFeatures(1, mapping.width)
        self.time_in_proj = nn.Linear(mapping.width, mapping.width, bias=False)

        # Augmentation conditioning (placeholder, can be used for data augmentation params)
        self.aug_emb = FourierFeatures(9, mapping.width)
        self.aug_in_proj = nn.Linear(mapping.width, mapping.width, bias=False)

        # Class embedding
        self.class_emb = nn.Embedding(num_classes, mapping.width) if num_classes else None

        # Additional conditioning (e.g., text embeddings)
        self.mapping_cond_in_proj = (
            nn.Linear(mapping_cond_dim, mapping.width, bias=False)
            if mapping_cond_dim else None
        )

        # Mapping network combines all conditioning signals
        self.mapping = tag_module(
            MappingNetwork(mapping.depth, mapping.width, mapping.d_ff, dropout=mapping.dropout),
            "mapping"
        )

        # Build hierarchical levels
        self.down_levels = nn.ModuleList()
        self.up_levels = nn.ModuleList()

        for i, spec in enumerate(levels):
            if i < len(levels) - 1:
                # Encoder and decoder levels
                self.down_levels.append(Level([
                    ConditionedVSSBlock(
                        hidden_dim=spec.width,
                        cond_features=mapping.width,
                        drop_path=spec.drop_path,
                        ssm_d_state=spec.d_state,
                        ssm_ratio=spec.ssm_ratio,
                        ssm_dt_rank=spec.ssm_dt_rank,
                        ssm_drop_rate=spec.dropout,
                        directions=spec.directions,
                    )
                    for _ in range(spec.depth)
                ]))

                self.up_levels.append(Level([
                    ConditionedVSSBlock(
                        hidden_dim=spec.width,
                        cond_features=mapping.width,
                        drop_path=spec.drop_path,
                        ssm_d_state=spec.d_state,
                        ssm_ratio=spec.ssm_ratio,
                        ssm_dt_rank=spec.ssm_dt_rank,
                        ssm_drop_rate=spec.dropout,
                        directions=spec.directions,
                    )
                    for _ in range(spec.depth)
                ]))
            else:
                # Middle level (bottleneck)
                self.mid_level = Level([
                    ConditionedVSSBlock(
                        hidden_dim=spec.width,
                        cond_features=mapping.width,
                        drop_path=spec.drop_path,
                        ssm_d_state=spec.d_state,
                        ssm_ratio=spec.ssm_ratio,
                        ssm_dt_rank=spec.ssm_dt_rank,
                        ssm_drop_rate=spec.dropout,
                        directions=spec.directions,
                    )
                    for _ in range(spec.depth)
                ])

        # Downsampling and upsampling layers
        self.merges = nn.ModuleList([
            TokenMerge(spec_1.width, spec_2.width, patch_size=(2, 2))
            for spec_1, spec_2 in zip(levels[:-1], levels[1:])
        ])

        self.splits = nn.ModuleList([
            TokenSplit(spec_2.width, spec_1.width, patch_size=(2, 2))
            for spec_1, spec_2 in zip(levels[:-1], levels[1:])
        ])

        # Output projection
        self.out_norm = RMSNorm(levels[0].width)
        self.patch_out = TokenSplitWithoutSkip(levels[0].width, out_channels, patch_size)
        nn.init.zeros_(self.patch_out.proj.weight)

        # Final 3x3 conv to smooth pixel outputs (initialized as no-op).
        self.output_smoother = apply_wd(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)
        )
        nn.init.zeros_(self.output_smoother.weight)
        nn.init.zeros_(self.output_smoother.bias)

    def param_groups(self, base_lr=5e-4, mapping_lr_scale=1/3):
        """Create parameter groups for optimizer with different learning rates."""
        wd = filter_params(lambda tags: "wd" in tags and "mapping" not in tags, self)
        no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" not in tags, self)
        mapping_wd = filter_params(lambda tags: "wd" in tags and "mapping" in tags, self)
        mapping_no_wd = filter_params(lambda tags: "wd" not in tags and "mapping" in tags, self)

        groups = [
            {"params": list(wd), "lr": base_lr},
            {"params": list(no_wd), "lr": base_lr, "weight_decay": 0.0},
            {"params": list(mapping_wd), "lr": base_lr * mapping_lr_scale},
            {"params": list(mapping_no_wd), "lr": base_lr * mapping_lr_scale, "weight_decay": 0.0}
        ]
        return groups

    def forward(self, x, sigma, aug_cond=None, class_cond=None, mapping_cond=None):
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W)
            sigma: Noise level (B,) or (B, 1)
            aug_cond: Augmentation conditioning (B, 9) [optional]
            class_cond: Class labels (B,) [optional, required if num_classes > 0]
            mapping_cond: Additional conditioning (B, mapping_cond_dim) [optional]

        Returns:
            Output tensor (B, C, H, W)
        """
        # Patching: (B, C, H, W) -> (B, H', W', width)
        x = x.movedim(-3, -1)  # (B, H, W, C)
        x = self.patch_in(x)   # (B, H/patch, W/patch, width)

        # Positional embeddings
        pos = make_axial_pos(x.shape[-3], x.shape[-2], device=x.device)
        pos = pos.view(x.shape[-3], x.shape[-2], 2)

        # Mapping network: combine all conditioning signals
        # When the architecture was instantiated with class or extra-conditioning
        # heads we expect callers to feed corresponding tensors. Failing early
        # keeps the forward pass easier to reason about than silently broadcasting.
        if class_cond is None and self.class_emb is not None:
            raise ValueError("class_cond must be specified if num_classes > 0")
        if mapping_cond is None and self.mapping_cond_in_proj is not None:
            raise ValueError("mapping_cond must be specified if mapping_cond_dim > 0")

        # Time conditioning
        c_noise = torch.log(sigma) / 4
        time_emb = self.time_in_proj(self.time_emb(c_noise[..., None]))

        # Augmentation conditioning
        # Default to zero augmentation signal so downstream math stays uniform
        # regardless of whether Karras-style augmentation metadata is provided.
        aug_cond = x.new_zeros([x.shape[0], 9]) if aug_cond is None else aug_cond
        aug_emb = self.aug_in_proj(self.aug_emb(aug_cond))

        # Class conditioning
        class_emb = self.class_emb(class_cond) if self.class_emb is not None else 0

        # Additional conditioning
        mapping_emb = (
            self.mapping_cond_in_proj(mapping_cond)
            if self.mapping_cond_in_proj is not None else 0
        )

        # Combine time/augmentation/class/text embeddings through the shared
        # mapping network so every block receives the same fused context.
        cond = self.mapping(time_emb + aug_emb + class_emb + mapping_emb)

        # U-Net encoder with skip connections
        skips, poses = [], []
        for down_level, merge in zip(self.down_levels, self.merges):
            x = down_level(x, pos, cond)
            skips.append(x)
            poses.append(pos)
            x = merge(x)
            pos = downscale_pos(pos)

        # Middle level (bottleneck)
        x = self.mid_level(x, pos, cond)

        # U-Net decoder with skip connections
        for up_level, split, skip, pos in reversed(list(zip(
            self.up_levels, self.splits, skips, poses
        ))):
            x = split(x, skip)
            x = up_level(x, pos, cond)

        # Unpatching: (B, H', W', width) -> (B, C, H, W)
        x = self.out_norm(x)
        x = self.patch_out(x)
        x = x.movedim(-1, -3)

        # Apply smoothing conv residual to avoid altering pretrained behavior.
        x = x + self.output_smoother(x)

        return x
