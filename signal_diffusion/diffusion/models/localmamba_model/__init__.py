"""
LocalMamba model components for diffusion.

This package contains the ported LocalVMamba implementation adapted for
diffusion model training, updated for mamba-ssm v2.0 compatibility.
"""

from .ss2d import SS2D, BiAttn, multi_selective_scan
from .vmamba_blocks import VSSBlock, PatchMerging2D, MultiScanVSSM
from .local_scan import LocalScanTriton, LocalReverseTriton, local_scan, local_reverse
from .localmamba_2d import (
    LocalMamba2DModel,
    LevelSpec,
    MappingSpec,
    ConditionedVSSBlock,
    TokenMerge,
    TokenSplit,
    TokenSplitWithoutSkip,
)

__all__ = [
    # Core SSM components
    "SS2D",
    "BiAttn",
    "multi_selective_scan",
    # Building blocks
    "VSSBlock",
    "PatchMerging2D",
    "MultiScanVSSM",
    # Local scan operations
    "LocalScanTriton",
    "LocalReverseTriton",
    "local_scan",
    "local_reverse",
    # Diffusion model
    "LocalMamba2DModel",
    "LevelSpec",
    "MappingSpec",
    "ConditionedVSSBlock",
    # Token operations
    "TokenMerge",
    "TokenSplit",
    "TokenSplitWithoutSkip",
]
