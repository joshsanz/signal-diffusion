"""Memory profiling utilities for debugging OOM issues."""
from __future__ import annotations

import gc
from typing import Any, Optional
import torch
from signal_diffusion.log_setup import get_logger

LOGGER = get_logger(__name__)


class MemoryProfiler:
    """Track memory usage at different points in training."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.baseline_allocated = 0
        self.baseline_reserved = 0

    def log_memory(self, label: str, force: bool = False) -> dict[str, float]:
        """Log current memory usage with a descriptive label.

        Args:
            label: Description of the current operation
            force: Force logging even if profiler is disabled

        Returns:
            Dictionary with memory statistics in MB
        """
        if not self.enabled and not force:
            return {}

        if not torch.cuda.is_available():
            return {}

        # Get memory stats
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # MB
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)

        # Calculate deltas from baseline
        delta_allocated = allocated - self.baseline_allocated
        delta_reserved = reserved - self.baseline_reserved

        stats = {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "max_allocated_mb": max_allocated,
            "max_reserved_mb": max_reserved,
            "delta_allocated_mb": delta_allocated,
            "delta_reserved_mb": delta_reserved,
        }

        LOGGER.info(
            "[%s] Memory: allocated=%.1f MB (Δ%+.1f), reserved=%.1f MB (Δ%+.1f), "
            "max_allocated=%.1f MB, max_reserved=%.1f MB",
            label,
            allocated,
            delta_allocated,
            reserved,
            delta_reserved,
            max_allocated,
            max_reserved,
        )

        return stats

    def set_baseline(self, label: str = "baseline") -> None:
        """Set the current memory state as baseline for delta calculations."""
        if not self.enabled or not torch.cuda.is_available():
            return

        self.baseline_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        self.baseline_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        LOGGER.info(
            "[%s] Set memory baseline: allocated=%.1f MB, reserved=%.1f MB",
            label,
            self.baseline_allocated,
            self.baseline_reserved,
        )

    def reset_peak_stats(self) -> None:
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def profile_model(self, model: torch.nn.Module, label: str) -> dict[str, float]:
        """Profile memory usage of a specific model.

        Args:
            model: The model to profile
            label: Description for logging

        Returns:
            Dictionary with model memory statistics
        """
        if not self.enabled or not torch.cuda.is_available():
            return {}

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Estimate parameter memory
        # This is approximate - actual memory may be higher due to buffers
        param_memory_mb = 0.0
        for p in model.parameters():
            bytes_per_elem = p.element_size()
            param_memory_mb += (p.numel() * bytes_per_elem) / (1024 ** 2)

        # Check for buffers
        buffer_memory_mb = 0.0
        for buf in model.buffers():
            bytes_per_elem = buf.element_size()
            buffer_memory_mb += (buf.numel() * bytes_per_elem) / (1024 ** 2)

        stats = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "param_memory_mb": param_memory_mb,
            "buffer_memory_mb": buffer_memory_mb,
            "total_model_mb": param_memory_mb + buffer_memory_mb,
        }

        LOGGER.info(
            "[%s] Model: params=%s (trainable=%s), param_mem=%.1f MB, "
            "buffer_mem=%.1f MB, total=%.1f MB",
            label,
            f"{total_params:,}",
            f"{trainable_params:,}",
            param_memory_mb,
            buffer_memory_mb,
            param_memory_mb + buffer_memory_mb,
        )

        return stats

    def profile_optimizer(self, optimizer: torch.optim.Optimizer, label: str) -> dict[str, float]:
        """Profile memory usage of optimizer states.

        Args:
            optimizer: The optimizer to profile
            label: Description for logging

        Returns:
            Dictionary with optimizer memory statistics
        """
        if not self.enabled or not torch.cuda.is_available():
            return {}

        # Estimate optimizer state memory
        # For AdamW: 2 states per parameter (momentum and variance)
        state_memory_mb = 0.0
        num_params_with_state = 0

        for group in optimizer.param_groups:
            for p in group['params']:
                if p in optimizer.state:
                    num_params_with_state += 1
                    state = optimizer.state[p]
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            bytes_per_elem = value.element_size()
                            state_memory_mb += (value.numel() * bytes_per_elem) / (1024 ** 2)

        stats = {
            "num_params_with_state": num_params_with_state,
            "optimizer_state_mb": state_memory_mb,
        }

        LOGGER.info(
            "[%s] Optimizer: params_with_state=%d, state_mem=%.1f MB",
            label,
            num_params_with_state,
            state_memory_mb,
        )

        return stats

    def profile_ema_model(self, ema_model: Any, label: str) -> dict[str, float]:
        """Profile memory usage of EMA model.

        Args:
            ema_model: The EMA model wrapper
            label: Description for logging

        Returns:
            Dictionary with EMA memory statistics
        """
        if not self.enabled or not torch.cuda.is_available():
            return {}

        # EMA stores shadow parameters
        shadow_memory_mb = 0.0
        num_shadow_params = 0

        if hasattr(ema_model, 'shadow_params'):
            for param in ema_model.shadow_params:
                if isinstance(param, torch.Tensor):
                    num_shadow_params += 1
                    bytes_per_elem = param.element_size()
                    shadow_memory_mb += (param.numel() * bytes_per_elem) / (1024 ** 2)

        stats = {
            "num_shadow_params": num_shadow_params,
            "ema_shadow_mb": shadow_memory_mb,
        }

        LOGGER.info(
            "[%s] EMA: shadow_params=%d, shadow_mem=%.1f MB",
            label,
            num_shadow_params,
            shadow_memory_mb,
        )

        return stats

    def garbage_collect(self, label: str = "gc") -> None:
        """Force garbage collection and CUDA cache clearing.

        Args:
            label: Description for logging
        """
        if not self.enabled:
            return

        mem_before = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        mem_after = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
        freed = mem_before - mem_after

        LOGGER.info(
            "[%s] GC: freed %.1f MB (before=%.1f MB, after=%.1f MB)",
            label,
            freed,
            mem_before,
            mem_after,
        )

    def comprehensive_profile(
        self,
        label: str,
        model: Optional[torch.nn.Module] = None,
        ema_model: Optional[Any] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        vae: Optional[torch.nn.Module] = None,
        text_encoder: Optional[torch.nn.Module] = None,
    ) -> dict[str, Any]:
        """Run a comprehensive memory profile.

        Args:
            label: Description for this profiling point
            model: Optional model (denoiser) to profile
            ema_model: Optional EMA model to profile
            optimizer: Optional optimizer to profile
            vae: Optional VAE model to profile
            text_encoder: Optional text encoder to profile

        Returns:
            Dictionary with all memory statistics
        """
        if not self.enabled:
            return {}

        LOGGER.info("=" * 80)
        LOGGER.info("COMPREHENSIVE MEMORY PROFILE: %s", label)
        LOGGER.info("=" * 80)

        stats = {}

        # Overall memory
        stats["overall"] = self.log_memory(f"{label}/overall")

        # Model memory (denoiser)
        if model is not None:
            stats["model"] = self.profile_model(model, f"{label}/denoiser")

        # VAE memory
        if vae is not None:
            stats["vae"] = self.profile_model(vae, f"{label}/vae")

        # Text encoder memory
        if text_encoder is not None:
            stats["text_encoder"] = self.profile_model(text_encoder, f"{label}/text_encoder")

        # EMA memory
        if ema_model is not None:
            stats["ema"] = self.profile_ema_model(ema_model, f"{label}/ema")

        # Optimizer memory
        if optimizer is not None:
            stats["optimizer"] = self.profile_optimizer(optimizer, f"{label}/optimizer")

        LOGGER.info("=" * 80)

        return stats
