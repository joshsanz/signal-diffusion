"""Functions for computing dataset fidelity metrics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping
import os
import shutil

import torch_fidelity as tfid
from torch.utils.data import Dataset


FEATURE_EXTRACTORS = ("dinov2-vit-l-14", "clip-vit-l-14")


@dataclass(frozen=True)
class FidelityConfig:
    """Arguments controlling KID/PRC evaluation."""

    kid_subset_size: int = 1000
    batch_size: int = 64
    cache_prefix: str | None = None
    cache_results: bool = False


def _resolve_cache_name(prefix: str | None, dataset_name: str) -> str:
    if prefix in (None, ""):
        return dataset_name
    return f"{prefix}-{dataset_name}"


def calculate_pair_metrics(
    real: Dataset,
    generated: Dataset,
    *,
    feature_extractor: str,
    config: FidelityConfig,
    real_dataset_name: str,
    generated_dataset_name: str,
) -> Mapping[str, float]:
    """Run torch-fidelity for a single feature extractor."""

    real_cache = None
    generated_cache = None
    if config.cache_results:
        real_cache = _resolve_cache_name(config.cache_prefix, real_dataset_name)
        generated_cache = _resolve_cache_name(config.cache_prefix, generated_dataset_name)

    metrics = tfid.calculate_metrics(
        isc=False,
        fid=False,
        kid=True,
        prc=True,
        ppl=False,
        kid_kernel="rbf",
        kid_kernel_rbf_sigma=10.0,
        prc_neighborhood=10,
        kid_subset_size=config.kid_subset_size,
        feature_extractor=feature_extractor,
        batch_size=config.batch_size,
        input1=real,
        input2=generated,
        input1_cache_name=real_cache,
        input2_cache_name=generated_cache,
    )
    return metrics


def calculate_metrics_for_extractors(
    real: Dataset,
    generated: Dataset,
    *,
    extractors: Iterable[str] = FEATURE_EXTRACTORS,
    config: FidelityConfig | None = None,
    real_dataset_name: str,
    generated_dataset_name: str,
) -> dict[str, Mapping[str, float]]:
    """Evaluate a pair of datasets across multiple feature extractors."""

    cfg = config or FidelityConfig()
    results: dict[str, Mapping[str, float]] = {}
    for extractor in extractors:
        metrics = calculate_pair_metrics(
            real,
            generated,
            feature_extractor=extractor,
            config=cfg,
            real_dataset_name=real_dataset_name,
            generated_dataset_name=generated_dataset_name,
        )
        results[extractor] = metrics
    return results


def clear_fidelity_cache(dataset_name: str) -> None:
    """Remove cached features for a dataset if present."""

    cache_root = _resolve_torch_home() / "fidelity_datasets"
    shutil.rmtree(cache_root / dataset_name, ignore_errors=True)


def _resolve_torch_home() -> Path:
    env_home = os.getenv("ENV_TORCH_HOME")
    torch_home = os.getenv("TORCH_HOME")
    if env_home:
        return Path(env_home)
    if torch_home:
        return Path(torch_home)
    return Path.home() / ".cache" / "torch"
