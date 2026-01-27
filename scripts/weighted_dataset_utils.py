"""Shared utilities for weighted dataset generation scripts."""
from __future__ import annotations

import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from signal_diffusion.data.metadata_utils import (
    DEFAULT_HEALTH,
    build_caption,
    normalize_age as _try_int_age,
    normalize_gender as _normalise_gender,
    normalize_health as _normalise_health,
)
from signal_diffusion.log_setup import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class DatasetStats:
    """Simple container for per-dataset accounting."""

    name: str
    original_samples: int
    generated_samples: int = 0


@dataclass(slots=True)
class WeightStats:
    """Track how many samples share a weight and how many copies were produced."""

    weight: float
    source_count: int = 0
    generated_copies: int = 0


# Metadata normalization and caption building functions now imported from
# signal_diffusion.data.metadata_utils (see imports at top of file)


def enrich_metadata(
    row: pd.Series,
    *,
    split: str,
    modality: str = "spectrogram image",
    subject_noun: str = "subject",
) -> pd.Series:
    metadata = row.to_dict()
    file_name = metadata["file_name"]

    gender_code = _normalise_gender(metadata.get("gender"))
    health_code = _normalise_health(metadata.get("health", DEFAULT_HEALTH))
    age_value = _try_int_age(metadata.get("age"))

    serialised: dict[str, Any] = {
        "file_name": file_name,
        "split": split,
        "gender": gender_code if gender_code is not None else pd.NA,
        "health": health_code,
        "age": age_value if age_value is not None else pd.NA,
    }
    serialised["caption"] = build_caption(serialised, modality=modality, subject_noun=subject_noun)
    return pd.Series(serialised)


def parse_splits(raw: str) -> dict[str, float]:
    """Parse comma-separated split specifications into a dictionary."""
    result: dict[str, float] = {}
    for item in raw.split(","):
        if not item:
            continue
        try:
            split, value = item.split(":", maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"Invalid split spec '{item}'. Expected format name:value") from exc
        result[split.strip()] = float(value)
    total = sum(result.values())
    if not math.isclose(total, 1.0, rel_tol=1e-3):
        raise ValueError(f"Split fractions must sum to 1.0 (received {total:.3f}).")
    return result


def parse_copy_splits(raw: str) -> tuple[str, ...]:
    splits = tuple(split.strip() for split in raw.split(",") if split.strip())
    if not splits:
        raise ValueError("--copy-splits must include at least one split name.")
    return splits


def validate_splits_compatibility(
    preprocess: bool,
    splits: dict[str, float] | None,
    copy_splits: tuple[str, ...],
) -> None:
    if not preprocess or splits is None:
        return

    preprocessed_split_names = set(splits.keys())
    copy_split_names = set(copy_splits)

    missing_splits = copy_split_names - preprocessed_split_names
    if missing_splits:
        logger.warning(
            "Requested copy-splits %s but preprocessing only creates splits %s. "
            "The following splits will not be found: %s. "
            "Update --copy-splits to match --splits, or remove unused splits.",
            copy_splits,
            tuple(preprocessed_split_names),
            tuple(missing_splits),
        )

    unused_splits = preprocessed_split_names - copy_split_names
    if unused_splits:
        logger.warning(
            "Preprocessing will create splits %s but copy-splits is %s. "
            "The following splits will be created but not copied to output: %s. "
            "Update --copy-splits to include these splits, or they will be wasted.",
            tuple(preprocessed_split_names),
            copy_splits,
            tuple(unused_splits),
        )


def prepare_output_dir(output_dir: Path, *, overwrite: bool, force: bool = False) -> None:
    """Prepare the output directory, optionally removing existing directory if overwrite is enabled.

    Args:
        output_dir: Target directory path
        overwrite: Whether to overwrite if directory exists
        force: If True, skip confirmation prompt when overwriting
    """
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
            )
        if not force:
            response = input(
                f"Output directory already exists at {output_dir}.\n"
                "This will delete all existing files. Continue? [y/N]: "
            ).strip().lower()
            if response not in {"y", "yes"}:
                logger.info("User declined to overwrite %s; aborting.", output_dir)
                sys.exit(0)
        logger.info("Removing existing output directory at %s", output_dir)
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Created output directory at %s", output_dir)


def set_random_seeds(seed: int) -> None:
    """Set random seeds - use signal_diffusion.utils.set_random_seeds instead."""
    # Import here to avoid circular dependencies during transition
    from signal_diffusion.utils import set_random_seeds as _set_seeds
    _set_seeds(seed)


def compute_scaled_weights(weights: torch.Tensor) -> torch.Tensor:
    """Scale sampler weights so the minimum weight becomes 1.0 for copy count computation."""
    scaled = weights.detach().cpu().double()
    if scaled.numel() == 0:
        raise RuntimeError("Sampler produced no weights; ensure datasets contain samples.")
    min_weight = float(torch.min(scaled))
    if min_weight <= 0:
        raise RuntimeError("Sampler weights must be positive to compute copy counts.")
    return scaled / min_weight


def scale_numpy_weights(weights: np.ndarray) -> np.ndarray:
    if weights.size == 0:
        return weights
    min_weight = float(np.min(weights))
    if min_weight <= 0:
        raise ValueError("Weights must be positive to compute copy counts.")
    return weights / min_weight


def assign_copies(weight: float, count: int) -> list[int]:
    """
    Distributes a weighted value across a specified number of items by assigning
    integer copies to each item. The fractional part of the weight is distributed
    as evenly as possible by adding extra copies to some items.
    """
    base = math.floor(weight)
    if base <= 0:
        raise ValueError(f"Weight {weight:.6f} would yield zero copies; check sampler output.")
    fractional = max(weight - base, 0.0)
    copies: list[int] = []
    remainder = 0.0
    for _ in range(count):
        extra = 0
        if fractional > 0:
            remainder += fractional
            if remainder + 1e-8 >= 1.0:
                extra = 1
                remainder -= 1.0
        copies.append(base + extra)
    return copies


def save_weights_plot(
    scaled_weights: torch.Tensor | np.ndarray,
    output_dir: Path,
    *,
    split: str | None = None,
) -> Path:
    weights_np = scaled_weights.detach().cpu().numpy() if isinstance(scaled_weights, torch.Tensor) else scaled_weights
    plt.figure(figsize=(10, 4))
    plt.plot(weights_np)
    plt.title("Scaled Sample Weights")
    plt.xlabel("Sample Index")
    plt.ylabel("Weight (relative to minimum)")
    plt.tight_layout()
    filename = "weights.png" if split is None else f"{split}_weights.png"
    plot_path = output_dir / filename
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def compute_balanced_weights(
    metadata: pd.DataFrame,
    *,
    dataset_order: Sequence[str],
    max_sampling_weight: float | None = None,
) -> np.ndarray:
    """Compute per-row weights that balance datasets and gender."""
    if metadata.empty:
        return np.asarray([], dtype=np.float64)

    from signal_diffusion.data.meta import META_LABELS  # Local import to avoid cycles

    gender_spec = META_LABELS["gender"]
    encoded_gender = metadata.apply(lambda row: int(gender_spec.encode(row)), axis=1).to_numpy()

    dataset_indices: list[np.ndarray] = []
    female = 0
    male = 0
    totals: list[int] = []

    for name in dataset_order:
        mask = metadata["dataset"] == name
        if not mask.any():
            continue
        indices = np.flatnonzero(mask.to_numpy())
        dataset_indices.append(indices)
        genders = encoded_gender[indices]
        totals.append(len(indices))
        female += int((genders == 1).sum())
        male += int((genders == 0).sum())

    total_samps = sum(totals)
    if total_samps == 0:
        return np.asarray([], dtype=np.float64)

    gend_weights = [female / total_samps, male / total_samps]
    dataset_weights = [total / total_samps for total in totals]

    summed_weights: list[float] = []
    weights: list[list[float]] = []
    for dw in dataset_weights:
        data_gender_w = []
        for gw in gend_weights:
            data_gender_w.append(gw + dw)
        summed_weights.append(sum(data_gender_w))
        weights.append(data_gender_w)

    rankings: dict[int, int] = {}
    for label in range(len(dataset_indices)):
        label_weight = summed_weights[label]
        rank = 0
        for weight in summed_weights:
            if label_weight > weight:
                rank += 1
        rankings[label] = rank

    new_label_weights = [weights[(len(dataset_indices) - 1) - rankings[i]] for i in range(len(dataset_indices))]

    if max_sampling_weight is not None:
        capped: list[list[float]] = []
        for weight_pair in new_label_weights:
            capped.append([min(w, float(max_sampling_weight)) for w in weight_pair])
        new_label_weights = capped

    per_row = np.zeros(len(metadata), dtype=np.float64)
    for i, indices in enumerate(dataset_indices):
        female_weight = new_label_weights[i][0] if len(new_label_weights[i]) > 0 else 0.0
        male_weight = new_label_weights[i][1] if len(new_label_weights[i]) > 1 else 0.0
        genders = encoded_gender[indices]
        per_row[indices] = np.where(genders == 1, female_weight, male_weight)

    norm = float(per_row.sum())
    if norm <= 0:
        logger.warning("Computed weights sum to zero; falling back to uniform weights.")
        return np.full(len(per_row), 1.0 / max(len(per_row), 1), dtype=np.float64)

    return per_row / norm


def initialize_statistics_tracking() -> tuple[
    dict[str, DatasetStats],
    dict[str, dict[str, DatasetStats]],
    dict[float, WeightStats],
    dict[str, dict[float, WeightStats]],
    dict[str, Path],
    dict[str, Path],
]:
    """Initialize empty dictionaries for tracking dataset statistics.

    Returns:
        Tuple of (dataset_stats_total, dataset_stats_by_split,
                 weight_stats_total, weight_stats_by_split,
                 plot_paths, parquet_paths)
    """
    dataset_stats_total: dict[str, DatasetStats] = {}
    dataset_stats_by_split: dict[str, dict[str, DatasetStats]] = {}
    weight_stats_total: dict[float, WeightStats] = {}
    weight_stats_by_split: dict[str, dict[float, WeightStats]] = {}
    plot_paths: dict[str, Path] = {}
    parquet_paths: dict[str, Path] = {}
    return (
        dataset_stats_total,
        dataset_stats_by_split,
        weight_stats_total,
        weight_stats_by_split,
        plot_paths,
        parquet_paths,
    )


def aggregate_split_statistics(
    split_dataset_stats: dict[str, DatasetStats],
    split_weight_stats: dict[float, WeightStats],
    dataset_stats_total: dict[str, DatasetStats],
    weight_stats_total: dict[float, WeightStats],
) -> None:
    """Aggregate statistics from a single split into totals.

    Updates dataset_stats_total and weight_stats_total in-place by merging
    the statistics from a single split.

    Args:
        split_dataset_stats: Per-dataset statistics for the current split
        split_weight_stats: Per-weight statistics for the current split
        dataset_stats_total: Accumulated dataset statistics across all splits (modified in-place)
        weight_stats_total: Accumulated weight statistics across all splits (modified in-place)
    """
    for name, stats in split_dataset_stats.items():
        aggregate = dataset_stats_total.setdefault(
            name, DatasetStats(name=name, original_samples=0, generated_samples=0)
        )
        aggregate.original_samples += stats.original_samples
        aggregate.generated_samples += stats.generated_samples

    for weight, stats in split_weight_stats.items():
        aggregate_weight = weight_stats_total.setdefault(weight, WeightStats(weight=weight))
        aggregate_weight.source_count += stats.source_count
        aggregate_weight.generated_copies += stats.generated_copies


def log_completion_summary(
    dataset_stats_total: dict[str, DatasetStats],
    output_dir: Path,
    parquet_paths: dict[str, Path],
    plot_paths: dict[str, Path],
    copy_splits: tuple[str, ...],
    readme_path: Path,
) -> None:
    """Log final completion summary with paths and counts.

    Args:
        dataset_stats_total: Accumulated dataset statistics across all splits
        output_dir: Output directory where files were written
        parquet_paths: Dict mapping split name to parquet file path
        plot_paths: Dict mapping plot label to plot file path
        copy_splits: Splits that were processed
        readme_path: Path to generated README file
    """
    total_generated = sum(stats.generated_samples for stats in dataset_stats_total.values())
    logger.info("Generated %d samples into %s", total_generated, output_dir)
    for split in copy_splits:
        if split in parquet_paths:
            logger.info("Saved %s dataset to %s", split, parquet_paths[split])
        else:
            logger.info("No parquet dataset saved for split '%s'", split)
    if plot_paths:
        for label, path in plot_paths.items():
            logger.info("Saved weight diagnostics (%s) to %s", label, path)
    logger.info("Documented run in %s", readme_path)


def write_readme(
    output_dir: Path,
    settings: Any,
    *,
    dataset_stats: dict[str, DatasetStats],
    dataset_stats_by_split: dict[str, dict[str, DatasetStats]],
    weight_stats: dict[float, WeightStats],
    weight_stats_by_split: dict[str, dict[float, WeightStats]],
    copy_splits: tuple[str, ...],
    args: Any,
    modality: str,
    preprocessing_fields: Mapping[str, Any],
) -> tuple[Path, list[str]]:
    total_generated = sum(stats.generated_samples for stats in dataset_stats.values())
    task_list = getattr(args, "tasks", ())
    summary = (
        "This dataset merges the "
        + ", ".join(args.datasets)
        + f" {modality} corpora with tasks "
        + ", ".join(task_list)
        + f". The weighted copies cover splits {', '.join(copy_splits)} and yield {total_generated} total samples."
    )

    front_matter = [
        "---",
        f"pretty_name: Weighted Meta Dataset ({', '.join(args.datasets)})",
        "language:",
        "  - en",
        "license: unknown",
        "task_categories:",
        "  - classification",
        "tags:",
        "  - eeg",
        f"  - {modality}",
        "dataset_summary: |",
        f"  {summary}",
        "configurations:",
        "  - name: weighted_copies",
        "    description: Weighted copies derived from sampler weights",
        "splits:",
    ] + [f"  - {split}" for split in copy_splits] + [
        "preprocessing:",
    ]
    for key, value in preprocessing_fields.items():
        front_matter.append(f"  {key}: {value}")
    front_matter.extend(
        [
            f"  preprocess_ran: {str(getattr(args, 'preprocess', False)).lower()}",
            "generation:",
            f"  seed: {args.seed}",
            f"  total_generated_samples: {total_generated}",
            "---",
            "",
        ]
    )

    lines = front_matter + [
        "# Weighted Meta Dataset",
        "",
        f"Generated on: {pd.Timestamp.utcnow().isoformat()}",
        f"Source config: {settings.config_path}",
        "",
        "## Component Datasets",
        "| dataset | original samples | generated samples |",
        "| ------- | ---------------- | ----------------- |",
    ]
    for stats in sorted(dataset_stats.values(), key=lambda x: x.name):
        lines.append(f"| {stats.name} | {stats.original_samples} | {stats.generated_samples} |")

    lines.extend(
        [
            "",
            "## Weight Buckets (All Splits)",
            "| weight | source samples | generated copies |",
            "| ------ | -------------- | ---------------- |",
        ]
    )
    for weight, stats in sorted(weight_stats.items(), key=lambda item: item[0]):
        lines.append(f"| {weight:.4f} | {stats.source_count} | {stats.generated_copies} |")

    lines.extend(["", "## Split Summaries"])
    for split in copy_splits:
        split_stats = dataset_stats_by_split.get(split)
        if not split_stats:
            lines.append(f"- No samples were generated for split '{split}'.")
            continue

        lines.extend(
            [
                "",
                f"### {split.capitalize()} Split",
                "| dataset | original samples | generated samples |",
                "| ------- | ---------------- | ----------------- |",
            ]
        )
        for stats in sorted(split_stats.values(), key=lambda x: x.name):
            lines.append(f"| {stats.name} | {stats.original_samples} | {stats.generated_samples} |")

        split_weight_stats = weight_stats_by_split.get(split, {})
        if split_weight_stats:
            lines.extend(
                [
                    "",
                    f"#### {split.capitalize()} Weight Buckets",
                    "| weight | source samples | generated copies |",
                    "| ------ | -------------- | ---------------- |",
                ]
            )
            for weight, stats in sorted(split_weight_stats.items(), key=lambda item: item[0]):
                lines.append(f"| {weight:.4f} | {stats.source_count} | {stats.generated_copies} |")

    lines.extend(
        [
            "",
            "## Notes",
            "- Parquet files (`train.parquet`, `val.parquet`, `test.parquet`) contain the weighted samples with enriched metadata columns.",
            "- Each parquet file has a maximum shard size of 5GB.",
            "- Load datasets with: `from datasets import load_dataset; ds = load_dataset('parquet', data_files='train.parquet')`",
            "- Weight diagnostic plots (`weights.png` or `<split>_weights.png`) show the relative scaling applied to the sampler outputs.",
            "- Copy counts are derived from sampler weights after normalising by the minimum weight.",
        ]
    )

    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(lines))
    return readme_path, lines
