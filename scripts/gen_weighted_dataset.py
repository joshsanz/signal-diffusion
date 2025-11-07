"""Generate a re-weighted meta-dataset as parquet files.

The script loads spectrogram files according to the sampling weights emitted by
``MetaSampler`` and saves them as HuggingFace parquet datasets so downstream
training code can consume a class-balanced dataset without relying on per-epoch
sampling tricks. The result folder contains parquet files (one per split) with
embedded images and enriched metadata, weight diagnostic plots, and a README
documenting configuration details.
"""
from __future__ import annotations

import argparse
import bisect
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, Image as DatasetImage
from PIL import Image
from tqdm.auto import tqdm

from signal_diffusion.config import Settings, load_settings
from signal_diffusion.data.meta import MetaDataset, MetaPreprocessor, MetaSampler
from signal_diffusion.log_setup import get_logger


logger = get_logger(__name__)

DEFAULT_DATASETS: tuple[str, ...] = ("math", "parkinsons", "seed", "longitudinal")
DEFAULT_TASKS: tuple[str, ...] = ("gender",)


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


DEFAULT_HEALTH = "H"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def _normalise_gender(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip().upper()
    if text in {"F", "FEMALE", "1", "TRUE"}:
        return "F"
    if text in {"M", "MALE", "0", "FALSE"}:
        return "M"
    return text


def _normalise_health(value: Any) -> str:
    if _is_missing(value):
        return DEFAULT_HEALTH
    text = str(value).strip().upper()
    if text in {"PD", "PARKINSONS", "1", "TRUE"}:
        return "PD"
    if text in {"H", "HEALTHY", "0", "FALSE"}:
        return "H"
    return text


def _try_int_age(value: Any) -> int | None:
    if _is_missing(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _gender_description(code: str | None) -> str | None:
    if _is_missing(code):
        return None
    mapping = {"F": "female", "M": "male"}
    text = str(code).strip().upper()
    return mapping.get(text, text.lower())


def _health_description(code: str | None) -> tuple[str | None, str | None]:
    if _is_missing(code):
        return None, None
    text = str(code).strip().upper()
    if text == "PD":
        return None, "with Parkinson's disease"
    if text == "H":
        return "healthy", None
    return text.lower(), None


def build_caption(metadata: Mapping[str, Any]) -> str:
    age = _try_int_age(metadata.get("age"))
    gender_word = _gender_description(metadata.get("gender"))
    health_code = metadata.get("health", DEFAULT_HEALTH)
    health_primary, health_clause = _health_description(health_code)

    primary_bits: list[str] = []
    if age is not None:
        primary_bits.append(f"{age} year old")
    if health_primary:
        primary_bits.append(health_primary)
    if gender_word:
        primary_bits.append(gender_word)

    if primary_bits:
        caption = f"a spectrogram image of a {', '.join(primary_bits)} subject"
    else:
        caption = "a spectrogram image of a subject"

    if health_clause:
        caption += f" {health_clause}"

    return caption


def enrich_metadata(row: pd.Series, *, split: str) -> pd.Series:
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
    serialised["caption"] = build_caption(serialised)
    return pd.Series(serialised)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a re-weighted meta dataset on disk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to a TOML config file.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help="Datasets to combine.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=DEFAULT_TASKS,
        help="Tasks to require when building the meta dataset.",
    )
    parser.add_argument("--nsamps", type=int, default=2000, help="Samples per STFT when preprocessing (MetaPreprocessor nsamps).")
    parser.add_argument("--fs", type=int, default=125, help="Target sample rate used during preprocessing.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Spectrogram resolution used when preprocessing (passed to preprocess).",
    )
    parser.add_argument(
        "--bin-spacing",
        choices=("log", "linear"),
        default="log",
        help="Frequency bin spacing used by the MetaPreprocessor.",
    )
    parser.add_argument(
        "--ovr-perc",
        type=float,
        default=0.5,
        help="Overlap percentage to forward to MetaPreprocessor (ovr_perc).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train:0.8,val:0.1,test:0.1",
        help="Comma-separated split fractions (e.g. train:0.8,val:0.2). Only used when --preprocess is set.",
    )
    parser.add_argument(
        "--copy-splits",
        type=str,
        default="train,val,test",
        help="Comma-separated dataset splits to materialise in the weighted output (e.g. 'train,val,test').",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory instead of deriving one from settings.output_root. Either relative to output_root or absolute path.",
    )
    parser.add_argument("--seed", type=int, default=205, help="Random seed applied to NumPy and PyTorch for reproducibility.")
    parser.add_argument("--preprocess", action="store_true", help="Run MetaPreprocessor before sampling.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output directory if present.")
    parser.add_argument("--skip-plot", action="store_true", help="Skip saving the weight diagnostic plot.")
    return parser.parse_args()


def parse_splits(raw: str) -> dict[str, float]:
    """Parse comma-separated split specifications into a dictionary."""
    result: dict[str, float] = {}
    # Split by comma and process each split specification
    for item in raw.split(","):
        if not item:
            continue
        try:
            split, value = item.split(":", maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"Invalid split spec '{item}'. Expected format name:value") from exc
        # Store the split name and its fraction
        result[split.strip()] = float(value)
    # Validate that all splits sum to 1.0 (100%)
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
    """Warn if copy-splits doesn't match splits when preprocessing is enabled.

    Args:
        preprocess: Whether preprocessing will be run
        splits: Dict of splits that will be created (e.g., {"train": 0.8, "val": 0.1, "test": 0.1})
        copy_splits: Tuple of splits to copy to output
    """
    if not preprocess or splits is None:
        return

    preprocessed_split_names = set(splits.keys())
    copy_split_names = set(copy_splits)

    # Check if copy_splits contains splits that won't be created
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

    # Check if some preprocessed splits won't be copied
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


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    """Prepare the output directory, optionally removing existing directory if overwrite is enabled."""
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
            )
        response = input(
            f"Output directory already exists at {output_dir}.\n"
            "This will delete all existing files. Continue? [y/N]: "
        ).strip().lower()
        if response not in {"y", "yes"}:
            logger.info("User declined to overwrite %s; aborting.", output_dir)
            sys.exit(0)
        logger.info("Removing existing output directory at %s", output_dir)
        shutil.rmtree(output_dir)

    # Create the output directory (and parent directories if needed)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Created output directory at %s", output_dir)


def set_random_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_preprocessor(
    settings: Settings,
    dataset_names: Iterable[str],
    *,
    nsamps: int,
    ovr_perc: float,
    fs: int,
    bin_spacing: str,
    resolution: int,
    splits: dict[str, float],
    overwrite: bool,
) -> None:
    preprocessor = MetaPreprocessor(
        settings=settings,
        dataset_names=tuple(dataset_names),
        nsamps=nsamps,
        ovr_perc=ovr_perc,
        fs=fs,
        bin_spacing=bin_spacing,
    )
    preprocessor.preprocess(splits=splits, resolution=resolution, overwrite=overwrite)


def compute_scaled_weights(sampler: MetaSampler) -> torch.Tensor:
    """Scale sampler weights so the minimum weight becomes 1.0 for copy count computation."""
    # Get weights from sampler and move to CPU for processing
    weights = sampler.weights.detach().cpu().double()
    if weights.numel() == 0:
        raise RuntimeError("MetaSampler produced no weights; ensure datasets contain samples.")
    # Find the minimum weight value
    min_weight = float(torch.min(weights))
    if min_weight <= 0:
        raise RuntimeError("Sampler weights must be positive to compute copy counts.")
    # Scale weights so minimum weight becomes 1.0 (used for determining copy counts)
    return weights / min_weight


def assign_copies(weight: float, count: int) -> list[int]:
    """
    Distributes a weighted value across a specified number of items by assigning
    integer copies to each item. The fractional part of the weight is distributed
    as evenly as possible by adding extra copies to some items.

    Args:
        weight: The total weight to distribute. Must be positive and yield at least 1 copy.
        count: The number of items to distribute copies across.

    Returns:
        A list of integers representing the number of copies assigned to each item.
        The sum of all copies equals the floor of the weight, with fractional parts
        distributed as evenly as possible.

    Raises:
        ValueError: If the weight would yield zero or negative copies.

    Example:
        assign_copies(3.7, 5) might return [1, 1, 1, 1, 0] or [1, 1, 1, 0, 1]
        depending on how the 0.7 fractional part is distributed.
    """
    # Calculate the base number of copies for each item (floor of weight)
    base = math.floor(weight)
    if base <= 0:
        raise ValueError(f"Weight {weight:.6f} would yield zero copies; check sampler output.")
    # Calculate the fractional part that needs to be distributed
    fractional = max(weight - base, 0.0)
    copies: list[int] = []
    remainder = 0.0
    # Distribute base copies plus fractional parts as evenly as possible
    for _ in range(count):
        extra = 0
        if fractional > 0:
            # Accumulate fractional parts and distribute when >= 1.0
            remainder += fractional
            if remainder + 1e-8 >= 1.0:
                extra = 1
                remainder -= 1.0
        # Add base copies plus any extra copies from fractional distribution
        copies.append(base + extra)
    return copies


def load_weighted_samples(
    dataset: MetaDataset,
    scaled_weights: torch.Tensor,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, DatasetStats], dict[float, WeightStats]]:
    """Load spectrogram images and metadata according to their weights for a specific split."""
    cumulative_sizes = dataset.cumulative_sizes
    all_examples: list[dict[str, Any]] = []
    dataset_stats: dict[str, DatasetStats] = {}
    weight_stats: dict[float, WeightStats] = {}

    for ds in dataset.datasets:
        name = ds.dataset_settings.name if hasattr(ds, "dataset_settings") else getattr(ds, "name", "unknown")
        dataset_stats[name] = DatasetStats(name=name, original_samples=len(ds))

    # Get unique weight values to process each weight bucket separately
    unique_weights = torch.unique(scaled_weights)
    progress = tqdm(total=len(scaled_weights), desc="Loading samples", unit="sample")

    # Process each unique weight value
    for weight_tensor in torch.sort(unique_weights).values:
        weight_value = float(weight_tensor.item())
        # Find all samples with this weight value
        mask = torch.isclose(scaled_weights, weight_tensor, atol=1e-6)
        idxs = torch.nonzero(mask, as_tuple=False).flatten()
        if idxs.numel() == 0:
            continue

        # Track statistics for this weight bucket
        weight_stat = weight_stats.setdefault(weight_value, WeightStats(weight=weight_value))
        weight_stat.source_count += idxs.numel()
        # Determine how many copies to make of each sample in this weight bucket
        copy_schedule = assign_copies(weight_value, idxs.numel())

        # Process each sample in this weight bucket
        for tensor_idx, copies in zip(idxs, copy_schedule):
            # Convert tensor index to global sample index
            index = int(tensor_idx.item())
            # Find which dataset this sample belongs to using binary search
            dataset_idx = bisect.bisect_right(cumulative_sizes, index)
            # Calculate the offset within the dataset
            sample_offset = index if dataset_idx == 0 else index - cumulative_sizes[dataset_idx - 1]
            source_dataset = dataset.datasets[dataset_idx]

            # Get dataset name
            dataset_name = (
                source_dataset.dataset_settings.name
                if hasattr(source_dataset, "dataset_settings")
                else f"dataset_{dataset_idx}"
            )
            # Get metadata for this sample
            metadata_row = source_dataset.metadata.iloc[sample_offset]
            metadata_dict = metadata_row.to_dict()

            relative_source = Path(str(metadata_dict["file_name"]))
            source_root = Path(source_dataset.root)
            source_path = source_root / relative_source

            # Create the specified number of copies for this sample
            for copy_index in range(copies):
                # Load the image
                if not source_path.exists():
                    logger.warning("Skipping missing source file: %s", source_path)
                    continue

                try:
                    pil_image = Image.open(source_path).convert("RGB")
                except Exception as e:
                    logger.warning("Failed to load image %s: %s", source_path, e)
                    continue

                # Update metadata for this copy
                metadata_copy = metadata_row.copy()
                enriched = enrich_metadata(metadata_copy, split=split)
                enriched_dict = enriched.to_dict()

                # Add image to example
                example = enriched_dict.copy()
                example["image"] = pil_image

                all_examples.append(example)

            # Update statistics
            dataset_stats[dataset_name].generated_samples += copies
            weight_stat.generated_copies += copies
            progress.update(1)

    progress.close()
    return all_examples, dataset_stats, weight_stats


def save_parquet_dataset(
    examples: list[dict[str, Any]],
    output_dir: Path,
    split: str,
) -> Path:
    """Save examples to parquet dataset using HuggingFace datasets library."""
    # Create DataFrame to validate and align columns
    df = pd.DataFrame([{k: v for k, v in ex.items() if k != "image"} for ex in examples])

    # Convert to HuggingFace Dataset with Image feature
    dataset_dict = {k: list(df[k]) if k in df.columns else [] for k in df.columns}
    dataset_dict["image"] = [ex["image"] for ex in examples]

    hf_dataset = Dataset.from_dict(dataset_dict)
    # Cast image column to HuggingFace Image type
    hf_dataset = hf_dataset.cast_column("image", DatasetImage())

    # Save as parquet file
    output_path = output_dir / f"{split}.parquet"
    hf_dataset.to_parquet(str(output_path))
    logger.info("Saved %d samples to %s", len(hf_dataset), output_path)
    return output_path


def save_weights_plot(scaled_weights: torch.Tensor, output_dir: Path, *, split: str | None = None) -> Path:
    plt.figure(figsize=(10, 4))
    plt.plot(scaled_weights.numpy())
    plt.title("Scaled Sample Weights")
    plt.xlabel("Sample Index")
    plt.ylabel("Weight (relative to minimum)")
    plt.tight_layout()
    filename = "weights.png" if split is None else f"{split}_weights.png"
    plot_path = output_dir / filename
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def write_readme(
    output_dir: Path,
    settings: Settings,
    *,
    dataset_stats: dict[str, DatasetStats],
    dataset_stats_by_split: dict[str, dict[str, DatasetStats]],
    weight_stats: dict[float, WeightStats],
    weight_stats_by_split: dict[str, dict[float, WeightStats]],
    copy_splits: tuple[str, ...],
    args: argparse.Namespace,
) -> tuple[Path, list[str]]:
    total_generated = sum(stats.generated_samples for stats in dataset_stats.values())
    summary = (
        "This dataset merges the "
        + ", ".join(args.datasets)
        + " spectrogram corpora with tasks "
        + ", ".join(args.tasks)
        + f". The weighted copies cover splits {', '.join(copy_splits)} and yield {total_generated} total spectrograms."
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
        "  - spectrogram",
        "  - signal-processing",
        "dataset_summary: |",
        f"  {summary}",
        "configurations:",
        f"  - name: {args.bin_spacing}_bin_spacing",
        f"    description: Weighted copies using {args.bin_spacing} frequency bins",
        f"  - name: nsamps_{args.nsamps}",
        f"    description: Spectrogram windows with nsamps={args.nsamps}",
        "splits:",
    ] + [f"  - {split}" for split in copy_splits] + [
        "preprocessing:",
        f"  nsamps: {args.nsamps}",
        f"  fs: {args.fs}",
        f"  resolution: {args.resolution}",
        f"  bin_spacing: {args.bin_spacing}",
        f"  overlap_percent: {args.ovr_perc}",
        f"  preprocess_ran: {str(args.preprocess).lower()}",
        "generation:",
        f"  seed: {args.seed}",
        f"  total_generated_samples: {total_generated}",
        "---",
        "",
    ]

    lines = front_matter + [
        "# Weighted Meta Dataset",
        "",
        f"Generated on: {datetime.now(timezone.utc).isoformat()}",
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
            "- Parquet files (`train.parquet`, `val.parquet`, `test.parquet`) contain the weighted spectrograms with HuggingFace Image feature and enriched metadata columns.",
            "- Each parquet file has a maximum shard size of 5GB.",
            "- Load datasets with: `from datasets import load_dataset; ds = load_dataset('parquet', data_files='train.parquet')`",
            "- Weight diagnostic plots (`weights.png` or `<split>_weights.png`) show the relative scaling applied to the sampler outputs.",
            "- Copy counts are derived from the MetaSampler weights after normalising by the minimum weight.",
        ]
    )

    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(lines))
    return readme_path, lines


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    set_random_seeds(args.seed)

    datasets = tuple(args.datasets)
    tasks = tuple(args.tasks)
    copy_splits = parse_copy_splits(args.copy_splits)

    logger.info(
        "Starting weighted dataset generation | datasets=%s | tasks=%s | nsamps=%d | fs=%d | bin_spacing=%s | copy_splits=%s",
        ",".join(datasets),
        ",".join(tasks),
        args.nsamps,
        args.fs,
        args.bin_spacing,
        ",".join(copy_splits),
    )
    logger.info("Using settings file %s", settings.config_path)

    if args.output_dir is None:
        output_dir = settings.output_root / f"reweighted_meta_dataset_{args.bin_spacing}_n{args.nsamps}_fs{args.fs}"
    else:
        output_dir = Path(args.output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = (settings.output_root / output_dir).resolve()
    output_dir = output_dir.resolve()
    prepare_output_dir(output_dir, overwrite=args.overwrite)

    # Run preprocessing if requested (creates spectrograms from raw data)
    if args.preprocess:
        splits = parse_splits(args.splits)
        # Validate that copy_splits matches the splits that will be created
        validate_splits_compatibility(args.preprocess, splits, copy_splits)
        logger.info("Running MetaPreprocessor with splits: %s", splits)
        run_preprocessor(
            settings,
            datasets,
            nsamps=args.nsamps,
            ovr_perc=args.ovr_perc,
            fs=args.fs,
            bin_spacing=args.bin_spacing,
            resolution=args.resolution,
            splits=splits,
            overwrite=args.overwrite,
        )

    dataset_stats_total: dict[str, DatasetStats] = {}
    dataset_stats_by_split: dict[str, dict[str, DatasetStats]] = {}
    weight_stats_total: dict[float, WeightStats] = {}
    weight_stats_by_split: dict[str, dict[float, WeightStats]] = {}
    plot_paths: dict[str, Path] = {}
    parquet_paths: dict[str, Path] = {}

    for split in copy_splits:
        logger.info("Processing split '%s'", split)
        try:
            meta_dataset = MetaDataset(settings, datasets, split=split, tasks=tasks)
        except FileNotFoundError as exc:
            logger.warning("Skipping split '%s' due to missing metadata: %s", split, exc)
            continue
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to assemble meta dataset for split '%s': %s", split, exc)
            continue

        if len(meta_dataset) == 0:
            logger.warning("Split '%s' contains no samples; skipping.", split)
            continue

        sampler = MetaSampler(meta_dataset, num_samples=len(meta_dataset))
        scaled_weights = compute_scaled_weights(sampler)
        logger.info(
            "Split '%s' has %d samples across %d weight buckets",
            split,
            len(scaled_weights),
            len(torch.unique(scaled_weights)),
        )

        if not args.skip_plot:
            plot_label = split if len(copy_splits) > 1 else "all"
            plot_split_arg = None if plot_label == "all" else split
            plot_paths[plot_label] = save_weights_plot(scaled_weights, output_dir, split=plot_split_arg)

        split_examples, split_dataset_stats, split_weight_stats = load_weighted_samples(
            meta_dataset,
            scaled_weights,
            split,
        )

        if not split_examples:
            logger.warning("Split '%s' produced no loaded samples; skipping.", split)
            continue

        # Save split as parquet
        parquet_paths[split] = save_parquet_dataset(split_examples, output_dir, split)

        dataset_stats_by_split[split] = split_dataset_stats
        weight_stats_by_split[split] = split_weight_stats

        for name, stats in split_dataset_stats.items():
            aggregate = dataset_stats_total.setdefault(name, DatasetStats(name=name, original_samples=0, generated_samples=0))
            aggregate.original_samples += stats.original_samples
            aggregate.generated_samples += stats.generated_samples

        for weight, stats in split_weight_stats.items():
            aggregate_weight = weight_stats_total.setdefault(weight, WeightStats(weight=weight))
            aggregate_weight.source_count += stats.source_count
            aggregate_weight.generated_copies += stats.generated_copies

        generated_split_total = sum(stat.generated_samples for stat in split_dataset_stats.values())
        logger.info(
            "Completed split '%s' | generated=%d | component_datasets=%s",
            split,
            generated_split_total,
            ",".join(f"{name}:{stat.generated_samples}" for name, stat in split_dataset_stats.items()),
        )

    if not parquet_paths:
        raise RuntimeError("No samples were loaded across the requested splits; verify dataset metadata and sampler configuration.")

    readme_path, _ = write_readme(
        output_dir,
        settings,
        dataset_stats=dataset_stats_total,
        dataset_stats_by_split=dataset_stats_by_split,
        weight_stats=weight_stats_total,
        weight_stats_by_split=weight_stats_by_split,
        copy_splits=copy_splits,
        args=args,
    )

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


if __name__ == "__main__":
    main()
