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
from pathlib import Path
from typing import Any, Iterable, Mapping, Sized, cast

import pandas as pd
import torch
from datasets import Dataset, Image as DatasetImage, concatenate_datasets
from PIL import Image
from tqdm.auto import tqdm

from signal_diffusion.config import Settings, load_settings
from signal_diffusion.data.meta import MetaDataset, MetaPreprocessor, MetaSampler
from signal_diffusion.log_setup import get_logger
from weighted_dataset_utils import (
    DatasetStats,
    WeightStats,
    assign_copies,
    compute_scaled_weights,
    enrich_metadata,
    parse_copy_splits,
    parse_splits,
    prepare_output_dir,
    save_weights_plot,
    set_random_seeds,
    validate_splits_compatibility,
    write_readme,
)


logger = get_logger(__name__)

DEFAULT_DATASETS: tuple[str, ...] = ("math", "parkinsons", "seed", "longitudinal")
DEFAULT_TASKS: tuple[str, ...] = ("gender",)


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
    parser.add_argument("--nsamps", type=int, default=2048, help="Samples per STFT when preprocessing (MetaPreprocessor nsamps).")
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
    parser.add_argument("-y", "--force", action="store_true", help="Skip overwrite confirmation and force overwrite.")
    return parser.parse_args()


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
        dataset_settings = getattr(ds, "dataset_settings", None)
        name = getattr(dataset_settings, "name", getattr(ds, "name", "unknown"))
        original_samples = len(cast(Sized, ds)) if hasattr(ds, "__len__") else 0
        dataset_stats[str(name)] = DatasetStats(name=str(name), original_samples=original_samples)

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
            dataset_settings = getattr(source_dataset, "dataset_settings", None)
            dataset_name = getattr(dataset_settings, "name", f"dataset_{dataset_idx}")
            # Get metadata for this sample
            metadata = getattr(source_dataset, "metadata", None)
            if metadata is None:
                logger.warning("Skipping dataset without metadata for index %s", dataset_idx)
                continue
            if not isinstance(metadata, pd.DataFrame):
                logger.warning("Skipping dataset with non-tabular metadata for index %s", dataset_idx)
                continue
            metadata_row = metadata.iloc[sample_offset]
            metadata_dict = metadata_row.to_dict()

            relative_source = Path(str(metadata_dict["file_name"]))
            source_root_value = getattr(source_dataset, "root", None)
            if source_root_value is None:
                logger.warning("Skipping dataset without root for index %s", dataset_idx)
                continue
            source_root = Path(str(source_root_value))
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
    """Save examples to parquet dataset using HuggingFace datasets library.

    Chunks examples to avoid memory overflow when creating large datasets.
    """
    chunk_size = 4096
    chunks = []

    for i in range(0, len(examples), chunk_size):
        chunk = examples[i : i + chunk_size]

        # Create DataFrame to validate and align columns
        df = pd.DataFrame([{k: v for k, v in ex.items() if k != "image"} for ex in chunk])

        # Convert to HuggingFace Dataset with Image feature
        dataset_dict = {k: list(df[k]) if k in df.columns else [] for k in df.columns}
        dataset_dict["image"] = [ex["image"] for ex in chunk]

        hf_dataset = Dataset.from_dict(dataset_dict)
        # Cast image column to HuggingFace Image type
        hf_dataset = hf_dataset.cast_column("image", DatasetImage())
        chunks.append(hf_dataset)

    # Concatenate all chunks
    hf_dataset = concatenate_datasets(chunks)

    # Save as parquet file
    output_path = output_dir / f"{split}.parquet"
    hf_dataset.to_parquet(str(output_path))
    logger.info("Saved %d samples to %s", len(hf_dataset), output_path)
    return output_path


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
    prepare_output_dir(output_dir, overwrite=args.overwrite, force=args.force)

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
        scaled_weights = compute_scaled_weights(sampler.weights)
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

    preprocessing_fields = {
        "nsamps": args.nsamps,
        "fs": args.fs,
        "resolution": args.resolution,
        "bin_spacing": args.bin_spacing,
        "overlap_percent": args.ovr_perc,
    }

    readme_path, _ = write_readme(
        output_dir,
        settings,
        dataset_stats=dataset_stats_total,
        dataset_stats_by_split=dataset_stats_by_split,
        weight_stats=weight_stats_total,
        weight_stats_by_split=weight_stats_by_split,
        copy_splits=copy_splits,
        args=args,
        modality="spectrogram",
        preprocessing_fields=preprocessing_fields,
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
