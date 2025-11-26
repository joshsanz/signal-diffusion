"""Generate a re-weighted meta time-series dataset as Hugging Face parquet files."""
from __future__ import annotations

# from rich.traceback import install
# install(show_locals=False)

import argparse
import importlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from datasets import Array2D, Dataset, Features, Value, concatenate_datasets
from tqdm.auto import tqdm

from signal_diffusion.config import DatasetSettings, Settings, load_settings
from signal_diffusion.log_setup import get_logger
from weighted_dataset_utils import (
    DatasetStats,
    WeightStats,
    assign_copies,
    compute_balanced_weights,
    enrich_metadata,
    parse_copy_splits,
    parse_splits,
    prepare_output_dir,
    save_weights_plot,
    scale_numpy_weights,
    set_random_seeds,
    validate_splits_compatibility,
    write_readme,
)

logger = get_logger(__name__)

DEFAULT_DATASETS: tuple[str, ...] = ("math", "parkinsons", "seed", "longitudinal")
DEFAULT_TASKS: tuple[str, ...] = ("gender",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a weighted meta time-series dataset as parquet files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to a TOML config file.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(DEFAULT_DATASETS),
        help="Datasets to include.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=DEFAULT_TASKS,
        help="Tasks to require when building the meta dataset.",
    )
    parser.add_argument("--nsamps", type=int, default=2048, help="Samples per window when preprocessing.")
    parser.add_argument("--fs", type=int, default=125, help="Target sample rate used during preprocessing.")
    parser.add_argument(
        "--ovr-perc",
        type=float,
        default=0.5,
        help="Overlap percentage to forward to timeseries preprocessors.",
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
    parser.add_argument("--seed", type=int, default=205, help="Random seed applied to NumPy for reproducibility.")
    parser.add_argument("--preprocess", action="store_true", help="Run time-series preprocessors before sampling.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output directory if present.")
    parser.add_argument("--skip-plot", action="store_true", help="Skip saving the weight diagnostic plot.")
    parser.add_argument(
        "--writer-batch-size",
        type=int,
        default=None,
        help="Optional writer batch size for Hugging Face dataset creation to avoid Arrow 2GB overflow.",
    )
    return parser.parse_args()


def resolve_timeseries_root(settings: DatasetSettings) -> Path:
    if settings.timeseries_output is not None:
        return settings.timeseries_output
    if settings.output.name == "stfts":
        return settings.output.parent / "timeseries"
    return settings.output


def _clean_value(value: Any) -> Any:
    """Normalize metadata values for HF serialization."""
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _feature_for_value(value: Any) -> Value:
    if isinstance(value, bool):
        return Value("bool")
    if isinstance(value, (int, np.integer)):
        return Value("int32")
    if isinstance(value, (float, np.floating)):
        return Value("float32")
    return Value("string")


def load_metadata(settings: Settings, dataset_names: Sequence[str], splits: Sequence[str]) -> pd.DataFrame:
    """Load and concatenate per-split metadata across datasets."""
    frames: list[pd.DataFrame] = []
    for dataset_name in dataset_names:
        ds_settings = settings.dataset(dataset_name)
        root = resolve_timeseries_root(ds_settings)

        for split in splits:
            metadata_path = root / f"{split}-metadata.csv"
            if not metadata_path.exists():
                logger.warning("Skipping missing metadata file: %s", metadata_path)
                continue
            df = pd.read_csv(metadata_path)
            df["dataset"] = dataset_name
            df["split"] = split
            frames.append(df)

    if not frames:
        raise FileNotFoundError("No metadata files found for the requested datasets/splits.")

    return pd.concat(frames, ignore_index=True)


def run_timeseries_preprocessor(
    settings: Settings,
    dataset_names: Sequence[str],
    *,
    nsamps: int,
    ovr_perc: float,
    fs: int,
    splits: dict[str, float],
    overwrite: bool,
) -> None:
    for name in dataset_names:
        logger.info("Preprocessing %s (time-series)...", name)
        module = importlib.import_module(f"signal_diffusion.data.{name}")
        base = name.upper() if name == "seed" else name.capitalize()
        preprocessor_class_name = f"{base}TimeSeriesPreprocessor"
        preprocessor_class = getattr(module, preprocessor_class_name)
        preprocessor = preprocessor_class(settings, nsamps=nsamps, ovr_perc=ovr_perc, fs=fs)
        preprocessor.preprocess(splits=splits, overwrite=overwrite)


def load_weighted_timeseries(
    split_metadata: pd.DataFrame,
    scaled_weights: np.ndarray,
    dataset_roots: Mapping[str, Path],
    *,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, DatasetStats], dict[float, WeightStats]]:
    """Load time-series windows and metadata according to their weights for a specific split."""
    dataset_stats: dict[str, DatasetStats] = {}
    weight_stats: dict[float, WeightStats] = {}
    samples: list[dict[str, Any]] = []

    for dataset_name, group in split_metadata.groupby("dataset"):
        dataset_stats[dataset_name] = DatasetStats(name=dataset_name, original_samples=len(group))

    unique_weights = np.unique(scaled_weights)
    progress = tqdm(total=len(scaled_weights), desc="Loading samples", unit="sample")

    for weight_value in np.sort(unique_weights):
        mask = np.isclose(scaled_weights, weight_value)
        idxs = np.flatnonzero(mask)
        if idxs.size == 0:
            continue

        weight_stat = weight_stats.setdefault(float(weight_value), WeightStats(weight=float(weight_value)))
        weight_stat.source_count += idxs.size
        copy_schedule = assign_copies(float(weight_value), idxs.size)

        for idx, copies in zip(idxs, copy_schedule):
            row = split_metadata.iloc[int(idx)]
            dataset_name = row["dataset"]
            root = dataset_roots[dataset_name]
            data_path = root / row["file_name"]
            if not data_path.exists():
                logger.warning("Skipping missing source file: %s", data_path)
                continue

            try:
                signal = np.load(data_path).astype(np.float32)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Failed to load %s: %s", data_path, exc)
                continue

            metadata_copy = row.copy()
            enriched = enrich_metadata(metadata_copy, split=split, modality="time-series segment")
            enriched_dict = {k: _clean_value(v) for k, v in enriched.to_dict().items()}
            enriched_dict["original_file"] = enriched_dict.get("file_name")

            for _ in range(copies):
                samples.append({"signal": signal.copy(), **enriched_dict})

            dataset_stats[dataset_name].generated_samples += copies
            weight_stat.generated_copies += copies
            progress.update(1)

    progress.close()
    return samples, dataset_stats, weight_stats


def build_features(sample: Mapping[str, Any]) -> Features:
    ts_shape = tuple(sample["signal"].shape)
    feature_defs: dict[str, Any] = {"signal": Array2D(shape=ts_shape, dtype="float32")}
    for key, value in sample.items():
        if key == "signal":
            continue
        feature_defs[key] = _feature_for_value(value)
    return Features(feature_defs)


def _estimate_writer_batch_size(sample: Mapping[str, Any], target_bytes: int = 512 * 1024 * 1024) -> int:
    """Derive a conservative writer batch size to stay under Arrow's 2GB limit."""
    signal = sample.get("signal")
    ts_bytes = int(signal.nbytes) if isinstance(signal, np.ndarray) else 0
    approx_sample_bytes = max(ts_bytes, 1) + 1024  # pad for metadata overhead
    batch_size = max(1, target_bytes // approx_sample_bytes)
    return int(batch_size)


def build_hf_dataset(samples: list[dict[str, Any]], *, writer_batch_size: int | None = None) -> tuple[Dataset, int]:
    """Build HF dataset from samples, chunking to avoid memory overflow.

    Chunks samples to avoid memory issues when creating large datasets.
    """
    if not samples:
        raise ValueError("No samples were generated for this split.")
    first = samples[0]
    ts_shape = tuple(first["signal"].shape)
    for sample in samples:
        if tuple(sample["signal"].shape) != ts_shape:
            raise ValueError("Inconsistent signal shapes detected across samples: "
                             "expected %s but got %s", ts_shape, tuple(sample["signal"].shape))
    features = build_features(first)
    resolved_batch_size = writer_batch_size or _estimate_writer_batch_size(first)

    # Chunk samples to avoid memory overflow
    chunk_size = 4096
    chunks = []

    for i in range(0, len(samples), chunk_size):
        chunk = samples[i : i + chunk_size]
        columns: dict[str, list[Any]] = {key: [] for key in first.keys()}

        for sample in chunk:
            for key, value in sample.items():
                columns[key].append(value)

        dataset_chunk = Dataset.from_dict(columns, features=features)
        chunks.append(dataset_chunk)

    # Concatenate all chunks
    dataset = concatenate_datasets(chunks)
    return dataset, resolved_batch_size


def main() -> None:
    args = parse_args()
    set_random_seeds(args.seed)
    settings = load_settings(args.config)

    datasets = tuple(args.datasets)
    tasks = tuple(args.tasks)
    copy_splits = parse_copy_splits(args.copy_splits)

    logger.info(
        "Starting weighted time-series generation | datasets=%s | tasks=%s | nsamps=%d | fs=%d | copy_splits=%s",
        ",".join(datasets),
        ",".join(tasks),
        args.nsamps,
        args.fs,
        ",".join(copy_splits),
    )
    logger.info("Using settings file %s", settings.config_path)

    if args.output_dir is None:
        output_dir = settings.output_root / f"reweighted_timeseries_meta_dataset_n{args.nsamps}_fs{args.fs}"
    else:
        output_dir = Path(args.output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = (settings.output_root / output_dir).resolve()
    output_dir = output_dir.resolve()
    prepare_output_dir(output_dir, overwrite=args.overwrite)

    if args.preprocess:
        splits = parse_splits(args.splits)
        validate_splits_compatibility(args.preprocess, splits, copy_splits)
        run_timeseries_preprocessor(
            settings,
            datasets,
            nsamps=args.nsamps,
            ovr_perc=args.ovr_perc,
            fs=args.fs,
            splits=splits,
            overwrite=args.overwrite,
        )

    metadata = load_metadata(settings, datasets, copy_splits)

    weights = compute_balanced_weights(
        metadata,
        dataset_order=datasets,
        max_sampling_weight=settings.max_sampling_weight,
    )

    dataset_roots = {
        name: resolve_timeseries_root(settings.dataset(name)) for name in datasets
    }

    dataset_stats_total: dict[str, DatasetStats] = {}
    dataset_stats_by_split: dict[str, dict[str, DatasetStats]] = {}
    weight_stats_total: dict[float, WeightStats] = {}
    weight_stats_by_split: dict[str, dict[float, WeightStats]] = {}
    plot_paths: dict[str, Path] = {}
    parquet_paths: dict[str, Path] = {}

    for split in copy_splits:
        logger.info("Processing split '%s'", split)
        split_mask = metadata["split"] == split
        if not split_mask.any():
            logger.warning("No rows found for split '%s'; skipping.", split)
            continue

        split_metadata = metadata.loc[split_mask].reset_index(drop=True)
        split_weights = weights[split_mask.to_numpy()]
        scaled_weights = scale_numpy_weights(split_weights)

        if not args.skip_plot:
            plot_label = split if len(copy_splits) > 1 else "all"
            plot_split_arg = None if plot_label == "all" else split
            plot_paths[plot_label] = save_weights_plot(scaled_weights, output_dir, split=plot_split_arg)

        samples, split_dataset_stats, split_weight_stats = load_weighted_timeseries(
            split_metadata,
            scaled_weights,
            dataset_roots,
            split=split,
        )

        if not samples:
            logger.warning("No samples generated for split '%s'; skipping write.", split)
            continue

        dataset, writer_batch_size = build_hf_dataset(samples, writer_batch_size=args.writer_batch_size)
        out_path = output_dir / f"{split}.parquet"
        logger.info("Saving split '%s' with writer_batch_size=%d", split, writer_batch_size)
        dataset.to_parquet(out_path, batch_size=writer_batch_size)
        parquet_paths[split] = out_path
        logger.info("Saved %d samples to %s", len(dataset), out_path)

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
        modality="time-series",
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
