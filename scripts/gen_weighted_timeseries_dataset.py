"""Generate a re-weighted meta time-series dataset as Hugging Face parquet files."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from datasets import Array2D, Dataset, Features, Value
from tqdm.auto import tqdm

from signal_diffusion.config import DatasetSettings, Settings, load_settings
from signal_diffusion.data.meta import META_LABELS
from signal_diffusion.log_setup import get_logger

logger = get_logger(__name__)

DEFAULT_DATASETS: tuple[str, ...] = ("math", "parkinsons", "seed", "longitudinal")
DEFAULT_SPLITS: tuple[str, ...] = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a weighted meta time-series dataset as parquet files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to TOML config.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Datasets to include (ignored if --all is set).",
    )
    parser.add_argument(
        "--all",
        dest="use_all_datasets",
        action="store_true",
        help="Include all datasets defined in the TOML config.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory for parquet files.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to materialise (must correspond to existing metadata files).",
    )
    parser.add_argument("--seed", type=int, default=205, help="Seed (reserved for future stochastic steps).")
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


def compute_balanced_weights(
    metadata: pd.DataFrame,
    *,
    dataset_order: Sequence[str],
    max_sampling_weight: float | None = None,
) -> np.ndarray:
    """Compute per-row weights that balance datasets and gender."""
    if metadata.empty:
        return np.asarray([], dtype=np.float64)

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


def scale_weights(weights: np.ndarray) -> np.ndarray:
    if weights.size == 0:
        return weights
    min_weight = float(np.min(weights))
    if min_weight <= 0:
        raise ValueError("Weights must be positive to compute copy counts.")
    return weights / min_weight


def assign_copies(weight: float, count: int) -> list[int]:
    """Distribute fractional weights into integer copy counts."""
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


def compute_copy_counts(scaled_weights: np.ndarray) -> np.ndarray:
    if scaled_weights.size == 0:
        return np.asarray([], dtype=np.int64)
    copy_counts = np.zeros_like(scaled_weights, dtype=np.int64)
    unique_weights = np.unique(scaled_weights)
    for weight in np.sort(unique_weights):
        indices = np.flatnonzero(np.isclose(scaled_weights, weight))
        if indices.size == 0:
            continue
        counts = assign_copies(float(weight), indices.size)
        copy_counts[indices] = counts
    return copy_counts


def build_split_samples(
    split_metadata: pd.DataFrame,
    copy_counts: np.ndarray,
    dataset_roots: Mapping[str, Path],
) -> list[dict[str, Any]]:
    """Load and duplicate signals based on copy counts."""
    samples: list[dict[str, Any]] = []

    for idx, copies in enumerate(copy_counts):
        if copies <= 0:
            continue
        row = split_metadata.iloc[idx]
        dataset_name = row["dataset"]
        root = dataset_roots[dataset_name]
        data_path = root / row["file_name"]
        signal = np.load(data_path).astype(np.float32)

        base_sample: dict[str, Any] = {"timeseries": signal}
        for key, value in row.items():
            base_sample[key] = _clean_value(value)
        base_sample["original_file"] = base_sample.get("file_name")

        for _ in range(copies):
            samples.append(base_sample.copy())

    return samples


def build_features(sample: Mapping[str, Any]) -> Features:
    ts_shape = tuple(sample["timeseries"].shape)
    feature_defs: dict[str, Any] = {"timeseries": Array2D(shape=ts_shape, dtype="float32")}
    for key, value in sample.items():
        if key == "timeseries":
            continue
        feature_defs[key] = _feature_for_value(value)
    return Features(feature_defs)


def build_hf_dataset(samples: list[dict[str, Any]]) -> Dataset:
    if not samples:
        raise ValueError("No samples were generated for this split.")
    first = samples[0]
    ts_shape = tuple(first["timeseries"].shape)
    for sample in samples:
        if tuple(sample["timeseries"].shape) != ts_shape:
            raise ValueError("Inconsistent timeseries shapes detected across samples.")
    features = build_features(first)
    columns: dict[str, list[Any]] = {key: [] for key in first.keys()}

    for sample in samples:
        for key, value in sample.items():
            columns[key].append(value)

    return Dataset.from_dict(columns, features=features)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    settings = load_settings(args.config)

    if args.use_all_datasets:
        dataset_names = tuple(settings.datasets.keys())
        if not dataset_names:
            raise ValueError("No datasets configured in the provided TOML.")
    else:
        if not args.datasets:
            raise ValueError("Specify --datasets or use --all to include every configured dataset.")
        dataset_names = tuple(args.datasets)

    logger.info("Loading metadata for datasets: %s", ", ".join(dataset_names))
    metadata = load_metadata(settings, dataset_names, args.splits)

    weights = compute_balanced_weights(
        metadata,
        dataset_order=dataset_names,
        max_sampling_weight=settings.max_sampling_weight,
    )

    dataset_roots = {
        name: resolve_timeseries_root(settings.dataset(name)) for name in dataset_names
    }

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating weighted datasets in %s", output_dir)
    for split in args.splits:
        split_mask = metadata["split"] == split
        if not split_mask.any():
            logger.warning("No rows found for split '%s'; skipping.", split)
            continue

        split_metadata = metadata.loc[split_mask].reset_index(drop=True)
        split_weights = weights[split_mask.to_numpy()]
        scaled_weights = scale_weights(split_weights)
        copy_counts = compute_copy_counts(scaled_weights)

        samples = build_split_samples(split_metadata, copy_counts, dataset_roots)
        if not samples:
            logger.warning("No samples generated for split '%s'; skipping write.", split)
            continue

        dataset = build_hf_dataset(samples)
        out_path = output_dir / f"{split}.parquet"
        dataset.to_parquet(out_path)
        logger.info("Saved %d samples to %s", len(dataset), out_path)


if __name__ == "__main__":
    main()
