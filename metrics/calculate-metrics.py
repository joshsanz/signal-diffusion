"""CLI for computing dataset fidelity metrics."""

import argparse
import json
from pathlib import Path
from typing import Sequence

from signal_diffusion.metrics import (
    FEATURE_EXTRACTORS,
    FidelityConfig,
    ParquetDatasetConfig,
    RandomSubsetDataset,
    calculate_metrics_for_extractors,
    clear_fidelity_cache,
    load_parquet_dataset,
)


FEATURE_EXTRACTOR_CHOICES = (
    "inception-v3-compat",
    "vgg16",
    "clip-rn50",
    "clip-rn101",
    "clip-rn50x4",
    "clip-rn50x16",
    "clip-rn50x64",
    "clip-vit-b-32",
    "clip-vit-b-16",
    "clip-vit-l-14",
    "clip-vit-l-14-336px",
    "dinov2-vit-s-14",
    "dinov2-vit-b-14",
    "dinov2-vit-l-14",
    "dinov2-vit-g-14",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate image fidelity metrics for generated datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-g", "--generated", required=True, help="Path to generated dataset")
    parser.add_argument("-r", "--real", required=True, help="Path to real dataset")
    parser.add_argument("--split", default="train", help="Dataset split to load when dataset is a parquet directory")
    parser.add_argument(
        "--generated-split",
        default=None,
        help="Override the split for the generated dataset (defaults to --split)",
    )
    parser.add_argument("--image-key", default="image", help="Column name for images in the parquet dataset")
    parser.add_argument("--kid-subset-size", type=int, default=1000, help="Samples per subset for KID calculation")
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="Batch size for feature extraction")
    parser.add_argument("-o", "--output", default="metrics.json", help="Path to write metrics JSON")
    parser.add_argument("--cache", action="store_true", help="Cache extracted features for reuse")
    parser.add_argument("--cache-prefix", default=None, help="Optional prefix for cache entries when caching is enabled")
    parser.add_argument(
        "--feature-extractor",
        dest="feature_extractors",
        action="append",
        choices=FEATURE_EXTRACTOR_CHOICES,
        help="Feature extractor identifier (may be provided multiple times)",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=-1,
        help="Random subset size for evaluation (-1 uses the full dataset)",
    )
    parser.add_argument("--clear-cache", action="store_true", help="Clear cached features before running")
    return parser.parse_args()


def build_dataset(path: str, subset_size: int, split: str, image_key: str) -> RandomSubsetDataset:
    dataset = load_parquet_dataset(
        ParquetDatasetConfig(
            data_dir=path,
            split=split,
            image_key=image_key,
        )
    )
    if subset_size > 0:
        return RandomSubsetDataset(dataset, subset_size=subset_size, image_key=image_key)
    return RandomSubsetDataset(dataset, image_key=image_key)


def main(args: argparse.Namespace) -> None:
    real_path = Path(args.real)
    generated_path = Path(args.generated)
    real_name = real_path.name
    generated_name = generated_path.name

    if args.clear_cache:
        clear_fidelity_cache(real_name)
        clear_fidelity_cache(generated_name)

    subset_size = args.subset_size
    split = args.split
    generated_split = args.generated_split or split
    image_key = args.image_key
    real_dataset = build_dataset(args.real, subset_size, split, image_key)
    generated_dataset = build_dataset(args.generated, subset_size, generated_split, image_key)

    config = FidelityConfig(
        kid_subset_size=args.kid_subset_size,
        batch_size=args.batch_size,
        cache_prefix=args.cache_prefix,
        cache_results=args.cache,
    )

    extractors: Sequence[str]
    if args.feature_extractors:
        extractors = args.feature_extractors
    else:
        extractors = FEATURE_EXTRACTORS

    print(f"Evaluating generated dataset '{generated_name}' against real dataset '{real_name}'")
    metrics = calculate_metrics_for_extractors(
        real_dataset,
        generated_dataset,
        extractors=extractors,
        config=config,
        real_dataset_name=real_name,
        generated_dataset_name=generated_name,
    )
    for extractor, values in metrics.items():
        print(f"Metrics for {extractor}: {values}")

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Saved metrics to {args.output}")


if __name__ == "__main__":
    main(parse_args())
