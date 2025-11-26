"""
Preprocess time-series EEG datasets to generate normalized .npy windows and metadata.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from signal_diffusion.config import load_settings
from signal_diffusion.data import (
    LongitudinalTimeSeriesPreprocessor,
    MathTimeSeriesPreprocessor,
    ParkinsonsTimeSeriesPreprocessor,
    SEEDTimeSeriesPreprocessor,
)
from signal_diffusion.data.base import BaseSpectrogramPreprocessor
from signal_diffusion.log_setup import get_logger

logger = get_logger(__name__)

# Default target sampling rates after decimation for each dataset
DATASET_DEFAULT_FS: dict[str, float] = {
    "seed": 125.0,
    "parkinsons": 125.0,
    "math": 125.0,
    "longitudinal": 125.0,
}

DATASET_PREPROCESSORS: dict[str, Callable[..., BaseSpectrogramPreprocessor]] = {
    "seed": SEEDTimeSeriesPreprocessor,
    "parkinsons": ParkinsonsTimeSeriesPreprocessor,
    "math": MathTimeSeriesPreprocessor,
    "longitudinal": LongitudinalTimeSeriesPreprocessor,
}


def _build_preprocessor(
    dataset: str,
    *,
    settings,
    nsamps: int,
    ovr_perc: float,
    fs_override: float | None,
    include_math_trials: bool,
) -> BaseSpectrogramPreprocessor:
    if dataset not in DATASET_PREPROCESSORS:
        raise ValueError(f"Unsupported dataset '{dataset}'. Options: {sorted(DATASET_PREPROCESSORS)}")

    cls = DATASET_PREPROCESSORS[dataset]
    fs = fs_override if fs_override is not None else DATASET_DEFAULT_FS[dataset]

    kwargs = {"settings": settings, "nsamps": nsamps, "ovr_perc": ovr_perc, "fs": fs}
    if dataset == "math":
        kwargs["include_math_trials"] = include_math_trials

    return cls(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess time-series EEG datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to the config file.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing data.")
    parser.add_argument("--all", action="store_true", help="Preprocess all datasets defined in the config.")
    parser.add_argument(
        "datasets",
        nargs="*",
        default=["math", "parkinsons", "seed", "longitudinal"],
        help="Datasets to preprocess (ignored if --all is set).",
    )
    parser.add_argument("--nsamps", type=int, default=2048, help="Window length for time-series examples.")
    parser.add_argument(
        "--ovr-perc",
        type=float,
        default=0.5,
        help="Overlap percentage between windows (0.0-1.0).",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=None,
        help="Target sampling rate after decimation (overrides dataset defaults).",
    )
    parser.add_argument(
        "--include-math-trials",
        action="store_true",
        help="Include math trials (state=2) for Math dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for subject split assignment.",
    )
    args = parser.parse_args()

    settings = load_settings(args.config)
    dataset_names = list(settings.datasets.keys()) if args.all else args.datasets

    logger.info(f"Loading datasets from          {settings.data_root}")
    logger.info(f"Saving time-series outputs to {settings.output_root}")
    logger.info(f"Datasets: {dataset_names}")

    splits = {"train": 0.8, "val": 0.1, "test": 0.1}

    for name in dataset_names:
        preprocessor = _build_preprocessor(
            name,
            settings=settings,
            nsamps=args.nsamps,
            ovr_perc=args.ovr_perc,
            fs_override=args.fs,
            include_math_trials=args.include_math_trials,
        )
        logger.info(
            f"Preprocessing {name} (nsamps={args.nsamps}, ovr_perc={args.ovr_perc}, fs={args.fs or DATASET_DEFAULT_FS[name]})"
        )
        preprocessor.preprocess(
            splits=splits,
            seed=args.seed,
            resolution=args.nsamps,
            overwrite=args.overwrite,
        )
    logger.info("Time-series preprocessing complete.")


if __name__ == "__main__":
    main()
