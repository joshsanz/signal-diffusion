"""Quick dry-run utility for time-series preprocessors.

Creates a handful of normalized .npy windows and metadata for a single subject
without running the full dataset. Optionally seeds stub normalization stats to
avoid long computations on first run. Can also emit a minimal spectrogram run
to ensure the legacy path remains intact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from signal_diffusion.config import load_settings
from signal_diffusion.data.base import BaseSpectrogramPreprocessor, MetadataRecord
from signal_diffusion.data.channel_maps import (
    longitudinal_channels,
    parkinsons_channels,
    seed_channels,
)
from signal_diffusion.data.math import MathPreprocessor, MathTimeSeriesPreprocessor
from signal_diffusion.data.parkinsons import (
    ParkinsonsPreprocessor,
    ParkinsonsTimeSeriesPreprocessor,
)
from signal_diffusion.data.seed import SEEDPreprocessor, SEEDTimeSeriesPreprocessor
from signal_diffusion.data.longitudinal import (
    LongitudinalPreprocessor,
    LongitudinalTimeSeriesPreprocessor,
)


DatasetConfig = tuple[Callable[..., BaseSpectrogramPreprocessor], int, int, str]

DATASETS: dict[str, DatasetConfig] = {
    "seed": (SEEDTimeSeriesPreprocessor, 250, len(seed_channels), "seed_normalization_stats.json"),
    "parkinsons": (
        ParkinsonsTimeSeriesPreprocessor,
        250,
        len(parkinsons_channels),
        "parkinsons_normalization_stats.json",
    ),
    "math": (
        MathTimeSeriesPreprocessor,
        250,
        len(seed_channels),  # Math recordings use a SEED-aligned 62-channel layout
        "math_normalization_stats.json",
    ),
    "longitudinal": (
        LongitudinalTimeSeriesPreprocessor,
        125,
        len(longitudinal_channels),
        "longitudinal_normalization_stats.json",
    ),
}


def _ensure_stub_stats(path: Path, n_channels: int) -> None:
    """Write stub stats if the file does not exist."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    zeros = [0.0] * n_channels
    ones = [1.0] * n_channels
    payload = {
        "channel_means": zeros,
        "channel_stds": ones,
        "n_samples_per_channel": [0] * n_channels,
    }
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def run_dry_run(
    *,
    dataset: str,
    settings_path: Path,
    nsamps: int,
    ovr_perc: float,
    max_examples: int,
    limit_subjects: int,
    include_math_trials: bool,
    stub_stats: bool,
    spectrogram_check: bool,
) -> None:
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset '{dataset}'. Options: {sorted(DATASETS)}")

    cls, default_fs, n_channels, stats_filename = DATASETS[dataset]
    settings = load_settings(settings_path)
    dataset_settings = settings.dataset(dataset)
    output_root = dataset_settings.timeseries_output or dataset_settings.output
    stats_path = output_root / stats_filename

    if stub_stats:
        _ensure_stub_stats(stats_path, n_channels)

    preprocessor_kwargs: dict[str, Any] = {
        "settings": settings,
        "nsamps": nsamps,
        "ovr_perc": ovr_perc,
        "fs": default_fs,
    }
    if dataset == "math":
        preprocessor_kwargs["include_math_trials"] = include_math_trials

    preprocessor = cls(**preprocessor_kwargs)

    # Narrow to a small subject set for the dry run.
    subjects = list(preprocessor.subjects())[:limit_subjects]
    if hasattr(preprocessor, "_subject_ids"):
        preprocessor._subject_ids = tuple(subjects)  # type: ignore[attr-defined]

    metadata_by_split: dict[str, list[MetadataRecord]] = {"train": []}
    written = 0

    for subject_id in subjects:
        for example in preprocessor.generate_examples(
            subject_id=subject_id, split="train", resolution=nsamps
        ):
            record = preprocessor._persist_example(example, split="train")
            metadata_by_split["train"].append(record)
            written += 1
            if written >= max_examples:
                break
        if written >= max_examples:
            break

    preprocessor._write_metadata_files(metadata_by_split)
    print(
        f"[dry-run] dataset={dataset}, subjects={len(subjects)}, examples={written}, "
        f"output_root={preprocessor.output_dir} (timeseries)"
    )
    print(f"[dry-run] metadata: {preprocessor.output_dir / 'train-metadata.csv'}")
    if stub_stats:
        print(f"[dry-run] stub stats ensured at {stats_path}")
    if spectrogram_check:
        _run_spectrogram_check(settings_path, dataset, nsamps, ovr_perc, limit_subjects)


def _run_spectrogram_check(
    settings_path: Path,
    dataset: str,
    nsamps: int,
    ovr_perc: float,
    limit_subjects: int,
) -> None:
    """Minimal spectrogram run to confirm the legacy path remains healthy."""
    spectro_map: dict[str, Callable[..., BaseSpectrogramPreprocessor]] = {
        "seed": SEEDPreprocessor,
        "parkinsons": ParkinsonsPreprocessor,
        "math": MathPreprocessor,
        "longitudinal": LongitudinalPreprocessor,
    }
    if dataset not in spectro_map:
        print(f"[spectrogram-check] Dataset '{dataset}' not supported; skipping.")
        return

    settings = load_settings(settings_path)
    cls = spectro_map[dataset]
    kwargs: dict[str, Any] = {"settings": settings, "nsamps": nsamps, "ovr_perc": ovr_perc}
    preprocessor = cls(**kwargs)

    subjects = list(preprocessor.subjects())[:limit_subjects]
    if hasattr(preprocessor, "_subject_ids"):
        preprocessor._subject_ids = tuple(subjects)  # type: ignore[attr-defined]

    metadata_by_split: dict[str, list[MetadataRecord]] = {"train": []}
    written = 0

    for subject_id in subjects:
        for example in preprocessor.generate_examples(
            subject_id=subject_id, split="train", resolution=nsamps
        ):
            record = preprocessor._persist_example(example, split="train")
            metadata_by_split["train"].append(record)
            written += 1
            break  # only need one example
        if written >= 1:
            break

    preprocessor._write_metadata_files(metadata_by_split)
    metadata_path = preprocessor.output_dir / "train-metadata.csv"
    print(
        f"[spectrogram-check] dataset={dataset}, wrote {written} example to {preprocessor.dataset_settings.output}"
    )
    print(f"[spectrogram-check] metadata: {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run time-series preprocessors.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--settings", type=Path, default=Path("config/default.toml"))
    parser.add_argument("--nsamps", type=int, default=512)
    parser.add_argument("--ovr-perc", type=float, default=0.0, help="Overlap percentage (0.0-1.0)")
    parser.add_argument("--max-examples", type=int, default=4, help="Maximum examples to write")
    parser.add_argument("--limit-subjects", type=int, default=1, help="Limit subjects processed")
    parser.add_argument(
        "--include-math-trials",
        action="store_true",
        help="Include math trials (state=2) for math dataset",
    )
    parser.add_argument(
        "--stub-stats",
        action="store_true",
        help="Seed stub normalization stats if missing (avoids full pass).",
    )
    parser.add_argument(
        "--spectrogram-check",
        action="store_true",
        help="Also emit a single spectrogram example to verify legacy path.",
    )
    args = parser.parse_args()

    run_dry_run(
        dataset=args.dataset,
        settings_path=args.settings,
        nsamps=args.nsamps,
        ovr_perc=args.ovr_perc,
        max_examples=args.max_examples,
        limit_subjects=args.limit_subjects,
        include_math_trials=args.include_math_trials,
        stub_stats=args.stub_stats,
        spectrogram_check=args.spectrogram_check,
    )


if __name__ == "__main__":
    main()
