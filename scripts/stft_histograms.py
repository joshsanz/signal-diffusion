"""
Compute raw STFT dB histograms and CDFs for all EEG datasets.

The script mirrors each dataset preprocessor's windowing (nsamps, overlap,
sampling rate, bin spacing) to produce dB-valued STFTs before clipping and
scaling. Histograms (rounded to 1 dB) are saved to disk alongside summary
statistics and recommended clip ranges for 8-bit quantization. Quantization
metadata (dB shift/scale and linear reconstruction hints) is recorded so that
uint8 dB bins can be mapped back to their bin-center magnitudes.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import pkgutil
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, Iterator

import librosa
import mne
import numpy as np
from scipy.signal import decimate
from scipy.stats import rankdata, wasserstein_distance

from signal_diffusion.config import load_settings
from signal_diffusion.data.base import BaseSpectrogramPreprocessor
from signal_diffusion.data.utils.multichannel_spectrograms import log_rstft

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional
    plt = None

mne.set_log_level("WARNING")

logger = logging.getLogger("stft_histograms")
LOWER_PCT_SWEEP = [float(i) for i in range(50)]  # 0%, 1%, ..., 49%
UPPER_PCT_SWEEP = [100.0 - 0.5 * i for i in range(5)]
HISTOGRAM_BINS = 1000


@dataclass(frozen=True)
class Uint8DbQuantizer:
    """Clip dB values to a range and map them to/from uint8 bin centers."""

    lower_db: float
    upper_db: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.lower_db) or not np.isfinite(self.upper_db):
            raise ValueError("Quantizer bounds must be finite.")
        if self.upper_db <= self.lower_db:
            raise ValueError("Quantizer upper_db must exceed lower_db.")

    @property
    def span_db(self) -> float:
        return float(self.upper_db - self.lower_db)

    @property
    def step_db(self) -> float:
        return float(self.span_db / 255.0)

    @property
    def scale_db_to_uint8(self) -> float:
        """Multiplicative factor that maps dB deltas into [0, 255] bins."""
        return float(255.0 / self.span_db)

    def clip_db(self, value_db: float) -> float:
        return float(min(max(value_db, self.lower_db), self.upper_db))

    def to_uint8(self, value_db: float) -> int:
        """Clip dB value then quantize to nearest uint8 bin index."""
        clipped = self.clip_db(value_db)
        bin_index = int(round((clipped - self.lower_db) * self.scale_db_to_uint8))
        return int(min(max(bin_index, 0), 255))

    def bin_center_db(self, bin_index: int) -> float:
        """Return the dB value at the center of a given uint8 bin."""
        return float(self.lower_db + self.step_db * min(max(bin_index, 0), 255))

    @staticmethod
    def db_to_magnitude(value_db: float) -> float:
        """Convert dB back to linear magnitude."""
        return float(10.0 ** (value_db / 20.0))

    def bin_center_magnitude(self, bin_index: int) -> float:
        return self.db_to_magnitude(self.bin_center_db(bin_index))

    def metadata(self) -> dict[str, float | str | int]:
        """Expose scale/shift to reconstruct linear magnitudes from uint8 bins."""
        return {
            "lower_db": self.lower_db,
            "upper_db": self.upper_db,
            "span_db": self.span_db,
            "step_db_per_bin": self.step_db,
            "scale_db_to_uint8": self.scale_db_to_uint8,
            "uint8_zero_point": 0,
            "uint8_max": 255,
            "bin0_linear": self.bin_center_magnitude(0),
            "bin255_linear": self.bin_center_magnitude(255),
            "reconstruction": "db = lower_db + step_db_per_bin * uint8; magnitude = 10**(db/20)",
        }


def sliding_blocks(data: np.ndarray, nsamps: int, noverlap: int) -> Iterator[np.ndarray]:
    """Yield consecutive windows from a (channels, samples) array."""
    shift = nsamps - noverlap
    if shift <= 0:
        raise ValueError("Overlap percentage results in non-positive shift size")
    start = 0
    end = nsamps
    total = data.shape[1]
    while end <= total:
        yield data[:, start:end]
        start += shift
        end += shift


def channel_stft_db(
    channel: np.ndarray,
    *,
    hop_length: int,
    win_length: int,
    bin_spacing: str,
) -> np.ndarray:
    """Compute channel STFT in dB without clipping or scaling."""
    if bin_spacing == "linear":
        stft = librosa.stft(
            y=channel,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=win_length * 2 - 1,
        )
    elif bin_spacing == "log":
        _, stft = log_rstft(
            x=channel,
            win_length=win_length * 2 - 1,
            hop_length=hop_length,
            window="hann",
        )
    else:
        raise ValueError(f"Invalid bin spacing {bin_spacing}")
    # Convert linear magnitudes to dB without clipping; epsilon avoids log(0).
    return 20.0 * np.log10(np.abs(stft) + 1e-9)


def percentile_from_counts(counts: Counter[float], percentile: float) -> float:
    """Compute percentile from a histogram represented as value->count."""
    if not counts:
        return float("nan")
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    target = (percentile / 100.0) * total
    running = 0
    for value in sorted(counts):
        running += counts[value]
        if running >= target:
            return float(value)
    return float(sorted(counts)[-1])


def fraction_in_range(counts: Counter[float], low: float, high: float) -> float:
    """Return fraction of total counts within [low, high]."""
    if not counts:
        return 0.0
    total = sum(counts.values())
    covered = sum(count for value, count in counts.items() if low <= value <= high)
    return covered / total if total else 0.0


def rebin_counts(
    counts: Counter[int], *, bins: int, min_db: float | None, max_db: float | None
) -> Counter[float]:
    """Re-bin integer-rounded counts into fixed-width bins across [min_db, max_db]."""
    if not counts or min_db is None or max_db is None:
        return Counter()
    if max_db <= min_db:
        # Avoid zero-width by expanding a tiny epsilon.
        max_db = min_db + 1e-3
    edges = np.linspace(min_db, max_db, bins + 1)
    centers = edges[:-1] + np.diff(edges) / 2.0
    rebinned: Counter[float] = Counter()
    for value, count in counts.items():
        # Clip to ensure assignment within edges.
        idx = int(np.searchsorted(edges, value, side="right") - 1)
        idx = max(0, min(bins - 1, idx))
        center = float(centers[idx])
        rebinned[center] += count
    return rebinned


def clip_bounds(counts: Counter[float], lower_pct: float, upper_pct: float) -> tuple[float | None, float | None]:
    """Resolve percentile bounds into clipping thresholds without rounding."""
    lower = percentile_from_counts(counts, lower_pct)
    upper = percentile_from_counts(counts, upper_pct)
    if not np.isfinite(lower) or not np.isfinite(upper):
        return None, None
    if upper <= lower:
        upper = lower + 1e-6
    return float(lower), float(upper)


def build_quantizer(lower_db: float | None, upper_db: float | None) -> Uint8DbQuantizer | None:
    """Safely construct a quantizer if bounds are valid and non-empty."""
    if lower_db is None or upper_db is None:
        return None
    try:
        return Uint8DbQuantizer(float(lower_db), float(upper_db))
    except ValueError:
        return None


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Jensen-Shannon divergence between two probability vectors."""
    m = 0.5 * (p + q)
    mask_p = p > 0
    mask_q = q > 0
    js = 0.0
    if np.any(mask_p):
        js += 0.5 * np.sum(p[mask_p] * np.log(p[mask_p] / (m[mask_p] + 1e-12)))
    if np.any(mask_q):
        js += 0.5 * np.sum(q[mask_q] * np.log(q[mask_q] / (m[mask_q] + 1e-12)))
    return float(js)


def compute_quantization_metrics(
    counts: Counter[float], quantizer: Uint8DbQuantizer | None
) -> dict[str, float] | None:
    """Compute metrics after clipping to [lower, upper] then quantizing to uint8 dB bins.

    Steps:
    1) Clip incoming dB values to the quantizer bounds.
    2) Map clipped dB values into uint8 bins with centers spaced by step_db.
    3) Compute metrics using:
       - entropy over quantized linear magnitudes,
       - JS divergence between the clipped dB distribution and quantized bin centers,
       - Wasserstein distance between those same dB distributions,
       - PSNR between clipped dB values and their quantized bin-center reconstruction.
    """
    if quantizer is None:
        return None

    clipped_counts: Counter[int] = Counter()
    quantized_db_counts: Counter[float] = Counter()
    quantized_mag_counts: Counter[float] = Counter()
    mse_accum = 0.0
    total = 0
    for value, count in counts.items():
        clipped_value = quantizer.clip_db(value)
        clipped_counts[clipped_value] += count
        bin_index = quantizer.to_uint8(clipped_value)
        bin_center_db = quantizer.bin_center_db(bin_index)
        magnitude = quantizer.bin_center_magnitude(bin_index)
        quantized_db_counts[bin_center_db] += count
        quantized_mag_counts[magnitude] += count
        mse_accum += count * (clipped_value - bin_center_db) ** 2
        total += count
    if total == 0:
        return None

    # Entropy over magnitudes.
    entropy_probs = np.fromiter(quantized_mag_counts.values(), dtype=np.float64) / float(total)
    entropy_val = float(-np.sum(entropy_probs * np.log(entropy_probs + 1e-12)))

    # JS divergence over dB distributions aligned on shared support.
    support = sorted(set(clipped_counts.keys()) | set(quantized_db_counts.keys()))
    p = np.array([clipped_counts.get(val, 0) for val in support], dtype=np.float64)
    q = np.array([quantized_db_counts.get(val, 0) for val in support], dtype=np.float64)
    p /= p.sum()
    q /= q.sum()
    js_val = js_divergence(p, q)

    # Wasserstein distance over dB values using weights.
    wass_val = float(
        wasserstein_distance(
            list(clipped_counts.keys()),
            list(quantized_db_counts.keys()),
            u_weights=list(clipped_counts.values()),
            v_weights=list(quantized_db_counts.values()),
        )
    )

    # PSNR between clipped dB values and quantized bin centers.
    mse = mse_accum / float(total)
    if mse <= 0:
        psnr_val = float("inf")
    else:
        peak = float(quantizer.span_db)
        psnr_val = 20.0 * np.log10(peak / np.sqrt(mse) + 1e-12)

    return {
        "entropy": entropy_val,
        "js_divergence": js_val,
        "wasserstein_db": wass_val,
        "psnr_db": psnr_val,
    }


class HistogramAccumulator:
    """Collect histogram statistics for a dataset."""

    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.counts: Counter[float] = Counter()
        self.window_count = 0
        self.value_count = 0
        self.min_db: int | None = None
        self.max_db: int | None = None

    def add_values(self, values_db: np.ndarray) -> None:
        rounded = np.rint(values_db).astype(np.int64)
        if rounded.size == 0:
            return
        unique, counts = np.unique(rounded, return_counts=True)
        self.value_count += int(counts.sum())
        for value, count in zip(unique, counts):
            self.counts[int(value)] += int(count)
        min_val = int(unique.min())
        max_val = int(unique.max())
        self.min_db = min_val if self.min_db is None else min(self.min_db, min_val)
        self.max_db = max_val if self.max_db is None else max(self.max_db, max_val)

    def increment_window(self) -> None:
        self.window_count += 1

    def summary(
        self,
        *,
        lower_pct: float,
        upper_pct: float,
        lower_pct_candidates: Iterable[float] | None = None,
        upper_pct_candidates: Iterable[float] | None = None,
    ) -> dict[str, object]:
        """Summarize histogram stats and run quantization metric sweeps."""
        # Re-bin to fixed-width bins for downstream metrics and saved histograms.
        self.counts = rebin_counts(self.counts, bins=HISTOGRAM_BINS, min_db=self.min_db, max_db=self.max_db)
        p1 = percentile_from_counts(self.counts, 1.0)
        p99 = percentile_from_counts(self.counts, 99.0)
        lower_int, upper_int = clip_bounds(self.counts, lower_pct, upper_pct)
        coverage = (
            fraction_in_range(self.counts, lower_int, upper_int)
            if lower_int is not None and upper_int is not None
            else 0.0
        )
        # Track the exact uint8↔dB mapping so metric calculations and reconstructions are transparent.
        quantizer = build_quantizer(lower_int, upper_int)
        quantization_info = quantizer.metadata() if quantizer else None
        step = quantizer.step_db if quantizer else None
        entropy_candidates: list[dict[str, object]] = []
        best_by_metric: dict[str, dict[str, object] | None] = {
            "entropy": None,
            "js_divergence": None,
            "wasserstein_db": None,
            "psnr_db": None,
        }
        if lower_pct_candidates:
            upper_candidates = list(upper_pct_candidates) if upper_pct_candidates else [upper_pct]
            for lower_candidate in lower_pct_candidates:
                for upper_candidate in upper_candidates:
                    cand_lower_int, cand_upper_int = clip_bounds(self.counts, lower_candidate, upper_candidate)
                    cand_quantizer = build_quantizer(cand_lower_int, cand_upper_int)
                    metrics = compute_quantization_metrics(self.counts, cand_quantizer)
                    if metrics is None:
                        continue
                    candidate_record: dict[str, object] = {
                        "lower_percentile": lower_candidate,
                        "upper_percentile": upper_candidate,
                        "lower_db": cand_lower_int,
                        "upper_db": cand_upper_int,
                        "quantization": cand_quantizer.metadata() if cand_quantizer else None,
                        "entropy": metrics["entropy"],
                        "js_divergence": metrics["js_divergence"],
                        "wasserstein_db": metrics["wasserstein_db"],
                        "psnr_db": metrics["psnr_db"],
                    }
                    entropy_candidates.append(candidate_record)
                    # Track best per metric.
                    if (
                        best_by_metric["entropy"] is None
                        or metrics["entropy"] > best_by_metric["entropy"]["entropy"]  # type: ignore[index]
                    ):
                        best_by_metric["entropy"] = candidate_record
                    if (
                        best_by_metric["js_divergence"] is None
                        or metrics["js_divergence"]
                        < best_by_metric["js_divergence"]["js_divergence"]  # type: ignore[index]
                    ):
                        best_by_metric["js_divergence"] = candidate_record
                    if (
                        best_by_metric["wasserstein_db"] is None
                        or metrics["wasserstein_db"]
                        < best_by_metric["wasserstein_db"]["wasserstein_db"]  # type: ignore[index]
                    ):
                        best_by_metric["wasserstein_db"] = candidate_record
                    if (
                        best_by_metric["psnr_db"] is None
                        or metrics["psnr_db"] > best_by_metric["psnr_db"]["psnr_db"]  # type: ignore[index]
                    ):
                        best_by_metric["psnr_db"] = candidate_record
        return {
            "dataset": self.dataset,
            "windows": self.window_count,
            "values": self.value_count,
            "min_db": self.min_db,
            "max_db": self.max_db,
            "p1_db": p1,
            "p99_db": p99,
            "recommended_clip": {
                "lower_db": lower_int,
                "upper_db": upper_int,
                "coverage": coverage,
                "step_db_per_bin": step,
                "quantization": quantization_info,
            },
            "entropy_sweep": {
                "upper_percentile": upper_pct,
                "candidates": entropy_candidates,
            },
            "best_clips": best_by_metric,
        }


def discover_preprocessors() -> dict[str, type[BaseSpectrogramPreprocessor]]:
    """Find dataset preprocessors living under signal_diffusion.data."""
    import signal_diffusion.data as data_pkg

    excluded = {"__pycache__", "base", "meta", "utils", "specs", "channel_maps"}
    preprocessors: dict[str, type[BaseSpectrogramPreprocessor]] = {}
    for module_info in pkgutil.iter_modules(data_pkg.__path__):
        if module_info.name.startswith("_") or module_info.name in excluded:
            continue
        module = importlib.import_module(f"signal_diffusion.data.{module_info.name}")
        for attr in dir(module):
            obj = getattr(module, attr)
            if (
                inspect.isclass(obj)
                and issubclass(obj, BaseSpectrogramPreprocessor)
                and obj is not BaseSpectrogramPreprocessor
            ):
                preprocessors[module_info.name] = obj
                break
    return preprocessors


def iter_math_blocks(preprocessor, max_windows_per_recording: int | None) -> Iterator[np.ndarray]:
    for subject_id in preprocessor.subjects():
        for state in preprocessor.states:
            windows = 0
            data = preprocessor._load_subject_state(subject_id, state)  # noqa: SLF001
            if data is None:
                continue
            for block in sliding_blocks(data, preprocessor.nsamps, preprocessor.noverlap):
                yield block
                windows += 1
                if max_windows_per_recording and windows >= max_windows_per_recording:
                    break


def iter_parkinsons_blocks(preprocessor, max_windows_per_recording: int | None) -> Iterator[np.ndarray]:
    for subject_id in preprocessor.subjects():
        windows = 0
        data = preprocessor._load_subject_data(subject_id)  # noqa: SLF001
        if data is None:
            continue
        for block in sliding_blocks(data, preprocessor.nsamps, preprocessor.noverlap):
            yield block
            windows += 1
            if max_windows_per_recording and windows >= max_windows_per_recording:
                break


def iter_longitudinal_blocks(preprocessor, max_windows_per_recording: int | None) -> Iterator[np.ndarray]:
    for subject_id in preprocessor.subjects():
        info = preprocessor._subject_metadata(subject_id)  # noqa: SLF001
        sessions = []
        if info.has_session1:
            sessions.append("ses-1")
        if info.has_session2:
            sessions.append("ses-2")
        for session in sessions:
            for acquisition in ("pre", "post"):
                windows = 0
                data = preprocessor._load_subject_data(  # noqa: SLF001
                    subject_id, session, "EyesOpen", acquisition
                )
                if data is None:
                    continue
                for block in sliding_blocks(data, preprocessor.nsamps, preprocessor.noverlap):
                    yield block
                    windows += 1
                    if max_windows_per_recording and windows >= max_windows_per_recording:
                        break


def iter_seed_blocks(preprocessor, max_windows_per_recording: int | None) -> Iterator[np.ndarray]:
    # Import constants lazily to avoid module import when dataset not requested.
    from signal_diffusion.data.seed import END_SECOND, START_SECOND

    for subject_id in preprocessor.subjects():
        for session_index, cnt_path in preprocessor.session_files.get(subject_id, []):
            windows = 0
            raw = mne.io.read_raw_cnt(cnt_path, preload=True, verbose="WARNING")
            data = raw.get_data()[preprocessor.channel_indices, :]
            if preprocessor.decimation > 1:
                data = decimate(data, preprocessor.decimation, axis=1, zero_phase=True)

            start_points = START_SECOND[session_index]
            end_points = END_SECOND[session_index]
            for trial_index in range(preprocessor.n_trials):
                start = int(start_points[trial_index] * preprocessor.fs)
                end = int(end_points[trial_index] * preprocessor.fs)
                trial_data = data[:, start:end]
                for block in sliding_blocks(trial_data, preprocessor.nsamps, preprocessor.noverlap):
                    yield block
                    windows += 1
                    if max_windows_per_recording and windows >= max_windows_per_recording:
                        break
                if max_windows_per_recording and windows >= max_windows_per_recording:
                    break
            if max_windows_per_recording and windows >= max_windows_per_recording:
                continue


ITERATORS: dict[str, Callable[[object, int | None], Iterable[np.ndarray]]] = {
    "math": iter_math_blocks,
    "parkinsons": iter_parkinsons_blocks,
    "longitudinal": iter_longitudinal_blocks,
    "seed": iter_seed_blocks,
}


def build_preprocessor(
    dataset: str,
    cls: type[BaseSpectrogramPreprocessor],
    settings,
    *,
    nsamps: int,
    ovr_perc: float,
    fs: float,
    bin_spacing: str,
    include_math_trials: bool,
) -> BaseSpectrogramPreprocessor:
    """Instantiate a dataset preprocessor with supported kwargs."""
    signature = inspect.signature(cls.__init__)
    kwargs = {}
    for name, value in {
        "nsamps": nsamps,
        "ovr_perc": ovr_perc,
        "fs": fs,
        "bin_spacing": bin_spacing,
        "include_math_trials": include_math_trials,
    }.items():
        if name in signature.parameters:
            kwargs[name] = value
    preprocessor = cls(settings, **kwargs)
    logger.info(
        "Prepared %s preprocessor nsamps=%s overlap=%.2f fs=%.2f bin_spacing=%s",
        dataset,
        preprocessor.nsamps,
        preprocessor.overlap_fraction,
        preprocessor.fs,
        preprocessor.bin_spacing,
    )
    return preprocessor


def process_dataset(
    dataset: str,
    preprocessor: BaseSpectrogramPreprocessor,
    *,
    resolution: int,
    hop_length: int | None,
    win_length: int,
    max_windows_per_recording: int | None,
    lower_pct: float,
    upper_pct: float,
    output_dir: Path,
) -> dict[str, object] | None:
    iterator = ITERATORS.get(dataset)
    if iterator is None:
        logger.warning("No iterator registered for dataset '%s'; skipping", dataset)
        return None
    if not preprocessor.dataset_settings.root.exists():
        logger.warning("Dataset root %s does not exist; skipping %s", preprocessor.dataset_settings.root, dataset)
        return None

    derived_hop = hop_length or preprocessor._derive_hop_length(resolution)  # noqa: SLF001
    accumulator = HistogramAccumulator(dataset)
    logger.info(
        "Processing %s (resolution=%d, hop_length=%d, win_length=%d, bin_spacing=%s)",
        dataset,
        resolution,
        derived_hop,
        win_length,
        preprocessor.bin_spacing,
    )

    for block in iterator(preprocessor, max_windows_per_recording):
        for channel in block:
            values_db = channel_stft_db(
                channel,
                hop_length=derived_hop,
                win_length=win_length,
                bin_spacing=preprocessor.bin_spacing,
            )
            accumulator.add_values(values_db)
        accumulator.increment_window()

    if accumulator.window_count == 0:
        logger.warning("No windows processed for dataset '%s'", dataset)
        return None

    summary = accumulator.summary(
        lower_pct=lower_pct,
        upper_pct=upper_pct,
        lower_pct_candidates=LOWER_PCT_SWEEP,
        upper_pct_candidates=UPPER_PCT_SWEEP,
    )
    summary["recommended_clip_percentiles"] = (lower_pct, upper_pct)
    counts_sorted = sorted(accumulator.counts.items())
    save_histogram(dataset, counts_sorted, summary, output_dir)
    print_dataset_summary(summary, output_dir)
    return summary


def summarize_best_by_metric(results: list[dict[str, object]]) -> list[tuple[str, dict[str, object]]]:
    """Pick the best clip per metric across datasets."""
    metrics = {
        "entropy": lambda current, best: best is None or current["entropy"] > best["entropy"],
        "js_divergence": lambda current, best: best is None or current["js_divergence"] < best["js_divergence"],
        "wasserstein_db": lambda current, best: best is None or current["wasserstein_db"] < best["wasserstein_db"],
        "psnr_db": lambda current, best: best is None or current["psnr_db"] > best["psnr_db"],
    }
    best_per_metric: dict[str, dict[str, object]] = {}
    for summary in results:
        dataset = summary.get("dataset")
        best_clips = summary.get("best_clips") or {}
        for metric, comparator in metrics.items():
            candidate = best_clips.get(metric)
            if not candidate:
                continue
            current = dict(candidate)
            current["dataset"] = dataset
            if metric not in best_per_metric or comparator(current, best_per_metric[metric]):
                best_per_metric[metric] = current
    return sorted(best_per_metric.items())


def _format_rounded(value: object, decimals: int = 1) -> str:
    """Format numeric values to nearest decimal place for table output."""
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    try:
        value = round(float(value), decimals)
        return f"{value}"
    except (TypeError, ValueError):
        return str(value)


def print_metric_recommendations(results: list[dict[str, object]]) -> None:
    """Print metric-grouped recommendations across datasets as a table."""
    tables = metric_tables(results)
    if not tables:
        return
    print("Metric-grouped recommendations (per-dataset best for each metric):")
    for metric, rows in tables.items():
        print(f"  {metric}:")
        header = (
            f"    {'dataset':<14}{'lower_pct':>12}{'upper_pct':>12}"
            f"{'lower_db':>12}{'upper_db':>12}{'min_db':>10}{'max_db':>10}{'value':>14}"
        )
        print(header)
        print("    " + "-" * (len(header) - 4))
        for row in rows:
            print(
                f"    {row['dataset']:<14}"
                f"{_format_rounded(row.get('lower_percentile')):>12}"
                f"{_format_rounded(row.get('upper_percentile')):>12}"
                f"{_format_rounded(row.get('lower_db'), 2):>12}"
                f"{_format_rounded(row.get('upper_db'), 2):>12}"
                f"{_format_rounded(row.get('min_db'), 2):>10}"
                f"{_format_rounded(row.get('max_db'), 2):>10}"
                f"{_format_rounded(row.get('value'), 4):>14}"
            )
        print()


def metric_recommendations(results: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return serialized best-per-metric recommendations across datasets."""
    recs: list[dict[str, object]] = []
    for metric, rec in summarize_best_by_metric(results):
        item = dict(rec)
        item["metric"] = metric
        recs.append(item)
    return recs


def metric_tables(results: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    """Return per-metric tables of per-dataset best clips."""
    metrics = ["entropy", "js_divergence", "wasserstein_db", "psnr_db"]
    tables: dict[str, list[dict[str, object]]] = {}
    for metric in metrics:
        rows: list[dict[str, object]] = []
        for summary in results:
            best = (summary.get("best_clips") or {}).get(metric)
            if not best:
                continue
            row = {
                "dataset": summary.get("dataset"),
                "min_db": summary.get("min_db"),
                "max_db": summary.get("max_db"),
                "lower_percentile": best.get("lower_percentile"),
                "upper_percentile": best.get("upper_percentile"),
                "lower_db": best.get("lower_db"),
                "upper_db": best.get("upper_db"),
                "value": best.get(metric),
            }
            rows.append(row)
        if rows:
            tables[metric] = sorted(rows, key=lambda r: str(r.get("dataset")))
    return tables


def _spearman(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation."""
    rx = rankdata(x)
    ry = rankdata(y)
    rx_mean = float(np.mean(rx))
    ry_mean = float(np.mean(ry))
    cov = float(np.sum((rx - rx_mean) * (ry - ry_mean)))
    var_x = float(np.sum((rx - rx_mean) ** 2))
    var_y = float(np.sum((ry - ry_mean) ** 2))
    if var_x <= 0 or var_y <= 0:
        return float("nan")
    return cov / np.sqrt(var_x * var_y)


def analyze_metric_correlations(results: list[dict[str, object]]) -> dict[str, object]:
    """Compute pairwise Spearman correlations across candidate metrics."""
    rows: list[dict[str, float]] = []
    metric_defs = {
        "psnr_db": 1.0,
        "js_divergence": -1.0,  # lower is better, flip sign for correlation
        "wasserstein_db": -1.0,  # lower is better, flip sign for correlation
        "entropy": 1.0,
    }
    for summary in results:
        sweep = summary.get("entropy_sweep") or {}
        for cand in sweep.get("candidates") or []:
            psnr = cand.get("psnr_db")
            js = cand.get("js_divergence")
            wass = cand.get("wasserstein_db")
            ent = cand.get("entropy")
            if psnr is None or not np.isfinite(psnr) or js is None or wass is None or ent is None:
                continue
            rows.append(
                {
                    "psnr_db": float(psnr) * metric_defs["psnr_db"],
                    "js_divergence": float(js) * metric_defs["js_divergence"],
                    "wasserstein_db": float(wass) * metric_defs["wasserstein_db"],
                    "entropy": float(ent) * metric_defs["entropy"],
                }
            )
    if not rows:
        return {"samples": 0, "pairwise": []}

    metrics = list(metric_defs.keys())
    value_lists = {m: [row[m] for row in rows] for m in metrics}
    correlations: list[dict[str, object]] = []
    for i, a in enumerate(metrics):
        for b in metrics[i + 1 :]:
            rho = _spearman(value_lists[a], value_lists[b])
            correlations.append({"metrics": [a, b], "spearman": float(rho)})
    return {
        "samples": len(rows),
        "pairwise": correlations,
        "direction_note": "js_divergence and wasserstein_db are sign-flipped so higher is better for correlation.",
    }


def _clip_signature(record: dict[str, object] | None) -> tuple[object, ...] | None:
    """Normalize clip bounds for equality checks."""
    if not record:
        return None
    return (
        record.get("lower_db"),
        record.get("upper_db"),
        record.get("lower_percentile"),
        record.get("upper_percentile"),
    )


def analyze_clip_agreements(results: list[dict[str, object]]) -> dict[str, object]:
    """Analyze how often metrics agree on optimal clip bounds."""
    metrics = ["entropy", "js_divergence", "psnr_db", "wasserstein_db"]
    per_dataset: list[dict[str, object]] = []
    for summary in results:
        best = summary.get("best_clips") or {}
        per_dataset.append(
            {
                "dataset": summary.get("dataset"),
                "clip_bounds": {metric: _clip_signature(best.get(metric)) for metric in metrics},
            }
        )

    pairwise: list[dict[str, object]] = []
    total = len(results)
    for i, a in enumerate(metrics):
        for b in metrics[i + 1 :]:
            matches = 0
            for summary in results:
                best = summary.get("best_clips") or {}
                if _clip_signature(best.get(a)) == _clip_signature(best.get(b)):
                    matches += 1
            pairwise.append({"metrics": [a, b], "matching_datasets": matches, "total_datasets": total})
    return {"pairwise": pairwise, "per_dataset": per_dataset}


def _reshape_candidates_for_contour(
    candidates: list[dict[str, object]], metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lower/upper percentile grids and metric surface for contour plots."""
    lower_vals = sorted({float(c["lower_percentile"]) for c in candidates if "lower_percentile" in c})
    upper_vals = sorted({float(c["upper_percentile"]) for c in candidates if "upper_percentile" in c})
    if not lower_vals or not upper_vals:
        return np.array([]), np.array([]), np.array([])
    lower_index = {val: idx for idx, val in enumerate(lower_vals)}
    upper_index = {val: idx for idx, val in enumerate(upper_vals)}
    surface = np.full((len(upper_vals), len(lower_vals)), np.nan, dtype=float)
    for cand in candidates:
        lower = cand.get("lower_percentile")
        upper = cand.get("upper_percentile")
        value = cand.get(metric)
        if lower is None or upper is None or value is None:
            continue
        surface[upper_index[float(upper)], lower_index[float(lower)]] = float(value)
    return np.array(lower_vals), np.array(upper_vals), surface


def plot_metric_landscapes(dataset: str, summary: dict[str, object], output_dir: Path) -> None:
    """Create colored contour plots for each metric across percentile sweeps."""
    if not plt:
        return
    sweep = summary.get("entropy_sweep") or {}
    candidates = sweep.get("candidates") or []
    if not candidates:
        return
    metric_labels = {
        "entropy": "Entropy",
        "js_divergence": "JS divergence",
        "wasserstein_db": "Wasserstein (dB)",
        "psnr_db": "PSNR (dB)",
    }
    metric_preferences = {
        "entropy": "↑ higher is better",
        "js_divergence": "↓ lower is better",
        "wasserstein_db": "↓ lower is better",
        "psnr_db": "↑ higher is better",
    }
    for metric, label in metric_labels.items():
        lower_vals, upper_vals, surface = _reshape_candidates_for_contour(candidates, metric)
        if surface.size == 0 or np.all(np.isnan(surface)):
            continue
        X, Y = np.meshgrid(lower_vals, upper_vals)
        masked_surface = np.ma.masked_invalid(surface)
        fig, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(X, Y, masked_surface, levels=20, cmap="viridis")
        fig.colorbar(contour, ax=ax, label=label)
        ax.set_xlabel("Lower percentile bound")
        ax.set_ylabel("Upper percentile bound")
        preference = metric_preferences.get(metric)
        title_suffix = f" ({preference})" if preference else ""
        ax.set_title(f"{dataset}: {label} contour{title_suffix}")
        fig.tight_layout()
        fig.savefig(output_dir / f"stft_{metric}_contour_{dataset}.png")
        plt.close(fig)


def save_histogram(dataset: str, counts: list[tuple[int, int]], summary: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    hist_json = output_dir / f"stft_hist_{dataset}.json"
    hist_csv = output_dir / f"stft_hist_{dataset}.csv"
    summary_json = output_dir / f"stft_hist_summary_{dataset}.json"
    cdf_png = output_dir / f"stft_cdf_{dataset}.png"

    hist_data = [{"db": value, "count": count} for value, count in counts]
    hist_json.write_text(json.dumps({"dataset": dataset, "histogram": hist_data}, indent=2))

    with hist_csv.open("w", encoding="utf-8") as handle:
        handle.write("db,count\n")
        for value, count in counts:
            handle.write(f"{value},{count}\n")

    summary_json.write_text(json.dumps(summary, indent=2))

    if plt and counts:
        values, freqs = zip(*counts)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(values, freqs, width=1.0)
        ax.set_xlabel("dB (rounded)")
        ax.set_ylabel("Count")
        ax.set_title(f"STFT histogram: {dataset}")
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(output_dir / f"stft_hist_{dataset}.png")
        plt.close(fig)

        # CDF plot on linear scale for coverage intuition
        total = float(sum(freqs))
        cumulative = np.cumsum(freqs) / total
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(values, cumulative, color="blue")
        ax.set_xlabel("dB (rounded)")
        ax.set_ylabel("CDF")
        ax.set_title(f"STFT CDF: {dataset}")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(cdf_png)
        plt.close(fig)
        plot_metric_landscapes(dataset, summary, output_dir)


def print_dataset_summary(summary: dict[str, object], output_dir: Path) -> None:
    clip = summary.get("recommended_clip", {})
    lower = clip.get("lower_db")
    upper = clip.get("upper_db")
    coverage = clip.get("coverage", 0.0)
    quantization = clip.get("quantization") or {}
    step = quantization.get("step_db_per_bin") or clip.get("step_db_per_bin")
    linear_min = quantization.get("bin0_linear")
    linear_max = quantization.get("bin255_linear")
    best_clips = summary.get("best_clips") or {}
    best_entropy = best_clips.get("entropy") or {}
    best_js = best_clips.get("js_divergence") or {}
    best_wasserstein = best_clips.get("wasserstein_db") or {}
    best_psnr = best_clips.get("psnr_db") or {}
    best_lower_pct = best_entropy.get("lower_percentile")
    best_upper_pct = best_entropy.get("upper_percentile")
    best_lower_db = best_entropy.get("lower_db")
    best_upper_db = best_entropy.get("upper_db")
    best_entropy_val = best_entropy.get("entropy")
    requested_lower_pct, requested_upper_pct = summary.get("recommended_clip_percentiles", (None, None))
    # Log user-requested percentile recommendation.
    logger.info(
        "[%s] requested clip (lower_pct=%.2f upper_pct=%.2f): [%s,%s] coverage=%.2f%% step≈%s dB",
        summary.get("dataset"),
        requested_lower_pct or float("nan"),
        requested_upper_pct or float("nan"),
        lower,
        upper,
        coverage * 100,
        f"{step:.2f}" if step else "n/a",
    )
    # Log entropy-driven recommendation.
    logger.info(
        "[%s] entropy-best clip (lower_pct=%s upper_pct=%s): [%s,%s] entropy=%s",
        summary.get("dataset"),
        f"{best_lower_pct:.2f}" if best_lower_pct is not None else "n/a",
        f"{best_upper_pct:.2f}" if best_upper_pct is not None else "n/a",
        best_lower_db if best_lower_db is not None else "n/a",
        best_upper_db if best_upper_db is not None else "n/a",
        f"{best_entropy_val:.4f}" if best_entropy_val is not None else "n/a",
    )
    logger.info(
        "[%s] JS-best clip: (lower_pct=%s upper_pct=%s) js=%.6f",
        summary.get("dataset"),
        f"{best_js.get('lower_percentile', 'n/a')}",
        f"{best_js.get('upper_percentile', 'n/a')}",
        best_js.get("js_divergence", float("nan")),
    )
    logger.info(
        "[%s] Wasserstein-best clip: (lower_pct=%s upper_pct=%s) wasserstein=%.6f dB",
        summary.get("dataset"),
        f"{best_wasserstein.get('lower_percentile', 'n/a')}",
        f"{best_wasserstein.get('upper_percentile', 'n/a')}",
        best_wasserstein.get("wasserstein_db", float("nan")),
    )
    logger.info(
        "[%s] PSNR-best clip: (lower_pct=%s upper_pct=%s) psnr=%s dB",
        summary.get("dataset"),
        f"{best_psnr.get('lower_percentile', 'n/a')}",
        f"{best_psnr.get('upper_percentile', 'n/a')}",
        f"{best_psnr.get('psnr_db', float('nan')):.4f}"
        if best_psnr.get("psnr_db") is not None
        else "n/a",
    )
    if quantization:
        logger.info(
            "[%s] uint8 mapping: db = lower_db + step_db_per_bin * uint8; magnitude = 10**(db/20) "
            "(linear range≈[%s,%s])",
            summary.get("dataset"),
            f"{linear_min:.3e}" if linear_min is not None else "n/a",
            f"{linear_max:.3e}" if linear_max is not None else "n/a",
        )
    dataset_label = summary.get("dataset")
    print(
        f"{dataset_label}: requested_clip(lower_pct={requested_lower_pct}, upper_pct={requested_upper_pct}) "
        f"db=[{lower},{upper}] coverage={coverage*100:.2f}% step≈{step if step else 'n/a'} dB "
        f"(files in {output_dir})"
    )
    print(
        f"{dataset_label}: entropy_best(lower_pct={best_lower_pct if best_lower_pct is not None else 'n/a'}, "
        f"upper_pct={best_upper_pct if best_upper_pct is not None else 'n/a'}) "
        f"db=[{best_lower_db if best_lower_db is not None else 'n/a'},"
        f"{best_upper_db if best_upper_db is not None else 'n/a'}] "
        f"entropy={best_entropy_val if best_entropy_val is not None else 'n/a'}"
    )
    print(
        f"{dataset_label}: js_best(lower_pct={best_js.get('lower_percentile', 'n/a')}, "
        f"upper_pct={best_js.get('upper_percentile', 'n/a')}) "
        f"js_divergence={best_js.get('js_divergence', 'n/a')}"
    )
    print(
        f"{dataset_label}: wasserstein_best(lower_pct={best_wasserstein.get('lower_percentile', 'n/a')}, "
        f"upper_pct={best_wasserstein.get('upper_percentile', 'n/a')}) "
        f"wasserstein_db={best_wasserstein.get('wasserstein_db', 'n/a')}"
    )
    print(
        f"{dataset_label}: psnr_best(lower_pct={best_psnr.get('lower_percentile', 'n/a')}, "
        f"upper_pct={best_psnr.get('upper_percentile', 'n/a')}) "
        f"psnr_db={best_psnr.get('psnr_db', 'n/a')}"
    )
    if quantization:
        print(
            f"{dataset_label}: uint8 quantization uses db = {lower} + {step} * uint8 "
            f"(linear≈[{linear_min if linear_min is not None else 'n/a'},"
            f"{linear_max if linear_max is not None else 'n/a'}])"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config TOML (defaults to SIGNAL_DIFFUSION_CONFIG or config/default.toml).",
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=None,
        help="Datasets to analyze (default: all discovered preprocessors).",
    )
    parser.add_argument("--nsamps", type=int, default=2000, help="Samples per window.")
    parser.add_argument("--ovr-perc", type=float, default=0.5, help="Overlap fraction for windows (0-1).")
    parser.add_argument("--fs", type=float, default=125.0, help="Target sampling rate after decimation.")
    parser.add_argument(
        "--bin-spacing",
        choices=["linear", "log"],
        default="log",
        help="Frequency bin spacing for STFT.",
    )
    parser.add_argument("--resolution", type=int, default=256, help="Spectrogram resolution (height/width).")
    parser.add_argument(
        "--hop-length",
        type=int,
        default=None,
        help="Optional hop length override. Default matches dataset preprocessor heuristic.",
    )
    parser.add_argument(
        "--win-length",
        type=int,
        default=None,
        help="Optional window length override (default: resolution).",
    )
    parser.add_argument(
        "--max-windows-per-recording",
        type=int,
        default=None,
        help="Limit windows processed per recording file (default: all windows).",
    )
    parser.add_argument(
        "--lower-percentile",
        type=float,
        default=0.5,
        help="Lower percentile for recommended clipping floor.",
    )
    parser.add_argument(
        "--upper-percentile",
        type=float,
        default=99.5,
        help="Upper percentile for recommended clipping ceiling.",
    )
    parser.add_argument(
        "--include-math-trials",
        action="store_true",
        help="Include math trials for the Math dataset when available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory for histogram outputs (default: current working directory).",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Optional limit on subjects per dataset (processed in sorted order).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    settings = load_settings(args.config)

    discovered = discover_preprocessors()
    requested = [name.lower() for name in (args.datasets or discovered.keys())]
    preprocessors: dict[str, BaseSpectrogramPreprocessor] = {}

    for dataset in requested:
        cls = discovered.get(dataset)
        if cls is None:
            logger.warning("Dataset '%s' not found among discovered preprocessors: %s", dataset, sorted(discovered))
            continue
        preprocessor = build_preprocessor(
            dataset,
            cls,
            settings,
            nsamps=args.nsamps,
            ovr_perc=args.ovr_perc,
            fs=args.fs,
            bin_spacing=args.bin_spacing,
            include_math_trials=args.include_math_trials,
        )
        if args.max_subjects is not None:
            subjects = list(preprocessor.subjects())[: args.max_subjects]
            preprocessor._subject_ids = tuple(subjects)  # noqa: SLF001
            logger.info("Limiting %s to first %d subjects", dataset, len(subjects))
        preprocessors[dataset] = preprocessor

    if not preprocessors:
        raise SystemExit("No datasets to process.")

    win_length = args.win_length or args.resolution
    results: list[dict[str, object]] = []
    for dataset, preprocessor in preprocessors.items():
        summary = process_dataset(
            dataset,
            preprocessor,
            resolution=args.resolution,
            hop_length=args.hop_length,
            win_length=win_length,
            max_windows_per_recording=args.max_windows_per_recording,
            lower_pct=args.lower_percentile,
            upper_pct=args.upper_percentile,
            output_dir=args.output_dir,
        )
        if summary:
            results.append(summary)

    if results:
        aggregate_path = args.output_dir / "stft_hist_summary_all.json"
        aggregate_path.write_text(json.dumps(results, indent=2))
        logger.info("Saved aggregate summary to %s", aggregate_path)
        print_metric_recommendations(results)
        analysis = {
            "metric_recommendations": metric_recommendations(results),
            "metric_correlations": analyze_metric_correlations(results),
            "clip_agreements": analyze_clip_agreements(results),
            "metric_tables": metric_tables(results),
        }
        analysis_path = args.output_dir / "stft_hist_analysis.json"
        analysis_path.write_text(json.dumps(analysis, indent=2))
        logger.info("Saved analysis to %s", analysis_path)


if __name__ == "__main__":
    main()
