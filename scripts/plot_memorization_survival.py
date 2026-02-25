#!/usr/bin/env python3
"""Plot survival curves of max cosine similarity for memorization analysis.

Survival = P(max_similarity > x) for each threshold x.
Two lines: synthetic vs real (per-subject), giving fine-grained comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from memorization_utils import (
    ComparisonResult,
    compute_comparisons,
    compute_comparisons_real_only,
    detect_embedding_columns,
)


def _load_and_validate(parquet_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load parquet and return (synth_df, real_df).

    Allows real-only data: synth_df may be empty when real_df has rows (control mode).
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if "label" not in df.columns:
        raise ValueError("Parquet must have 'label' column ('F'=synthetic, 'T'=real)")
    if "subject" not in df.columns:
        raise ValueError("Parquet must have 'subject' column")

    synth_df = df[df["label"] == "F"].reset_index(drop=True)
    real_df = df[df["label"] == "T"].reset_index(drop=True)

    if len(real_df) == 0:
        raise ValueError("No real samples (label='T') in parquet")
    if len(synth_df) == 0 and len(real_df) > 0:
        pass  # Real-only mode (e.g. control dataset)

    return synth_df, real_df


def _survival_at_threshold(values: np.ndarray, threshold: float) -> float:
    """Fraction of values strictly greater than threshold."""
    if len(values) == 0:
        return 0.0
    return float(np.mean(values > threshold))


def _format_stats(values: np.ndarray) -> str:
    """Format mean and median for display."""
    if len(values) == 0:
        return "mean=--, median=--"
    return f"mean={np.mean(values):.3f}, median={np.median(values):.3f}"


def _plot_survival(
    result: ComparisonResult,
    col_name: str,
    output_dir: Path,
    demographic_mismatch_only: bool = False,
    real_only: bool = False,
) -> None:
    """Produce survival plot: P(max_sim > x) vs x for synthetic and real."""
    synth_real_max, real_real_max, synth_synth_max = result

    thresholds = np.linspace(0, 1, 201)

    surv_synth = np.array([_survival_at_threshold(synth_real_max, t) for t in thresholds])
    surv_real = np.array([_survival_at_threshold(real_real_max, t) for t in thresholds])
    surv_synth_synth = np.array(
        [_survival_at_threshold(synth_synth_max, t) for t in thresholds]
    )

    n_synth = len(synth_real_max)
    n_real = len(real_real_max)
    n_synth_synth = len(synth_synth_max)
    if demographic_mismatch_only:
        real_label = f"Real (per-subject, demo-mismatch) (n={n_real})"
    else:
        real_label = f"Real (per-subject) (n={n_real})"
    fig, ax = plt.subplots(figsize=(6, 4))
    if not real_only:
        ax.plot(
            thresholds,
            surv_synth,
            label=f"Synthetic (synth->real, n={n_synth})",
            color="C0",
        )
    ax.plot(thresholds, surv_real, label=real_label, color="C1")
    if n_synth_synth > 0:
        ax.plot(
            thresholds,
            surv_synth_synth,
            label=f"Synthetic (synth->synth, n={n_synth_synth})",
            color="C2",
        )
    ax.axvline(x=0.95, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Max cosine similarity threshold")
    ax.set_ylabel("Survival P(max_sim > threshold)")
    ax.set_title(f"Memorization survival ({col_name})")
    ax.legend()
    all_vals_parts = [v for v in [synth_real_max, real_real_max, synth_synth_max] if len(v) > 0]
    all_vals = np.concatenate(all_vals_parts) if all_vals_parts else np.array([0.0])
    x_lo = min(0.85, float(np.min(all_vals)) - 0.02)
    ax.set_xlim(max(0.0, x_lo), 1.0)
    ax.set_ylim(0, 1.02)

    stats_lines = []
    if len(synth_real_max) > 0:
        stats_lines.append(f"Synth->Real: {_format_stats(synth_real_max)}")
    if len(real_real_max) > 0:
        if real_only:
            real_name = "Control"
        elif demographic_mismatch_only:
            real_name = "Real (demo-mismatch)"
        else:
            real_name = "Real->Real"
        stats_lines.append(f"{real_name}: {_format_stats(real_real_max)}")
    if len(synth_synth_max) > 0:
        stats_lines.append(f"Synth->Synth: {_format_stats(synth_synth_max)}")
    if stats_lines:
        stats_text = "\n".join(stats_lines)
        fig.tight_layout(rect=[0, 0.065, 1, 1])
        fig.text(
            0.5, 0.062, stats_text,
            transform=fig.transFigure,
            fontsize=8,
            ha="center",
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )
    else:
        fig.tight_layout()
    suffix = "_demo_mismatch" if demographic_mismatch_only else ""
    out_path = output_dir / f"memorization_survival_{col_name}{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot survival curves of max cosine similarity for memorization analysis."
    )
    parser.add_argument("parquet", type=Path, help="Path to embeddings parquet file")
    parser.add_argument(
        "--embedding-cols",
        type=str,
        nargs="*",
        default=None,
        help="Embedding column names; if omitted, auto-detect from schema",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for plots",
    )
    parser.add_argument(
        "--demographic-mismatch-only",
        action="store_true",
        help="For real data, only compare to samples with mismatched demographics (age diff > 15y, different gender, or different health)",
    )
    parser.add_argument(
        "--max-synth-samples",
        type=int,
        default=None,
        help="Subsample synthetic data to at most N samples for faster plotting (e.g. 1000)",
    )
    args = parser.parse_args()

    SAMPLE_SEED = 42

    synth_df, real_df = _load_and_validate(args.parquet)

    if args.max_synth_samples is not None and len(synth_df) > args.max_synth_samples:
        synth_df = synth_df.sample(n=args.max_synth_samples, random_state=SAMPLE_SEED).reset_index(drop=True)
        print(f"Subsampled synthetic to {len(synth_df)} samples")

    embedding_cols = args.embedding_cols
    if not embedding_cols:
        embedding_cols = detect_embedding_columns(pd.concat([synth_df, real_df], ignore_index=True))
        if not embedding_cols:
            raise ValueError(
                "No embedding columns found. Pass --embedding-cols explicitly."
            )
        print(f"Auto-detected embedding columns: {embedding_cols}")

    args.output.mkdir(parents=True, exist_ok=True)

    real_only = len(synth_df) == 0
    if real_only:
        print("Real-only mode (no synthetic samples)")

    for col in tqdm(embedding_cols, desc="Embedding columns"):
        if real_only:
            result = compute_comparisons_real_only(real_df, col)
        else:
            result = compute_comparisons(synth_df, real_df, col, args.demographic_mismatch_only)
        _plot_survival(
            result,
            col,
            args.output,
            args.demographic_mismatch_only,
            real_only=real_only,
        )


if __name__ == "__main__":
    main()
