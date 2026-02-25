#!/usr/bin/env python3
"""Plot max cosine similarity of embeddings for memorization analysis.

Compares synthetic samples vs real training samples to detect memorization.
Produces strip plots (one per embedding column) saved to the output directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import seaborn as sns
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


def _format_stats(values: np.ndarray) -> str:
    """Format mean and median for display."""
    if len(values) == 0:
        return "mean=--, median=--"
    return f"mean={np.mean(values):.3f}, median={np.median(values):.3f}"


def _plot_strip(
    result: ComparisonResult,
    col_name: str,
    output_dir: Path,
    demographic_mismatch_only: bool = False,
    real_only: bool = False,
) -> None:
    """Produce strip plot comparing synthetic vs real max similarities."""
    synth_real_max, real_real_max, synth_synth_max = result
    n_synth = len(synth_real_max)
    n_real = len(real_real_max)
    n_synth_synth = len(synth_synth_max)
    real_suffix = ", demo-mismatch" if demographic_mismatch_only else ""
    synth_label = f"Synth->Real (n={n_synth})"
    real_label = f"Real->Real{real_suffix} (n={n_real})"
    synth_synth_label = f"Synth->Synth (n={n_synth_synth})"

    if real_only:
        plot_df = pd.DataFrame({
            "max_cosine_similarity": real_real_max,
            "group": [real_label] * n_real,
        })
        order = [real_label]
    else:
        concat_vals = [synth_real_max, real_real_max]
        concat_groups = [synth_label] * n_synth + [real_label] * n_real
        if n_synth_synth > 0:
            concat_vals.append(synth_synth_max)
            concat_groups.extend([synth_synth_label] * n_synth_synth)
        plot_df = pd.DataFrame({
            "max_cosine_similarity": np.concatenate(concat_vals),
            "group": concat_groups,
        })
        order = [synth_label, real_label]
        if n_synth_synth > 0:
            order.append(synth_synth_label)

    width = max(6, 2.5 * len(order))
    fig, ax = plt.subplots(figsize=(width, 5))
    sns.stripplot(
        data=plot_df,
        x="group",
        y="max_cosine_similarity",
        order=order,
        jitter=0.2,
        alpha=0.7,
        ax=ax,
    )
    ax.tick_params(axis="x", rotation=15)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.7, label="0.95")
    ax.set_title(f"Max cosine similarity ({col_name})")
    ax.set_ylabel("Max cosine similarity")
    all_vals_parts = [v for v in [synth_real_max, real_real_max, synth_synth_max] if len(v) > 0]
    all_vals = np.concatenate(all_vals_parts) if all_vals_parts else np.array([0.0])
    y_lo = min(0.85, float(np.min(all_vals)) - 0.02)
    ax.set_ylim(max(0.0, y_lo), 1.0)
    ax.legend()

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
    out_path = output_dir / f"memorization_maxsim_{col_name}{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot max cosine similarity for memorization analysis."
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
        _plot_strip(
            result,
            col,
            args.output,
            args.demographic_mismatch_only,
            real_only=real_only,
        )


if __name__ == "__main__":
    main()
