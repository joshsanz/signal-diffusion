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
    detect_embedding_columns,
)


def _load_and_validate(parquet_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load parquet and return (synth_df, real_df)."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if "label" not in df.columns:
        raise ValueError("Parquet must have 'label' column ('F'=synthetic, 'T'=real)")
    if "subject" not in df.columns:
        raise ValueError("Parquet must have 'subject' column")

    synth_df = df[df["label"] == "F"].reset_index(drop=True)
    real_df = df[df["label"] == "T"].reset_index(drop=True)

    if len(synth_df) == 0:
        raise ValueError("No synthetic samples (label='F') in parquet")
    if len(real_df) == 0:
        raise ValueError("No real samples (label='T') in parquet")

    return synth_df, real_df


def _plot_strip(
    result: ComparisonResult,
    col_name: str,
    output_dir: Path,
    demographic_mismatch_only: bool = False,
) -> None:
    """Produce strip plot comparing synthetic vs real max similarities."""
    synth_max, real_max = result
    n_synth = len(synth_max)
    n_real = len(real_max)
    real_suffix = ", demo-mismatch" if demographic_mismatch_only else ""
    synth_label = f"Synthetic (n={n_synth})"
    real_label = f"Real (per-subject{real_suffix}) (n={n_real})"
    df = pd.DataFrame({
        "max_cosine_similarity": np.concatenate([synth_max, real_max]),
        "group": [synth_label] * n_synth + [real_label] * n_real,
    })

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.stripplot(
        data=df,
        x="group",
        y="max_cosine_similarity",
        order=[synth_label, real_label],
        jitter=0.2,
        alpha=0.7,
        ax=ax,
    )
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.7, label="0.95")
    ax.set_title(f"Max cosine similarity ({col_name})")
    ax.set_ylabel("Max cosine similarity")
    all_vals = np.concatenate([synth_max, real_max])
    y_lo = min(0.85, float(np.min(all_vals)) - 0.02)
    ax.set_ylim(max(0.0, y_lo), 1.0)
    ax.legend()
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
    args = parser.parse_args()

    synth_df, real_df = _load_and_validate(args.parquet)

    embedding_cols = args.embedding_cols
    if not embedding_cols:
        embedding_cols = detect_embedding_columns(pd.concat([synth_df, real_df], ignore_index=True))
        if not embedding_cols:
            raise ValueError(
                "No embedding columns found. Pass --embedding-cols explicitly."
            )
        print(f"Auto-detected embedding columns: {embedding_cols}")

    args.output.mkdir(parents=True, exist_ok=True)

    for col in tqdm(embedding_cols, desc="Embedding columns"):
        result = compute_comparisons(synth_df, real_df, col, args.demographic_mismatch_only)
        _plot_strip(result, col, args.output, args.demographic_mismatch_only)


if __name__ == "__main__":
    main()
