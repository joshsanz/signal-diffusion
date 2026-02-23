"""Shared extraction logic for memorization similarity analysis.

Used by plot_memorization_similarity.py and plot_memorization_survival.py.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd


META_COLS = {"age", "gender", "health", "subject", "label"}

# Preferred order for known embedding columns; others are appended.
PREFERRED_EMBEDDING_ORDER = ("cnn_embedding", "siglipv2_embedding", "sscd_embedding")


def detect_embedding_columns(df: pd.DataFrame) -> list[str]:
    """Return column names that contain list/array embeddings (exclude meta).

    Columns are returned in preferred order (cnn, siglipv2, sscd), with any
    additional embedding columns appended.
    """
    out = []
    for col in df.columns:
        if col in META_COLS:
            continue
        sample = df[col].iloc[0]
        if isinstance(sample, (list, np.ndarray)) and len(np.asarray(sample).shape) == 1:
            out.append(col)
    order_set = set(PREFERRED_EMBEDDING_ORDER)
    preferred = [c for c in PREFERRED_EMBEDDING_ORDER if c in out]
    others = [c for c in out if c not in order_set]
    return preferred + others


def embeddings_to_array(df: pd.DataFrame, col: str) -> np.ndarray:
    """Extract embedding column as (n, d) float32 array."""
    return np.stack([np.asarray(x, dtype=np.float32) for x in df[col].values], axis=0)


def _max_cosine_similarity(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """For each query row, return max cosine similarity to any reference row."""
    if query.size == 0 or reference.size == 0:
        return np.array([], dtype=np.float32)
    q = query.astype(np.float32)
    r = reference.astype(np.float32)
    q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    r_norm = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-8)
    sims = q_norm @ r_norm.T
    return np.max(sims, axis=1).astype(np.float32)


def synthetic_max_sims(synth_emb: np.ndarray, real_emb: np.ndarray) -> np.ndarray:
    """Max cosine sim of each synthetic sample to any real sample."""
    return _max_cosine_similarity(synth_emb, real_emb)


def per_subject_max_sims(
    real_df: pd.DataFrame,
    emb_array: np.ndarray,
    demographic_mismatch_only: bool = False,
) -> np.ndarray:
    """For each subject, max over their samples of (max sim to other subjects' samples).

    If demographic_mismatch_only, only compare to samples from other subjects whose
    demographics don't match (age diff > 15 years, different gender, or different health).
    Requires age, gender, health columns in real_df.

    Returns array of length n_subjects; NaNs for subjects with no valid comparison samples.
    """
    subjects = real_df["subject"].values
    ages = real_df["age"].values if "age" in real_df.columns else np.full(len(real_df), np.nan)
    genders = real_df["gender"].values if "gender" in real_df.columns else np.full(len(real_df), "")
    healths = real_df["health"].values if "health" in real_df.columns else np.full(len(real_df), "")

    unique_subjects = np.unique(subjects)
    max_sims = []

    for subj in unique_subjects:
        mask_self = subjects == subj
        mask_other_subj = ~mask_self
        self_emb = emb_array[mask_self]
        other_emb = emb_array[mask_other_subj]
        other_indices = np.where(mask_other_subj)[0]

        if other_emb.size == 0:
            max_sims.append(np.nan)
            continue

        if not demographic_mismatch_only:
            per_sample_max = _max_cosine_similarity(self_emb, other_emb)
            max_sims.append(float(np.max(per_sample_max)))
            continue

        self_indices = np.where(mask_self)[0]
        ages_self = pd.to_numeric(ages[self_indices], errors="coerce").astype(np.float64)
        ages_other = pd.to_numeric(ages[other_indices], errors="coerce").astype(np.float64)
        genders_self = np.asarray([str(x).strip() for x in genders[self_indices]])
        genders_other = np.asarray([str(x).strip() for x in genders[other_indices]])
        healths_self = np.asarray([str(x).strip() for x in healths[self_indices]])
        healths_other = np.asarray([str(x).strip() for x in healths[other_indices]])

        age_valid = ~(np.isnan(ages_self)[:, None] | np.isnan(ages_other)[None, :])
        age_diff = np.abs(ages_self[:, None] - ages_other[None, :])
        age_ok = np.where(age_valid, age_diff > 15, False)
        gender_ok = genders_self[:, None] != genders_other[None, :]
        health_ok = healths_self[:, None] != healths_other[None, :]
        valid = age_ok | gender_ok | health_ok

        q = emb_array[self_indices].astype(np.float32)
        r = emb_array[other_indices].astype(np.float32)
        q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
        r_norm = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-8)
        full_sims = (q_norm @ r_norm.T).astype(np.float32)
        full_sims = np.where(valid, full_sims, -np.inf)
        per_sample_max = np.max(full_sims, axis=1)
        per_sample_max = np.where(np.any(valid, axis=1), per_sample_max, np.nan)
        valid_max = per_sample_max[~np.isnan(per_sample_max)]
        max_sims.append(float(np.max(valid_max)) if len(valid_max) > 0 else np.nan)

    return np.array(max_sims, dtype=np.float32)


class ComparisonResult(NamedTuple):
    """Max similarity values for synthetic and real (per-subject)."""

    synth_max: np.ndarray
    real_max: np.ndarray


def compute_comparisons(
    synth_df: pd.DataFrame,
    real_df: pd.DataFrame,
    col: str,
    demographic_mismatch_only: bool = False,
) -> ComparisonResult:
    """Compute synthetic and per-subject max similarities for one embedding column."""
    synth_emb = embeddings_to_array(synth_df, col)
    real_emb = embeddings_to_array(real_df, col)

    synth_max = synthetic_max_sims(synth_emb, real_emb)
    real_max = per_subject_max_sims(real_df, real_emb, demographic_mismatch_only)
    real_max = real_max[~np.isnan(real_max)]

    return ComparisonResult(synth_max=synth_max, real_max=real_max)
