"""Metadata normalization and caption generation utilities.

Provides functions for normalizing metadata fields (gender, health, age) and
generating descriptive captions from metadata dictionaries.
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

DEFAULT_HEALTH = "H"


def _is_missing(value: Any) -> bool:
    """Check if a value is missing or empty."""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def normalize_gender(value: Any) -> str | None:
    """Normalize gender values to 'M' or 'F'.

    Args:
        value: Raw gender value (can be str, int, bool, or None)

    Returns:
        'M' for male, 'F' for female, None if missing, or the original
        string if it doesn't match known patterns

    Examples:
        >>> normalize_gender("F")
        'F'
        >>> normalize_gender("female")
        'F'
        >>> normalize_gender("1")
        'F'
        >>> normalize_gender("M")
        'M'
        >>> normalize_gender("male")
        'M'
        >>> normalize_gender("0")
        'M'
        >>> normalize_gender(None)
        None
    """
    if _is_missing(value):
        return None
    text = str(value).strip().upper()
    if text in {"F", "FEMALE", "1", "TRUE"}:
        return "F"
    if text in {"M", "MALE", "0", "FALSE"}:
        return "M"
    return text


def normalize_health(value: Any) -> str:
    """Normalize health status to 'H' (healthy) or 'PD' (Parkinson's disease).

    Args:
        value: Raw health value (can be str, int, bool, or None)

    Returns:
        'H' for healthy (default), 'PD' for Parkinson's disease, or the
        original string if it doesn't match known patterns

    Examples:
        >>> normalize_health("PD")
        'PD'
        >>> normalize_health("parkinsons")
        'PD'
        >>> normalize_health("1")
        'PD'
        >>> normalize_health("H")
        'H'
        >>> normalize_health("healthy")
        'H'
        >>> normalize_health("0")
        'H'
        >>> normalize_health(None)
        'H'
    """
    if _is_missing(value):
        return DEFAULT_HEALTH
    text = str(value).strip().upper()
    if text in {"PD", "PARKINSONS", "1", "TRUE"}:
        return "PD"
    if text in {"H", "HEALTHY", "0", "FALSE"}:
        return "H"
    return text


def normalize_age(value: Any) -> int | None:
    """Convert age value to integer.

    Args:
        value: Raw age value (can be int, float, str, or None)

    Returns:
        Integer age or None if missing or invalid

    Examples:
        >>> normalize_age(25)
        25
        >>> normalize_age("30")
        30
        >>> normalize_age(45.7)
        45
        >>> normalize_age(None)
        None
        >>> normalize_age("")
        None
    """
    if _is_missing(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _gender_description(code: str | None) -> str | None:
    """Convert gender code to descriptive word."""
    if _is_missing(code):
        return None
    mapping = {"F": "female", "M": "male"}
    text = str(code).strip().upper()
    return mapping.get(text, text.lower())


def _health_description(code: str | None) -> tuple[str | None, str | None]:
    """Convert health code to (primary descriptor, clause).

    Returns a tuple of (primary, clause) where:
    - 'H' -> ('healthy', None)
    - 'PD' -> (None, 'with Parkinson\'s disease')
    """
    if _is_missing(code):
        return None, None
    text = str(code).strip().upper()
    if text == "PD":
        return None, "with Parkinson's disease"
    if text == "H":
        return "healthy", None
    return text.lower(), None


def build_caption(
    metadata: Mapping[str, Any],
    *,
    modality: str = "spectrogram image",
    subject_noun: str = "subject",
) -> str:
    """Build a natural language caption from metadata.

    Args:
        metadata: Dictionary containing 'age', 'gender', and 'health' keys
        modality: Type of data (e.g., "spectrogram image", "timeseries")
        subject_noun: Noun to describe the subject (e.g., "subject", "patient")

    Returns:
        Natural language caption describing the subject

    Examples:
        >>> build_caption({"age": 25, "gender": "F", "health": "H"})
        'a spectrogram image of a 25 year old healthy female subject'
        >>> build_caption({"age": 60, "gender": "M", "health": "PD"})
        'a spectrogram image of a 60 year old male subject with Parkinson\\'s disease'
        >>> build_caption({"gender": "F", "health": "H"})
        'a spectrogram image of a healthy female subject'
    """
    age = normalize_age(metadata.get("age"))
    gender_word = _gender_description(metadata.get("gender"))
    health_code = metadata.get("health", DEFAULT_HEALTH)
    health_primary, health_clause = _health_description(health_code)

    primary_bits: list[str] = []
    if age is not None:
        primary_bits.append(f"{age} year old")
    if health_primary:
        primary_bits.append(health_primary)
    if gender_word:
        primary_bits.append(gender_word)

    if primary_bits:
        caption = f"a {modality} of a {', '.join(primary_bits)} {subject_noun}"
    else:
        caption = f"a {modality} of a {subject_noun}"

    if health_clause:
        caption += f" {health_clause}"

    return caption
