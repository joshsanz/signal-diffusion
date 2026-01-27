"""HPO result loading and parsing utilities."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class HPOStudyResult:
    """Parsed HPO result from JSON file."""

    spec_type: str
    task_type: str
    file_path: Path
    best_params: dict[str, Any]
    best_epoch: int
    best_metric: float
    best_user_attrs: dict[str, Any]
    timestamp: str


def extract_spec_type_and_task(filename: str) -> tuple[str, str, str] | None:
    """
    Extract spec_type, task_type, and timestamp from HPO result filename.

    Pattern: hpo_study_{spec_type}_{task_type}_{timestamp}.json

    Args:
        filename: HPO result filename

    Returns:
        Tuple of (spec_type, task_type, timestamp) or None if pattern doesn't match

    Examples:
        >>> extract_spec_type_and_task("hpo_study_db-iq_gender_20260127_143052.json")
        ('db-iq', 'gender', '20260127_143052')
        >>> extract_spec_type_and_task("hpo_study_timeseries_mixed_20260127.json")
        ('timeseries', 'mixed', '20260127')
    """
    pattern = re.compile(r"hpo_study_([^_]+)_(gender|mixed)_(\d{8}_\d{6}|\d{8})\.json")
    match = pattern.match(filename)
    if match:
        return match.groups()
    return None


def load_hpo_result(file_path: Path) -> dict[str, Any]:
    """
    Load HPO result from JSON file.

    Args:
        file_path: Path to hpo_study_results.json

    Returns:
        Parsed JSON data
    """
    with file_path.open("r") as f:
        return json.load(f)


def parse_hpo_result(
    file_path: Path,
    spec_type: str,
    task_type: str,
    timestamp: str,
) -> HPOStudyResult:
    """
    Parse HPO result JSON file into structured object.

    Args:
        file_path: Path to HPO result JSON file
        spec_type: Spectrogram type (e.g., 'db-iq', 'db-only')
        task_type: Task type (e.g., 'gender', 'mixed')
        timestamp: Timestamp from filename

    Returns:
        HPOStudyResult object
    """
    data = load_hpo_result(file_path)

    best_params = data.get("best_params", {})
    best_user_attrs = data.get("best_user_attrs", {})
    best_epoch = best_user_attrs.get("best_epoch", 1)
    best_metric = best_user_attrs.get("best_metric", 0.0)

    return HPOStudyResult(
        spec_type=spec_type,
        task_type=task_type,
        file_path=file_path,
        best_params=best_params,
        best_epoch=best_epoch,
        best_metric=best_metric,
        best_user_attrs=best_user_attrs,
        timestamp=timestamp,
    )


def find_hpo_results(
    hpo_dir: Path,
    spec_types: list[str],
    task_types: list[str],
) -> dict[tuple[str, str], HPOStudyResult]:
    """
    Find the most recent HPO result file for each (spec_type, task_type) combination.

    Pattern: hpo_study_{spec_type}_{task_type}_{timestamp}.json

    Args:
        hpo_dir: Directory containing HPO result JSON files
        spec_types: List of spec types to search for
        task_types: List of task types to search for

    Returns:
        Dictionary mapping (spec_type, task_type) → HPOStudyResult
    """
    # Track latest result for each combination
    results: dict[tuple[str, str], tuple[Path, str]] = {}

    for json_file in hpo_dir.glob("hpo_study_*.json"):
        extracted = extract_spec_type_and_task(json_file.name)
        if not extracted:
            continue

        spec_type, task_type, timestamp = extracted

        # Skip if not in requested types
        if spec_type not in spec_types or task_type not in task_types:
            continue

        key = (spec_type, task_type)

        # Keep the most recent result for each combination
        if key not in results or timestamp > results[key][1]:
            results[key] = (json_file, timestamp)

    # Parse JSON files into HPOStudyResult objects
    parsed_results = {}
    for (spec_type, task_type), (file_path, timestamp) in results.items():
        try:
            hpo_result = parse_hpo_result(file_path, spec_type, task_type, timestamp)
            parsed_results[(spec_type, task_type)] = hpo_result
        except Exception as e:
            # Log warning but continue processing other files
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to parse HPO result {file_path}: {e}. Skipping.")

    return parsed_results
