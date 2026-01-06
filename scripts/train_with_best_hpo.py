"""
Train classifier models using best hyperparameters from HPO results.

This script:
1. Loads best hyperparameters from HPO result JSON files
2. Merges them with base config files for each spec type
3. Trains models for selected spec_type × task_type combinations
4. Saves trained models and optimized configs in the output directory

Usage:
    uv run python scripts/train_with_best_hpo.py runs/optimized
    uv run python scripts/train_with_best_hpo.py runs/optimized \\
        --spec-types db-iq db-polar --task-types gender
    uv run python scripts/train_with_best_hpo.py runs/optimized \\
        --epochs 30 --early-stopping
"""

from __future__ import annotations

import json
import re
import copy
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tomli_w
import typer

from signal_diffusion.training.classification import (
    load_experiment_config,
    train_from_config,
    TrainingSummary,
)
from signal_diffusion.classification.tasks import build_task_specs
from signal_diffusion.log_setup import get_logger

LOGGER = get_logger(__name__)

# Constants
SPEC_TYPES = ["db-only", "db-polar", "db-iq", "timeseries"]
TASK_TYPES = ["gender", "mixed"]

# Spec type → base config path
SPEC_TO_CONFIG = {
    "db-only": "config/classification/baseline.toml",
    "db-polar": "config/classification/baseline-db-polar.toml",
    "db-iq": "config/classification/baseline-db-iq.toml",
    "timeseries": "config/classification/baseline-timeseries.toml",
}

# Task type → tasks list
TASK_TYPE_TO_TASKS = {
    "gender": ["gender"],
    "mixed": ["gender", "health", "age"],
}


@dataclass(slots=True)
class HPOResult:
    """Parsed HPO result from JSON file."""
    spec_type: str
    task_type: str
    file_path: Path
    best_params: dict
    best_epoch: int
    best_metric: float
    timestamp: str


@dataclass(slots=True)
class TrainingJob:
    """Configuration for a single training run."""
    spec_type: str
    task_type: str
    base_config_path: Path
    output_dir: Path
    hpo_result: HPOResult
    user_overrides: dict


def find_hpo_results(
    hpo_dir: Path,
    spec_types: List[str],
    task_types: List[str],
) -> Dict[Tuple[str, str], HPOResult]:
    """
    Find the most recent HPO result file for each (spec_type, task_type) combination.

    Pattern: hpo_study_{spec_type}_{task_type}_{timestamp}.json

    Args:
        hpo_dir: Directory containing HPO result JSON files
        spec_types: List of spec types to search for
        task_types: List of task types to search for

    Returns:
        Dictionary mapping (spec_type, task_type) → HPOResult
    """
    pattern = re.compile(r"hpo_study_([^_]+)_(gender|mixed)_(\d{8}_\d{6}|\d{8})\.json")

    # Track latest result for each combination
    results: Dict[Tuple[str, str], Tuple[Path, str]] = {}

    for json_file in hpo_dir.glob("hpo_study_*.json"):
        match = pattern.match(json_file.name)
        if not match:
            continue

        spec_type, task_type, timestamp = match.groups()

        # Skip if not in requested types
        if spec_type not in spec_types or task_type not in task_types:
            continue

        key = (spec_type, task_type)

        # Keep the most recent result for each combination
        if key not in results or timestamp > results[key][1]:
            results[key] = (json_file, timestamp)

    # Parse JSON files into HPOResult objects
    parsed_results = {}
    for (spec_type, task_type), (file_path, timestamp) in results.items():
        try:
            hpo_result = parse_hpo_result(file_path, spec_type, task_type, timestamp)
            parsed_results[(spec_type, task_type)] = hpo_result
        except Exception as e:
            LOGGER.warning(f"Failed to parse HPO result {file_path}: {e}. Skipping.")

    return parsed_results


def parse_hpo_result(
    file_path: Path,
    spec_type: str,
    task_type: str,
    timestamp: str,
) -> HPOResult:
    """Parse HPO result JSON file."""
    with file_path.open("r") as f:
        data = json.load(f)

    best_params = data.get("best_params", {})
    best_user_attrs = data.get("best_user_attrs", {})
    best_epoch = best_user_attrs.get("best_epoch", 1)
    best_metric = best_user_attrs.get("best_metric", 0.0)

    return HPOResult(
        spec_type=spec_type,
        task_type=task_type,
        file_path=file_path,
        best_params=best_params,
        best_epoch=best_epoch,
        best_metric=best_metric,
        timestamp=timestamp,
    )


def merge_hpo_params_to_config(
    base_config_path: Path,
    hpo_params: dict,
    task_type: str,
    user_overrides: dict,
) -> dict:
    """
    Load base config TOML and merge with HPO best_params and user overrides.

    HPO params mapping:
    - learning_rate → [optimizer] learning_rate
    - weight_decay → [optimizer] weight_decay
    - scheduler → [scheduler] name
    - dropout → [model] dropout
    - depth → [model] depth
    - layer_repeats → [model] layer_repeats
    - embedding_dim → [model] embedding_dim
    - batch_size → [dataset] batch_size

    Args:
        base_config_path: Path to base TOML config
        hpo_params: Best HPO parameters from results
        task_type: Task type (gender or mixed)
        user_overrides: User-specified overrides

    Returns:
        Merged config dictionary
    """
    # Load base config
    with base_config_path.open("rb") as f:
        config = tomllib.load(f)

    # Deep copy to avoid mutations
    config = copy.deepcopy(config)

    # Initialize sections if missing
    if "optimizer" not in config:
        config["optimizer"] = {}
    if "scheduler" not in config:
        config["scheduler"] = {}
    if "model" not in config:
        config["model"] = {}
    if "dataset" not in config:
        config["dataset"] = {}
    if "training" not in config:
        config["training"] = {}

    # Apply HPO parameters
    if "learning_rate" in hpo_params:
        config["optimizer"]["learning_rate"] = hpo_params["learning_rate"]
    if "weight_decay" in hpo_params:
        config["optimizer"]["weight_decay"] = hpo_params["weight_decay"]
    if "scheduler" in hpo_params:
        config["scheduler"]["name"] = hpo_params["scheduler"]
    if "dropout" in hpo_params:
        config["model"]["dropout"] = hpo_params["dropout"]
    if "depth" in hpo_params:
        config["model"]["depth"] = hpo_params["depth"]
    if "layer_repeats" in hpo_params:
        config["model"]["layer_repeats"] = hpo_params["layer_repeats"]
    if "embedding_dim" in hpo_params:
        config["model"]["embedding_dim"] = hpo_params["embedding_dim"]
    if "batch_size" in hpo_params:
        config["dataset"]["batch_size"] = hpo_params["batch_size"]

    # Set tasks based on task_type
    config["dataset"]["tasks"] = TASK_TYPE_TO_TASKS[task_type]

    # Apply user overrides
    if "epochs" in user_overrides and user_overrides["epochs"] is not None:
        config["training"]["epochs"] = user_overrides["epochs"]

    if "early_stopping" in user_overrides:
        config["training"]["early_stopping"] = user_overrides["early_stopping"]
        if user_overrides["early_stopping"]:
            config["training"]["early_stopping_patience"] = user_overrides.get(
                "early_stopping_patience", 5
            )

    if "swa_enabled" in user_overrides:
        config["training"]["swa_enabled"] = user_overrides["swa_enabled"]
    if "swa_extra_ratio" in user_overrides and user_overrides["swa_extra_ratio"] is not None:
        config["training"]["swa_extra_ratio"] = user_overrides["swa_extra_ratio"]

    return config


def execute_training_job(job: TrainingJob, cwd: Path) -> Tuple[bool, Optional[TrainingSummary], Optional[str]]:
    """
    Execute a single training job.

    Args:
        job: Training job specification
        cwd: Current working directory (for resolving relative paths)

    Returns:
        Tuple of (success: bool, checkpoint_path: Optional[Path], error_msg: Optional[str])
    """
    LOGGER.info(f"\n{'='*80}")
    LOGGER.info(f"Training: {job.spec_type} × {job.task_type}")
    LOGGER.info(f"Output: {job.output_dir}")
    LOGGER.info(f"HPO params: {job.hpo_result.best_params}")
    LOGGER.info(f"{'='*80}\n")

    try:
        # Create output directory
        job.output_dir.mkdir(parents=True, exist_ok=True)

        # Merge config
        merged_config_dict = merge_hpo_params_to_config(
            base_config_path=job.base_config_path,
            hpo_params=job.hpo_result.best_params,
            task_type=job.task_type,
            user_overrides=job.user_overrides,
        )

        # Save merged config as TOML
        config_toml_path = job.output_dir / "config_optimized.toml"
        with config_toml_path.open("wb") as f:
            tomli_w.dump(merged_config_dict, f)

        LOGGER.info(f"Saved optimized config to: {config_toml_path}")

        # Load experiment config
        experiment = load_experiment_config(config_toml_path)

        # Override output_dir
        experiment.training.output_dir = job.output_dir

        # Train
        summary = train_from_config(experiment)

        LOGGER.info(f"Training complete: {summary.best_checkpoint}")

        return True, summary, None

    except Exception as e:
        error_msg = f"Training failed: {e}"
        LOGGER.error(error_msg, exc_info=True)
        return False, None, error_msg


def execute_training_job_with_swa_comparison(
    job: TrainingJob,
    cwd: Path,
) -> Tuple[
    Tuple[bool, Optional[TrainingSummary], Optional[str]],
    Tuple[bool, Optional[TrainingSummary], Optional[str]],
]:
    """Train without SWA, then with SWA, for side-by-side comparison."""
    base_job = copy.deepcopy(job)
    base_job.output_dir = job.output_dir / "no_swa"
    base_job.user_overrides = {**job.user_overrides, "swa_enabled": False}
    base_result = execute_training_job(base_job, cwd)

    swa_job = copy.deepcopy(job)
    swa_job.output_dir = job.output_dir / "with_swa"
    swa_job.user_overrides = {**job.user_overrides, "swa_enabled": True}
    swa_result = execute_training_job(swa_job, cwd)

    return base_result, swa_result


def _load_classification_targets(run_dir: Path) -> list[str]:
    config_path = run_dir / "config_resolved.json"
    if not config_path.exists():
        return []
    with config_path.open("r") as handle:
        config = json.load(handle)
    dataset = config.get("dataset", {})
    dataset_name = dataset.get("name")
    task_names = dataset.get("tasks", [])
    if not dataset_name or not task_names:
        return []
    task_specs = build_task_specs(dataset_name, task_names)
    return [spec.name for spec in task_specs if spec.task_type == "classification"]


def _load_best_val_accuracies(run_dir: Path) -> dict[str, float]:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        return {}
    with history_path.open("r") as handle:
        history = json.load(handle)

    classification_targets = set(_load_classification_targets(run_dir))
    if not classification_targets:
        LOGGER.warning(
            "Could not resolve classification targets for %s; using all tasks in val_accuracy.",
            run_dir,
        )
    best: dict[str, float] = {}
    for epoch_metrics in history:
        val_accuracy = epoch_metrics.get("val_accuracy", {})
        for task_name, accuracy in val_accuracy.items():
            if classification_targets and task_name not in classification_targets:
                continue
            if accuracy is None:
                continue
            value = float(accuracy)
            best[task_name] = max(best.get(task_name, float("-inf")), value)
    return best


def create_training_jobs(
    hpo_results: Dict[Tuple[str, str], HPOResult],
    output_dir: Path,
    user_overrides: dict,
    cwd: Path,
) -> List[TrainingJob]:
    """Create training jobs for each available HPO result."""
    jobs = []

    for (spec_type, task_type), hpo_result in hpo_results.items():
        # Get base config path (relative to cwd)
        base_config_rel = SPEC_TO_CONFIG[spec_type]
        base_config_path = (cwd / base_config_rel).resolve()

        if not base_config_path.exists():
            LOGGER.warning(f"Base config not found: {base_config_path}. Skipping {spec_type}_{task_type}")
            continue

        # Create output directory for this job
        job_output_dir = output_dir / f"{spec_type}_{task_type}_optimized"

        job = TrainingJob(
            spec_type=spec_type,
            task_type=task_type,
            base_config_path=base_config_path,
            output_dir=job_output_dir,
            hpo_result=hpo_result,
            user_overrides=user_overrides,
        )

        jobs.append(job)

    return jobs


def print_summary(results: List[dict], compare_swa: bool = False) -> None:
    """Print summary table of training results."""
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)

    if compare_swa:
        print(f"{'Config':<35} {'Base Status':<12} {'SWA Status':<12}")
        print("-"*80)
        for result in results:
            config_name = f"{result['spec_type']}_{result['task_type']}_optimized"
            base_status = "✓ SUCCESS" if result["base"]["success"] else "✗ FAILED"
            swa_status = "✓ SUCCESS" if result["swa"]["success"] else "✗ FAILED"
            print(f"{config_name:<35} {base_status:<12} {swa_status:<12}")
    else:
        print(f"{'Config':<35} {'Spec Type':<15} {'Task Type':<10} {'Status':<10}")
        print("-"*80)
        for result in results:
            config_name = f"{result['spec_type']}_{result['task_type']}_optimized"
            status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
            print(
                f"{config_name:<35} "
                f"{result['spec_type']:<15} "
                f"{result['task_type']:<10} "
                f"{status:<10}"
            )

    print("="*80)


def print_swa_comparison_summary(comparisons: List[dict]) -> None:
    """Print per-target accuracy comparison for SWA vs non-SWA runs."""
    print("\n" + "="*90)
    print("SWA VS NON-SWA ACCURACY COMPARISON")
    print("="*90)
    print(f"{'Config':<30} {'Target':<20} {'Base Acc':<12} {'SWA Acc':<12} {'Delta':<10}")
    print("-"*90)

    for comp in comparisons:
        config_name = f"{comp['spec_type']}_{comp['task_type']}"
        base_metrics = comp.get("base_metrics", {})
        swa_metrics = comp.get("swa_metrics", {})
        targets = sorted(set(base_metrics) | set(swa_metrics))
        if not targets:
            print(f"{config_name:<30} {'n/a':<20} {'FAILED':<12} {'FAILED':<12} {'N/A':<10}")
            continue
        for target in targets:
            base_value = base_metrics.get(target)
            swa_value = swa_metrics.get(target)
            base_display = f"{base_value:.4f}" if base_value is not None else "FAILED"
            swa_display = f"{swa_value:.4f}" if swa_value is not None else "FAILED"
            if base_value is not None and swa_value is not None:
                delta_display = f"{(swa_value - base_value):+.4f}"
            else:
                delta_display = "N/A"
            print(
                f"{config_name:<30} {target:<20} {base_display:<12} {swa_display:<12} {delta_display:<10}"
            )

    print("="*90)


app = typer.Typer(help="Train classifiers using HPO-optimized hyperparameters")


@app.command()
def main(
    output_dir: Path = typer.Argument(..., help="Base directory to save trained models"),
    hpo_results_dir: Path = typer.Option(
        "hpo_results",
        "--hpo-results-dir",
        help="Directory containing HPO result JSON files",
    ),
    spec_types: Optional[List[str]] = typer.Option(
        None,
        "--spec-types",
        help="Spec types to train (default: all). Options: db-only, db-polar, db-iq, timeseries",
    ),
    task_types: Optional[List[str]] = typer.Option(
        None,
        "--task-types",
        help="Task types to train (default: all). Options: gender, mixed",
    ),
    epochs: Optional[int] = typer.Option(None, "--epochs", help="Override epochs"),
    early_stopping: bool = typer.Option(False, "--early-stopping", help="Enable early stopping"),
    early_stopping_patience: int = typer.Option(5, "--early-stopping-patience", help="Early stopping patience"),
    compare_swa: bool = typer.Option(
        True,
        "--compare-swa/--no-compare-swa",
        help="Train both with and without SWA for comparison",
    ),
    swa_extra_ratio: float = typer.Option(
        0.333,
        "--swa-extra-ratio",
        help="Ratio of SWA epochs to base epochs when SWA is enabled",
    ),
) -> None:
    """
    Train classifier models using optimized hyperparameters from HPO results.

    Examples:
        # Train all combinations
        uv run python scripts/train_with_best_hpo.py runs/optimized

        # Train only db-iq with gender task
        uv run python scripts/train_with_best_hpo.py runs/optimized \\
            --spec-types db-iq --task-types gender

        # Train with custom epochs and early stopping
        uv run python scripts/train_with_best_hpo.py runs/optimized \\
            --epochs 30 --early-stopping --early-stopping-patience 7

        # Disable SWA comparison runs
        uv run python scripts/train_with_best_hpo.py runs/optimized \\
            --no-compare-swa

        # Train all db-polar combinations
        uv run python scripts/train_with_best_hpo.py runs/optimized \\
            --spec-types db-polar
    """

    # Get working directory
    cwd = Path.cwd()

    # Validate and expand paths
    output_dir = output_dir.expanduser().resolve()
    hpo_results_dir = hpo_results_dir.expanduser().resolve()

    # Default to all types if not specified
    if spec_types is None or "all" in spec_types:
        spec_types = SPEC_TYPES
    if task_types is None or "all" in task_types:
        task_types = TASK_TYPES

    # Validate inputs
    invalid_specs = set(spec_types) - set(SPEC_TYPES)
    if invalid_specs:
        LOGGER.error(f"Invalid spec types: {invalid_specs}. Valid: {SPEC_TYPES}")
        raise typer.Exit(1)

    invalid_tasks = set(task_types) - set(TASK_TYPES)
    if invalid_tasks:
        LOGGER.error(f"Invalid task types: {invalid_tasks}. Valid: {TASK_TYPES}")
        raise typer.Exit(1)

    # Check HPO results directory
    if not hpo_results_dir.exists():
        LOGGER.error(f"HPO results directory not found: {hpo_results_dir}")
        raise typer.Exit(1)

    # Find HPO results
    try:
        hpo_results = find_hpo_results(hpo_results_dir, spec_types, task_types)
    except FileNotFoundError as e:
        LOGGER.error(str(e))
        raise typer.Exit(1)

    if not hpo_results:
        LOGGER.error("No HPO results found for requested combinations")
        raise typer.Exit(1)

    LOGGER.info(f"Found {len(hpo_results)} HPO results")

    # Build user overrides
    user_overrides = {
        "early_stopping": early_stopping,
        "early_stopping_patience": early_stopping_patience,
        "epochs": epochs,
        "swa_extra_ratio": swa_extra_ratio,
    }

    # Create and execute training jobs
    jobs = create_training_jobs(hpo_results, output_dir, user_overrides, cwd)

    if not jobs:
        LOGGER.error("No training jobs created")
        raise typer.Exit(1)

    LOGGER.info(f"Created {len(jobs)} training job(s)")

    results = []
    comparisons = []
    for i, job in enumerate(jobs, 1):
        LOGGER.info(f"\n[{i}/{len(jobs)}] Starting training job: {job.spec_type} × {job.task_type}")
        if compare_swa:
            base_result, swa_result = execute_training_job_with_swa_comparison(job, cwd)
            base_success, base_summary, base_error = base_result
            swa_success, swa_summary, swa_error = swa_result
            base_run_dir = base_summary.run_dir if base_summary is not None else None
            swa_run_dir = swa_summary.run_dir if swa_summary is not None else None
            base_metrics = _load_best_val_accuracies(base_run_dir) if base_run_dir else {}
            swa_metrics = _load_best_val_accuracies(swa_run_dir) if swa_run_dir else {}
            improvements = {
                key: swa_metrics.get(key, 0.0) - base_metrics.get(key, 0.0)
                for key in set(base_metrics) | set(swa_metrics)
                if key in base_metrics and key in swa_metrics
            }
            results.append({
                "spec_type": job.spec_type,
                "task_type": job.task_type,
                "base": {
                    "success": base_success,
                    "run_dir": str(base_run_dir) if base_run_dir else None,
                    "best_checkpoint": str(base_summary.best_checkpoint) if base_summary else None,
                    "error": base_error,
                },
                "swa": {
                    "success": swa_success,
                    "run_dir": str(swa_run_dir) if swa_run_dir else None,
                    "best_checkpoint": str(swa_summary.best_checkpoint) if swa_summary else None,
                    "error": swa_error,
                },
                "base_metrics": base_metrics,
                "swa_metrics": swa_metrics,
                "improvements": improvements,
            })
            comparisons.append({
                "spec_type": job.spec_type,
                "task_type": job.task_type,
                "base_metrics": base_metrics,
                "swa_metrics": swa_metrics,
                "improvements": improvements,
            })
        else:
            success, summary, error_msg = execute_training_job(job, cwd)
            results.append({
                "spec_type": job.spec_type,
                "task_type": job.task_type,
                "success": success,
                "run_dir": str(summary.run_dir) if summary else None,
                "checkpoint_path": str(summary.best_checkpoint) if summary else None,
                "error": error_msg,
            })

    # Generate summary
    print_summary(results, compare_swa=compare_swa)
    if compare_swa:
        print_swa_comparison_summary(comparisons)

    # Save summary JSON
    summary_path = output_dir / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        payload = {"results": results, "comparisons": comparisons} if compare_swa else {"results": results}
        json.dump(payload, f, indent=2)

    LOGGER.info(f"\nTraining summary saved to: {summary_path}")

    # Exit with error code if any jobs failed
    if compare_swa:
        failures = sum(
            1
            for r in results
            if not r["base"]["success"] or not r["swa"]["success"]
        )
    else:
        failures = sum(1 for r in results if not r["success"])
    if failures > 0:
        LOGGER.error(f"\n{failures} training job(s) failed")
        raise typer.Exit(1)
    else:
        LOGGER.info(f"\nAll {len(results)} training jobs completed successfully!")


if __name__ == "__main__":
    app()
