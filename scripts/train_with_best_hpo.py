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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tomli_w
import typer

from signal_diffusion.training.classification import (
    load_experiment_config,
    train_from_config,
    TrainingSummary,
)
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

    # Set tasks: use all available tasks for the dataset
    from signal_diffusion.classification.tasks import available_tasks
    dataset_name = config["dataset"].get("name")
    if dataset_name:
        all_tasks = available_tasks(dataset_name)
        config["dataset"]["tasks"] = list(all_tasks)
        LOGGER.info(f"Configured all available tasks for dataset: {all_tasks}")
    else:
        # Fallback to TASK_TYPE_TO_TASKS if dataset name not found
        config["dataset"]["tasks"] = TASK_TYPE_TO_TASKS[task_type]
        LOGGER.warning(f"Dataset name not found in config, using task_type mapping: {TASK_TYPE_TO_TASKS[task_type]}")

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

    # Set a fixed seed for reproducibility if not already set
    if "seed" not in config["training"] or config["training"]["seed"] is None:
        config["training"]["seed"] = 42

    return config


def evaluate_checkpoint(
    checkpoint_path: Path,
    run_dir: Path,
    cwd: Path,
) -> dict[str, dict[str, float]]:
    """
    Load a checkpoint and evaluate it on validation and test sets.

    Args:
        checkpoint_path: Path to the .pt checkpoint file
        run_dir: Directory containing config_resolved.json
        cwd: Current working directory

    Returns:
        Dictionary with structure:
        {
            "val": {"task_name": accuracy, ...},
            "test": {"task_name": accuracy, ...}
        }
    """
    # Load config from run directory
    config_path = run_dir / "config_resolved.json"
    if not config_path.exists():
        LOGGER.warning(f"Config not found at {config_path}, skipping evaluation")
        return {"val": {}, "test": {}}

    with config_path.open("r") as f:
        config_data = json.load(f)

    # Load settings
    from signal_diffusion.config import load_settings
    from signal_diffusion.classification import build_classifier, build_dataset, build_task_specs, ClassifierConfig

    settings_path = config_data.get("settings_path")
    if not settings_path:
        LOGGER.warning("settings_path not found in config, skipping evaluation")
        return {"val": {}, "test": {}}

    settings = load_settings(settings_path)

    # Apply data overrides if present
    if "output_type" in config_data.get("dataset", {}).get("extras", {}):
        settings.output_type = config_data["dataset"]["extras"]["output_type"]

    # Extract configuration
    dataset_cfg = config_data["dataset"]
    model_cfg = config_data["model"]

    dataset_name = dataset_cfg["name"]
    tasks = dataset_cfg["tasks"]
    task_specs = build_task_specs(dataset_name, tasks)

    # Build classifier
    classifier_config = ClassifierConfig(
        backbone=model_cfg["backbone"],
        input_channels=model_cfg["input_channels"],
        tasks=task_specs,
        embedding_dim=model_cfg.get("embedding_dim", 256),
        dropout=model_cfg.get("dropout", 0.3),
        activation=model_cfg.get("activation", "gelu"),
        depth=model_cfg.get("depth", 3),
        layer_repeats=model_cfg.get("layer_repeats", 2),
        extras=model_cfg.get("extras", {}),
    )

    import torch
    from torch.utils.data import DataLoader

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Build model and load checkpoint
    model = build_classifier(classifier_config)

    # Load checkpoint state
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    except Exception as e:
        LOGGER.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
        return {"val": {}, "test": {}}

    # Build datasets
    batch_size = dataset_cfg.get("batch_size", 32)
    num_workers = dataset_cfg.get("num_workers", 4)

    results = {}

    for split in ["val", "test"]:
        try:
            # Build dataset for this split
            dataset = build_dataset(
                settings,
                dataset_name=dataset_name,
                split=split,
                tasks=tasks,
                target_format="dict",
            )

            # Create dataloader
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )

            # Run evaluation
            task_lookup = {spec.name: spec for spec in task_specs}

            # Import _run_epoch from classification module
            from signal_diffusion.training.classification import _run_epoch
            from torch import nn

            # Create criteria for each task
            criteria = {}
            for name in tasks:
                spec = task_lookup[name]
                if spec.task_type == "classification":
                    criteria[name] = nn.CrossEntropyLoss()
                else:
                    criteria[name] = nn.HuberLoss(delta=2.0)

            # Create task weights (uniform)
            task_weights = {name: 1.0 for name in tasks}

            # Run evaluation epoch
            metrics, _ = _run_epoch(
                model,
                data_loader=loader,
                criteria=criteria,
                task_weights=task_weights,
                task_specs=task_lookup,
                device=device,
                optimizer=None,
                scaler=None,
                clip_grad=None,
                log_every=0,
                train=False,
            )

            # Extract accuracy metrics (which includes MAE for regression tasks)
            split_results = {}
            for task_name in tasks:
                accuracy = metrics["accuracy"].get(task_name)
                if accuracy is not None:
                    split_results[task_name] = float(accuracy)

            results[split] = split_results

        except Exception as e:
            LOGGER.warning(f"Failed to evaluate on {split} split: {e}")
            results[split] = {}

    return results


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
    """Print per-target accuracy comparison for SWA vs non-SWA runs on val and test sets."""
    print("\n" + "="*105)
    print("SWA VS NON-SWA ACCURACY COMPARISON")
    print("="*105)
    print(f"{'Config':<25} {'Split':<6} {'Task':<15} {'Base Acc':<12} {'SWA Acc':<12} {'Delta':<12} {'% Change':<10}")
    print("-"*105)

    for comp in comparisons:
        config_name = f"{comp['spec_type']}_{comp['task_type']}"
        base_eval = comp.get("base_eval", {})
        swa_eval = comp.get("swa_eval", {})

        # Process both val and test splits
        for split in ["val", "test"]:
            base_split_metrics = base_eval.get(split, {})
            swa_split_metrics = swa_eval.get(split, {})

            # Get all tasks from both base and SWA
            all_tasks = sorted(set(base_split_metrics.keys()) | set(swa_split_metrics.keys()))

            if not all_tasks:
                print(f"{config_name:<25} {split:<6} {'n/a':<15} {'FAILED':<12} {'FAILED':<12} {'N/A':<12} {'N/A':<10}")
                continue

            for task_name in all_tasks:
                base_value = base_split_metrics.get(task_name)
                swa_value = swa_split_metrics.get(task_name)

                base_display = f"{base_value:.4f}" if base_value is not None else "FAILED"
                swa_display = f"{swa_value:.4f}" if swa_value is not None else "FAILED"

                if base_value is not None and swa_value is not None:
                    delta = swa_value - base_value
                    delta_display = f"{delta:+.4f}"
                    # Compute percentage change
                    if base_value != 0:
                        pct_change = (delta / base_value) * 100
                        pct_display = f"{pct_change:+.2f}%"
                    else:
                        pct_display = "N/A"
                else:
                    delta_display = "N/A"
                    pct_display = "N/A"

                print(
                    f"{config_name:<25} {split:<6} {task_name:<15} {base_display:<12} "
                    f"{swa_display:<12} {delta_display:<12} {pct_display:<10}"
                )

    print("="*105)


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

            # Evaluate both base and SWA models on val and test sets
            base_eval_results = {}
            swa_eval_results = {}

            if base_run_dir and base_summary and base_summary.best_checkpoint.exists():
                LOGGER.info(f"Evaluating base model checkpoint: {base_summary.best_checkpoint}")
                base_eval_results = evaluate_checkpoint(base_summary.best_checkpoint, base_run_dir, cwd)

            if swa_run_dir and swa_summary and swa_summary.swa_checkpoint and swa_summary.swa_checkpoint.exists():
                LOGGER.info(f"Evaluating SWA model checkpoint: {swa_summary.swa_checkpoint}")
                swa_eval_results = evaluate_checkpoint(swa_summary.swa_checkpoint, swa_run_dir, cwd)
            elif swa_run_dir and swa_summary and swa_summary.best_checkpoint.exists():
                # Fallback to best checkpoint if SWA checkpoint not available
                LOGGER.info(f"Evaluating SWA model checkpoint (best): {swa_summary.best_checkpoint}")
                swa_eval_results = evaluate_checkpoint(swa_summary.best_checkpoint, swa_run_dir, cwd)

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
                    "swa_checkpoint": str(swa_summary.swa_checkpoint) if swa_summary and swa_summary.swa_checkpoint else None,
                    "error": swa_error,
                },
                "base_eval": base_eval_results,
                "swa_eval": swa_eval_results,
            })
            comparisons.append({
                "spec_type": job.spec_type,
                "task_type": job.task_type,
                "base_eval": base_eval_results,
                "swa_eval": swa_eval_results,
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
