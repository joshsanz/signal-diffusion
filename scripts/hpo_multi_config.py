#!/usr/bin/env python3
"""
Multi-configuration HPO script for classification.

Runs HPO across different spectrogram types and task objectives:
- Spectrogram types: db-only, db-polar, db-iq
- Task objectives: gender-only, mixed (gender + health + age)

Generates descriptively-named output files and summarizes results.
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import argparse

import coloredlogs

# Setup logging
coloredlogs.install(level='INFO', fmt='%(levelname)s:%(name)s - %(message)s')
logger = logging.getLogger(__name__)


class HPOConfig:
    """Configuration for a single HPO run."""

    def __init__(
        self,
        spec_type: str,
        task_objective: str,
        config_path: str,
        n_trials: int = 50,
        timeout: int = None,
        seed: int = 42,
    ):
        """Initialize HPO config.

        Args:
            spec_type: Spectrogram type ('db-only', 'db-polar', 'db-iq')
            task_objective: Task objective ('gender' or 'mixed')
            config_path: Path to base classification config
            n_trials: Number of trials for this HPO run
            timeout: Timeout in seconds (None for unlimited)
            seed: Random seed
        """
        self.spec_type = spec_type
        self.task_objective = task_objective
        self.config_path = Path(config_path).resolve()
        self.n_trials = n_trials
        self.timeout = timeout
        self.seed = seed

    @property
    def output_name(self) -> str:
        """Generate descriptive output filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"hpo_study_{self.spec_type}_{self.task_objective}_{timestamp}.json"

    def __repr__(self) -> str:
        return (
            f"HPOConfig(spec={self.spec_type}, task={self.task_objective}, "
            f"n_trials={self.n_trials})"
        )


def create_hpo_configs(
    n_trials: int = 50,
    timeout: int = None,
    seed: int = 42,
) -> List[HPOConfig]:
    """Create HPO configurations for all combinations.

    Args:
        n_trials: Number of trials per HPO run
        timeout: Timeout in seconds per run
        seed: Random seed

    Returns:
        List of HPOConfig objects
    """
    spec_types = [
        ("db-only", "config/classification/baseline.toml"),
        ("db-polar", "config/classification/baseline-db-polar.toml"),
        ("db-iq", "config/classification/baseline-db-iq.toml"),
        ("timeseries", "config/classification/baseline-timeseries.toml"),
    ]
    task_objectives = ["gender", "mixed"]

    configs = []
    for spec_type, config_path in spec_types:
        for task_objective in task_objectives:
            configs.append(
                HPOConfig(
                    spec_type=spec_type,
                    task_objective=task_objective,
                    config_path=config_path,
                    n_trials=n_trials,
                    timeout=timeout,
                    seed=seed,
                )
            )

    return configs


def get_optimize_task(task_objective: str) -> str:
    """Get optimize_task argument for HPO script.

    Args:
        task_objective: 'gender' or 'mixed'

    Returns:
        Argument value for --optimize-task
    """
    if task_objective == "gender":
        return "gender"
    elif task_objective == "mixed":
        return "combined"
    else:
        raise ValueError(f"Unknown task objective: {task_objective}")


def run_hpo(config: HPOConfig, output_dir: Path) -> tuple[bool, Path]:
    """Run a single HPO study.

    Args:
        config: HPO configuration
        output_dir: Directory to save results

    Returns:
        Tuple of (success: bool, results_path: Path)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting HPO: {config}")
    logger.info(f"{'='*80}\n")

    # Build HPO command
    optimize_task = get_optimize_task(config.task_objective)
    cmd = [
        "uv",
        "run",
        "python",
        "hpo/classification_hpo.py",
        str(config.config_path),
        "--n-trials",
        str(config.n_trials),
        "--seed",
        str(config.seed),
        "--optimize-task",
        optimize_task,
        "--output-dir",
        str(output_dir / f"trial_runs_{config.spec_type}_{config.task_objective}"),
        "--prune-max-steps", "auto",
        "--prune-min-steps", "300",
    ]

    if config.timeout is not None:
        cmd.extend(["--timeout", str(config.timeout)])

    # Add study name based on config
    study_name = f"hpo_{config.spec_type}_{config.task_objective}"
    cmd.extend(["--study-name", study_name])

    logger.info(f"Running command: {' '.join(cmd)}\n")

    try:
        # Run HPO
        result = subprocess.run(cmd, cwd="/home/jsanz/git/signal-diffusion", check=True)

        # Find and rename the generated hpo_study_results.json
        hpo_results_path = Path("/home/jsanz/git/signal-diffusion/hpo_study_results.json")
        if hpo_results_path.exists():
            output_path = output_dir / config.output_name
            hpo_results_path.rename(output_path)
            logger.info(f"HPO results saved to: {output_path}")
            return True, output_path
        else:
            logger.error(f"HPO results file not found at {hpo_results_path}")
            return False, None

    except subprocess.CalledProcessError as e:
        logger.error(f"HPO run failed with return code {e.returncode}")
        return False, None
    except Exception as e:
        logger.error(f"HPO run failed with error: {e}")
        return False, None


def load_hpo_results(results_path: Path) -> Dict[str, Any]:
    """Load HPO results from JSON file.

    Args:
        results_path: Path to hpo_study_results.json

    Returns:
        Parsed JSON data
    """
    with open(results_path, "r") as f:
        return json.load(f)


def summarize_trial_metrics(hpo_results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and summarize best trial metrics.

    Args:
        hpo_results: Parsed HPO results

    Returns:
        Dictionary with summarized metrics
    """
    best_trial = hpo_results["best_trial"]
    best_user_attrs = hpo_results["best_user_attrs"]

    summary = {
        "best_trial_num": best_trial,
        "best_objective": hpo_results["best_combined_objective"],
        "best_params": hpo_results["best_params"],
    }

    # Extract task metrics
    task_metrics = {}
    for attr_name, attr_value in best_user_attrs.items():
        if attr_name.startswith("task_") and attr_name.endswith("_accuracy"):
            task_name = attr_name.replace("task_", "").replace("_accuracy", "")
            if task_name not in task_metrics:
                task_metrics[task_name] = {}
            task_metrics[task_name]["accuracy"] = attr_value
        elif attr_name.startswith("task_") and attr_name.endswith("_mse"):
            task_name = attr_name.replace("task_", "").replace("_mse", "")
            if task_name not in task_metrics:
                task_metrics[task_name] = {}
            task_metrics[task_name]["mse"] = attr_value
        elif attr_name.startswith("task_") and attr_name.endswith("_mae"):
            task_name = attr_name.replace("task_", "").replace("_mae", "")
            if task_name not in task_metrics:
                task_metrics[task_name] = {}
            task_metrics[task_name]["mae"] = attr_value

    summary["task_metrics"] = task_metrics
    return summary


def format_summary_table(results: Dict[str, Dict[str, Any]]) -> str:
    """Format results as readable summary table.

    Args:
        results: Dictionary of {config_name: summary}

    Returns:
        Formatted table string
    """
    lines = []
    lines.append("\n" + "=" * 120)
    lines.append("HPO SUMMARY - BEST TRIAL METRICS")
    lines.append("=" * 120)

    # Header
    lines.append(
        f"{'Config':<30} {'Spec Type':<12} {'Task Obj':<10} {'Gender Acc':<12} "
        f"{'Health Acc':<12} {'Age (MSE)':<12} {'Age (MAE)':<12}"
    )
    lines.append("-" * 120)

    # Data rows
    for config_name, summary in results.items():
        spec_type, task_obj = config_name.rsplit("_", 1)

        gender_acc = "N/A"
        health_acc = "N/A"
        age_mse = "N/A"
        age_mae = "N/A"

        task_metrics = summary.get("task_metrics", {})

        if "gender" in task_metrics:
            gender_acc = f"{task_metrics['gender'].get('accuracy', 0):.4f}"
        if "health" in task_metrics:
            health_acc = f"{task_metrics['health'].get('accuracy', 0):.4f}"
        if "age" in task_metrics:
            age_mse = f"{task_metrics['age'].get('mse', 0):.4f}"
            age_mae = f"{task_metrics['age'].get('mae', 0):.4f}"

        lines.append(
            f"{config_name:<30} {spec_type:<12} {task_obj:<10} "
            f"{gender_acc:<12} {health_acc:<12} {age_mse:<12} {age_mae:<12}"
        )

    lines.append("=" * 120)
    return "\n".join(lines)


def save_summary(results: Dict[str, Dict[str, Any]], output_dir: Path) -> None:
    """Save summary as JSON file.

    Args:
        results: Dictionary of {config_name: summary}
        output_dir: Directory to save summary
    """
    summary_path = output_dir / "hpo_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Summary saved to: {summary_path}")


def main():
    """Main function to run multi-config HPO."""
    parser = argparse.ArgumentParser(
        description="Run HPO across multiple spectrogram types and task objectives"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of trials per HPO run",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds per HPO run (None for unlimited)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for HPO results (default: ./hpo_results)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--skip-mixed",
        action="store_true",
        help="Skip mixed task objective runs (only gender)",
    )
    parser.add_argument(
        "--skip-gender",
        action="store_true",
        help="Skip gender-only task objective runs (only mixed)",
    )
    parser.add_argument(
        "--spec-types",
        type=str,
        nargs="+",
        default=None,
        help="Spec types to run (default: all). Choose from: db-only, db-polar, db-iq, timeseries",
    )

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = Path("/home/jsanz/git/signal-diffusion/hpo_results")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Create HPO configs
    hpo_configs = create_hpo_configs(
        n_trials=args.n_trials,
        timeout=args.timeout,
        seed=args.seed,
    )

    # Filter configs if requested
    if args.skip_mixed:
        hpo_configs = [c for c in hpo_configs if c.task_objective != "mixed"]
        logger.info("Skipping mixed task objective runs")

    if args.skip_gender:
        hpo_configs = [c for c in hpo_configs if c.task_objective != "gender"]
        logger.info("Skipping gender-only task objective runs")

    if args.spec_types:
        hpo_configs = [c for c in hpo_configs if c.spec_type in args.spec_types]
        logger.info(f"Running only spec types: {args.spec_types}")

    logger.info(f"Total HPO runs to execute: {len(hpo_configs)}")
    for cfg in hpo_configs:
        logger.info(f"  - {cfg}")

    # Run all HPO studies
    results = {}
    results_files = {}

    for i, config in enumerate(hpo_configs, 1):
        logger.info(f"\n[{i}/{len(hpo_configs)}] Running: {config}")

        success, results_path = run_hpo(config, output_dir)

        config_name = f"{config.spec_type}_{config.task_objective}"

        if success:
            try:
                hpo_results = load_hpo_results(results_path)
                summary = summarize_trial_metrics(hpo_results)
                results[config_name] = summary
                results_files[config_name] = str(results_path)
                logger.info(f"✓ HPO completed successfully for {config_name}")
                logger.info(f"  Best trial: #{summary['best_trial_num']}")
                logger.info(f"  Best objective: {summary['best_objective']:.4f}")
            except Exception as e:
                logger.error(f"✗ Failed to load results for {config_name}: {e}")
        else:
            logger.error(f"✗ HPO run failed for {config_name}")

    # Print summary table
    if results:
        logger.info(format_summary_table(results))

    # Save summary
    save_summary(results, output_dir)

    # Log results files
    logger.info(f"\n{'='*80}")
    logger.info("HPO Results Files:")
    logger.info(f"{'='*80}")
    for config_name, path in results_files.items():
        logger.info(f"  {config_name:<25} -> {path}")

    logger.info(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
