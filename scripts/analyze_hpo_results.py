#!/usr/bin/env python3
"""
Analyze and display HPO results from completed runs.

Provides utilities to parse HPO results and generate formatted summaries,
comparisons, and detailed analysis.
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

import coloredlogs

coloredlogs.install(level='INFO', fmt='%(levelname)s:%(name)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TrialMetrics:
    """Parsed metrics for a trial."""

    trial_num: int
    objective: float
    gender_acc: float = None
    health_acc: float = None
    age_mse: float = None
    age_mae: float = None
    hyperparams: Dict[str, Any] = None

    def __repr__(self) -> str:
        parts = [f"trial_{self.trial_num}: obj={self.objective:.4f}"]
        if self.gender_acc is not None:
            parts.append(f"gender={self.gender_acc:.4f}")
        if self.health_acc is not None:
            parts.append(f"health={self.health_acc:.4f}")
        if self.age_mse is not None:
            parts.append(f"age_mse={self.age_mse:.4f}")
        if self.age_mae is not None:
            parts.append(f"age_mae={self.age_mae:.4f}")
        return " | ".join(parts)


@dataclass
class HPORunSummary:
    """Summary of an HPO run."""

    config_name: str
    results_path: Path
    best_metrics: TrialMetrics
    num_trials: int
    success_rate: float

    def __repr__(self) -> str:
        return (
            f"{self.config_name:<25} | "
            f"Best: {self.best_metrics.objective:.4f} "
            f"(trial {self.best_metrics.trial_num}) | "
            f"Trials: {self.num_trials} ({self.success_rate:.1%} success)"
        )


def load_hpo_results(results_path: Path) -> Dict[str, Any]:
    """Load HPO results from JSON file.

    Args:
        results_path: Path to hpo_study_results.json

    Returns:
        Parsed JSON data
    """
    with open(results_path, "r") as f:
        return json.load(f)


def parse_trial_metrics(
    trial_num: int,
    best_user_attrs: Dict[str, Any],
    params: Dict[str, Any] = None,
) -> TrialMetrics:
    """Extract metrics from trial user attributes.

    Args:
        trial_num: Trial number
        best_user_attrs: User attributes from best trial
        params: Hyperparameters for the trial

    Returns:
        TrialMetrics object
    """
    metrics = TrialMetrics(
        trial_num=trial_num,
        objective=best_user_attrs.get("combined_objective", 0.0),
        hyperparams=params or {},
    )

    # Parse task metrics
    for attr_name, attr_value in best_user_attrs.items():
        if attr_name == "task_gender_accuracy":
            metrics.gender_acc = attr_value
        elif attr_name == "task_health_accuracy":
            metrics.health_acc = attr_value
        elif attr_name == "task_age_mse":
            metrics.age_mse = attr_value
        elif attr_name == "task_age_mae":
            metrics.age_mae = attr_value

    return metrics


def analyze_hpo_directory(results_dir: Path) -> List[HPORunSummary]:
    """Analyze all HPO results in a directory.

    Args:
        results_dir: Directory containing hpo_study_*.json files

    Returns:
        List of HPORunSummary objects
    """
    summaries = []

    hpo_files = sorted(results_dir.glob("hpo_study_*.json"))
    logger.info(f"Found {len(hpo_files)} HPO result files")

    for results_path in hpo_files:
        # Extract config name from filename
        # Format: hpo_study_{spec_type}_{task_obj}_{timestamp}.json
        parts = results_path.stem.split("_")
        if len(parts) >= 4:
            config_name = f"{parts[2]}_{parts[3]}"
        else:
            config_name = results_path.stem

        try:
            hpo_results = load_hpo_results(results_path)

            best_trial = hpo_results["best_trial"]
            best_user_attrs = hpo_results["best_user_attrs"]
            best_params = hpo_results.get("best_params", {})
            all_trials = hpo_results["all_trials"]

            # Parse metrics
            best_metrics = parse_trial_metrics(best_trial, best_user_attrs, best_params)

            # Calculate success rate
            completed_trials = sum(
                1 for t in all_trials if t.get("user_attrs", {}).get("success", False)
            )
            success_rate = completed_trials / len(all_trials)

            summary = HPORunSummary(
                config_name=config_name,
                results_path=results_path,
                best_metrics=best_metrics,
                num_trials=len(all_trials),
                success_rate=success_rate,
            )
            summaries.append(summary)

        except Exception as e:
            logger.error(f"Failed to parse {results_path}: {e}")

    return summaries


def print_summary_table(summaries: List[HPORunSummary]) -> None:
    """Print formatted summary table.

    Args:
        summaries: List of HPORunSummary objects
    """
    print("\n" + "=" * 140)
    print("HPO RESULTS SUMMARY")
    print("=" * 140)
    print(
        f"{'Config':<25} {'Best Objective':<18} {'Best Trial':<12} "
        f"{'Num Trials':<12} {'Success Rate':<15} {'File':<30}"
    )
    print("-" * 140)

    for summary in sorted(summaries, key=lambda s: s.best_metrics.objective, reverse=True):
        print(
            f"{summary.config_name:<25} {summary.best_metrics.objective:<18.4f} "
            f"{summary.best_metrics.trial_num:<12d} {summary.num_trials:<12d} "
            f"{summary.success_rate:<15.1%} {summary.results_path.name:<30}"
        )

    print("=" * 140 + "\n")


def print_detailed_metrics(summaries: List[HPORunSummary]) -> None:
    """Print detailed metrics table.

    Args:
        summaries: List of HPORunSummary objects
    """
    print("\n" + "=" * 140)
    print("DETAILED METRICS - BEST TRIALS")
    print("=" * 140)
    print(
        f"{'Config':<25} {'Gender Acc':<15} {'Health Acc':<15} "
        f"{'Age MSE':<15} {'Age MAE':<15}"
    )
    print("-" * 140)

    for summary in summaries:
        m = summary.best_metrics
        gender = f"{m.gender_acc:.4f}" if m.gender_acc is not None else "N/A"
        health = f"{m.health_acc:.4f}" if m.health_acc is not None else "N/A"
        age_mse = f"{m.age_mse:.4f}" if m.age_mse is not None else "N/A"
        age_mae = f"{m.age_mae:.4f}" if m.age_mae is not None else "N/A"

        print(
            f"{summary.config_name:<25} {gender:<15} {health:<15} "
            f"{age_mse:<15} {age_mae:<15}"
        )

    print("=" * 140 + "\n")


def print_best_comparison(
    summaries: List[HPORunSummary],
    metric: str = "objective",
) -> None:
    """Print best runs by metric.

    Args:
        summaries: List of HPORunSummary objects
        metric: Metric to sort by ('objective', 'gender', 'health', 'age_mse', 'age_mae')
    """
    print(f"\nTop Runs by {metric}:".upper())
    print("=" * 100)

    def get_metric_value(summary: HPORunSummary) -> float:
        m = summary.best_metrics
        if metric == "objective":
            return m.objective
        elif metric == "gender":
            return m.gender_acc if m.gender_acc is not None else -1
        elif metric == "health":
            return m.health_acc if m.health_acc is not None else -1
        elif metric == "age_mse":
            return -m.age_mse if m.age_mse is not None else float("inf")  # Lower is better
        elif metric == "age_mae":
            return -m.age_mae if m.age_mae is not None else float("inf")  # Lower is better
        return 0

    sorted_summaries = sorted(
        summaries,
        key=get_metric_value,
        reverse=True,
    )

    for i, summary in enumerate(sorted_summaries[:5], 1):
        print(f"\n{i}. {summary.config_name:<25} | {summary.best_metrics}")

        # Print hyperparameters for this run
        if summary.best_metrics.hyperparams:
            print("   Hyperparameters:")
            hyperparams = summary.best_metrics.hyperparams
            for param_name, param_value in sorted(hyperparams.items()):
                if isinstance(param_value, float):
                    print(f"     • {param_name:<20} = {param_value:.6g}")
                else:
                    print(f"     • {param_name:<20} = {param_value}")

    print("\n" + "=" * 100)


def print_top_hyperparameters_by_dataset(summaries: List[HPORunSummary]) -> None:
    """Print top 5 hyperparameter settings for each dataset type.

    Args:
        summaries: List of HPORunSummary objects
    """
    from collections import defaultdict

    # Group summaries by dataset type (e.g., "db-iq", "db-only", "db-polar")
    dataset_groups = defaultdict(list)
    for summary in summaries:
        # Extract dataset type from config_name
        # Format: "{spec_type}_{task}" (e.g., "db-iq_gender", "db-only_mixed")
        parts = summary.config_name.split("_")
        if len(parts) >= 2:
            # Handle multi-part dataset types like "db-iq" or "db-polar"
            if parts[0] == "db" and len(parts) >= 2:
                dataset_type = f"{parts[0]}-{parts[1]}"
            else:
                dataset_type = parts[0]
        else:
            dataset_type = summary.config_name

        dataset_groups[dataset_type].append(summary)

    # Mapping of parameter names to shortened headers
    header_map = {
        "batch_size": "Batch",
        "depth": "Depth",
        "dropout": "Dropout",
        "embedding_dim": "Embed",
        "layer_repeats": "Layers",
        "learning_rate": "LR",
        "scheduler": "Sched",
        "weight_decay": "WD",
    }

    # Print table for each dataset type
    for dataset_type in sorted(dataset_groups.keys()):
        group_summaries = dataset_groups[dataset_type]

        # Sort by objective score (descending)
        sorted_group = sorted(
            group_summaries,
            key=lambda s: s.best_metrics.objective,
            reverse=True,
        )[:5]  # Top 5

        print(f"\nTOP 5 HYPERPARAMETERS - {dataset_type.upper()}")
        print("=" * 110)

        # Collect all unique hyperparameter keys from this group
        all_param_keys = set()
        for summary in sorted_group:
            if summary.best_metrics.hyperparams:
                all_param_keys.update(summary.best_metrics.hyperparams.keys())

        # Sort param keys for consistent ordering
        param_keys = sorted(all_param_keys)

        # Build header with shortened names
        header_parts = ["Obj", "Task", "Trial"]
        for key in param_keys:
            header_parts.append(header_map.get(key, key.replace("_", " ").title()[:8]))

        # Calculate column widths based on content type
        col_widths = [6, 8, 5]  # Obj, Task, Trial
        for key in param_keys:
            if key in ["batch_size", "embedding_dim"]:
                col_widths.append(6)
            elif key in ["depth", "layer_repeats"]:
                col_widths.append(5)
            elif key in ["dropout", "learning_rate", "weight_decay"]:
                col_widths.append(9)
            elif key == "scheduler":
                col_widths.append(8)
            else:
                col_widths.append(8)

        # Print header
        header_line = " | ".join(
            header_parts[i].ljust(col_widths[i]) for i in range(len(header_parts))
        )
        print(header_line)
        print("-" * len(header_line))

        # Print each run
        for summary in sorted_group:
            # Extract task name from config_name
            parts = summary.config_name.split("_")
            if len(parts) >= 2:
                # For "db-iq_gender", task is everything after dataset type
                if parts[0] == "db" and len(parts) >= 3:
                    task_name = "_".join(parts[2:])
                else:
                    task_name = "_".join(parts[1:])
            else:
                task_name = "unknown"

            # Build row data
            row_parts = [
                f"{summary.best_metrics.objective:.4f}",
                task_name[:10],
                str(summary.best_metrics.trial_num),
            ]

            # Add hyperparameter values with custom formatting
            for param_key in param_keys:
                if summary.best_metrics.hyperparams and param_key in summary.best_metrics.hyperparams:
                    param_value = summary.best_metrics.hyperparams[param_key]
                    if isinstance(param_value, float):
                        # Special formatting for specific parameters
                        if param_key in ["learning_rate", "weight_decay"]:
                            # Scientific notation with 2 significant figures
                            row_parts.append(f"{param_value:.1e}")
                        elif param_key == "dropout":
                            # Two decimal places
                            row_parts.append(f"{param_value:.2f}")
                        else:
                            row_parts.append(f"{param_value:.6g}")
                    else:
                        row_parts.append(str(param_value)[:10])
                else:
                    row_parts.append("N/A")

            # Print row
            row_line = " | ".join(
                row_parts[i].ljust(col_widths[i]) for i in range(len(row_parts))
            )
            print(row_line)

        print("=" * len(header_line))


def save_comparison_json(
    summaries: List[HPORunSummary],
    output_path: Path,
) -> None:
    """Save comparison as JSON file.

    Args:
        summaries: List of HPORunSummary objects
        output_path: Output file path
    """
    comparison = {}
    for summary in summaries:
        m = summary.best_metrics
        comparison[summary.config_name] = {
            "best_trial": m.trial_num,
            "best_objective": m.objective,
            "gender_accuracy": m.gender_acc,
            "health_accuracy": m.health_acc,
            "age_mse": m.age_mse,
            "age_mae": m.age_mae,
            "num_trials": summary.num_trials,
            "success_rate": summary.success_rate,
            "results_file": str(summary.results_path),
        }

    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"Comparison saved to {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Analyze and display HPO results"
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="hpo_results",
        help="Directory containing HPO results (default: hpo_results)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["objective", "gender", "health", "age_mse", "age_mae"],
        default="objective",
        help="Metric to rank runs by",
    )
    parser.add_argument(
        "--save-comparison",
        type=str,
        default=None,
        help="Save comparison as JSON file",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print detailed metrics table",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return

    logger.info(f"Analyzing HPO results in {results_dir}")

    # Analyze directory
    summaries = analyze_hpo_directory(results_dir)

    if not summaries:
        logger.warning("No HPO results found")
        return

    # Print summary
    print_summary_table(summaries)

    # Print detailed metrics if requested
    if args.detailed:
        print_detailed_metrics(summaries)

    # Print best comparison
    print_best_comparison(summaries, metric=args.metric)

    # Print top hyperparameters by dataset type
    print_top_hyperparameters_by_dataset(summaries)

    # Save comparison if requested
    if args.save_comparison:
        save_comparison_json(summaries, Path(args.save_comparison))


if __name__ == "__main__":
    main()
