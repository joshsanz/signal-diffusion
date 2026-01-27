"""HPO results formatting and display utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .metrics import TrialSummary


class HPOResultsFormatter:
    """Formatter for HPO results display."""

    @staticmethod
    def format_summary_table(
        results: dict[str, dict],
        show_files: bool = True
    ) -> str:
        """
        Format HPO results as a summary table.

        Args:
            results: Dictionary of {config_name: summary_dict}
            show_files: Whether to show results file names

        Returns:
            Formatted table string
        """
        lines = []

        if show_files:
            lines.append("\n" + "=" * 140)
            lines.append("HPO RESULTS SUMMARY")
            lines.append("=" * 140)
            lines.append(
                f"{'Config':<25} {'Best Objective':<18} {'Best Trial':<12} "
                f"{'Num Trials':<12} {'Success Rate':<15} {'File':<30}"
            )
            lines.append("-" * 140)
        else:
            lines.append("\n" + "=" * 120)
            lines.append("HPO SUMMARY - BEST TRIAL METRICS")
            lines.append("=" * 120)
            lines.append(
                f"{'Config':<30} {'Spec Type':<12} {'Task Obj':<10} {'Gender Acc':<12} "
                f"{'Health Acc':<12} {'Age (MSE)':<12} {'Age (MAE)':<12}"
            )
            lines.append("-" * 120)

        # Data rows - format depends on structure
        for config_name, summary in results.items():
            if show_files:
                # Format for file-based summary
                best_obj = summary.get("best_objective", 0.0)
                best_trial = summary.get("best_trial_num", 0)
                num_trials = summary.get("num_trials", 0)
                success_rate = summary.get("success_rate", 0.0)
                results_file = summary.get("results_file", "N/A")

                lines.append(
                    f"{config_name:<25} {best_obj:<18.4f} "
                    f"{best_trial:<12d} {num_trials:<12d} "
                    f"{success_rate:<15.1%} {Path(results_file).name if results_file != 'N/A' else 'N/A':<30}"
                )
            else:
                # Format for task metrics
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

        lines.append("=" * (140 if show_files else 120))
        return "\n".join(lines)

    @staticmethod
    def format_detailed_metrics(summaries: list[dict]) -> str:
        """
        Format detailed task metrics table.

        Args:
            summaries: List of summary dictionaries with best_metrics

        Returns:
            Formatted table string
        """
        lines = []
        lines.append("\n" + "=" * 140)
        lines.append("DETAILED METRICS - BEST TRIALS")
        lines.append("=" * 140)
        lines.append(
            f"{'Config':<25} {'Gender Acc':<15} {'Health Acc':<15} "
            f"{'Age MSE':<15} {'Age MAE':<15}"
        )
        lines.append("-" * 140)

        for summary in summaries:
            config_name = summary.get("config_name", "unknown")
            best_metrics = summary.get("best_metrics", {})

            # Handle both dict and dataclass-style access
            if hasattr(best_metrics, "task_metrics"):
                # TrialSummary object
                task_metrics = best_metrics.task_metrics
                gender = f"{task_metrics['gender'].accuracy:.4f}" if "gender" in task_metrics and task_metrics["gender"].accuracy is not None else "N/A"
                health = f"{task_metrics['health'].accuracy:.4f}" if "health" in task_metrics and task_metrics["health"].accuracy is not None else "N/A"
                age_mse = f"{task_metrics['age'].mse:.4f}" if "age" in task_metrics and task_metrics["age"].mse is not None else "N/A"
                age_mae = f"{task_metrics['age'].mae:.4f}" if "age" in task_metrics and task_metrics["age"].mae is not None else "N/A"
            else:
                # Dict-based access
                gender = f"{best_metrics.get('gender_acc', 0):.4f}" if best_metrics.get("gender_acc") is not None else "N/A"
                health = f"{best_metrics.get('health_acc', 0):.4f}" if best_metrics.get("health_acc") is not None else "N/A"
                age_mse = f"{best_metrics.get('age_mse', 0):.4f}" if best_metrics.get("age_mse") is not None else "N/A"
                age_mae = f"{best_metrics.get('age_mae', 0):.4f}" if best_metrics.get("age_mae") is not None else "N/A"

            lines.append(
                f"{config_name:<25} {gender:<15} {health:<15} "
                f"{age_mse:<15} {age_mae:<15}"
            )

        lines.append("=" * 140)
        return "\n".join(lines)

    @staticmethod
    def format_best_comparison(
        summaries: list[TrialSummary],
        metric: str = "objective",
        top_n: int = 5
    ) -> str:
        """
        Format best runs comparison by metric.

        Args:
            summaries: List of TrialSummary objects
            metric: Metric to sort by
            top_n: Number of top results to show

        Returns:
            Formatted comparison string
        """
        lines = []
        lines.append(f"\nTop {top_n} Runs by {metric}:".upper())
        lines.append("=" * 100)

        def get_metric_value(summary: TrialSummary) -> float:
            if metric == "objective":
                return summary.objective
            elif metric in summary.task_metrics:
                task_metric = summary.task_metrics[metric]
                if task_metric.accuracy is not None:
                    return task_metric.accuracy
                elif task_metric.mae is not None:
                    return -task_metric.mae  # Lower is better
                elif task_metric.mse is not None:
                    return -task_metric.mse  # Lower is better
            return -float("inf")

        sorted_summaries = sorted(
            summaries,
            key=get_metric_value,
            reverse=True,
        )

        for i, summary in enumerate(sorted_summaries[:top_n], 1):
            lines.append(f"\n{i}. {summary}")

            # Print hyperparameters
            if summary.hyperparams:
                lines.append("   Hyperparameters:")
                for param_name, param_value in sorted(summary.hyperparams.items()):
                    if isinstance(param_value, float):
                        lines.append(f"     • {param_name:<20} = {param_value:.6g}")
                    else:
                        lines.append(f"     • {param_name:<20} = {param_value}")

        lines.append("\n" + "=" * 100)
        return "\n".join(lines)


# Import Path for file name extraction
from pathlib import Path
