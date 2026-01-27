"""
Hyperparameter tuning for classification training using Optuna.
Uses TPESampler and HyperbandPruner for efficient hyperparameter optimization.
"""

import sys
import gc
import json
import argparse
import logging
import coloredlogs
import contextlib
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

from signal_diffusion.training.classification import (
    load_experiment_config,
    train_from_config,
    ClassificationExperimentConfig,
)
from signal_diffusion.classification import build_task_specs

# Setup logging
coloredlogs.install(level='INFO', fmt='%(levelname)s:%(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Hyperparameter search space (refined based on HPO analysis)
# - Dropout: Narrowed from [0.1, 0.8] → [0.15, 0.50] (top performers cluster here)
# - Embedding dim: Removed 128 → [192, 256, 384] (128 underperformed)
# - Layer repeats: Removed 1 added 5 → [2, 5] (all top performers use 2-4)
# - Learning rate: Narrowed from [1e-5, 1e-2] → [5e-5, 5e-3] (removes ineffective extremes)
# - Weight decay: Narrowed from [1e-5, 1e-1] → [1e-5, 1e-2] (most successful ≤ 1e-2)
# - Batch size: Removed [64, 128] → [192] (size 64 from failed trials, 128 unsuccessful)
# - Label smoothing: [0.0, 0.33] (regularization technique to prevent overconfident predictions)
SEARCH_SPACE = {
    "learning_rate": [5e-5, 5e-3],
    "weight_decay": [1e-5, 1e-2],
    "scheduler": ["constant", "linear", "cosine"],
    "dropout": [0.15, 0.50],
    "depth": [2, 4],
    "layer_repeats": [2, 5],
    "embedding_dim": [192, 256, 384],
    "batch_size": [192],
    "label_smoothing": [0.0, 0.33],
}


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


@contextlib.contextmanager
def preserve_hpo_rng_state():
    state = _capture_rng_state()
    try:
        yield
    finally:
        _restore_rng_state(state)


def _get_float_range(name: str) -> tuple[float, float]:
    values = SEARCH_SPACE.get(name)
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"Expected '{name}' to be a 2-item range, got: {values!r}")
    low, high = values
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        raise ValueError(f"Expected '{name}' range values to be numeric, got: {values!r}")
    return float(low), float(high)


def _get_int_range(name: str) -> tuple[int, int]:
    values = SEARCH_SPACE.get(name)
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"Expected '{name}' to be a 2-item range, got: {values!r}")
    low, high = values
    if not isinstance(low, int) or not isinstance(high, int):
        raise ValueError(f"Expected '{name}' range values to be int, got: {values!r}")
    return low, high


def _parse_prune_max(value: str) -> str | int:
    """Coerce prune max resource argument into an int or 'auto'."""
    normalized = value.strip().lower()
    if normalized == "auto":
        return "auto"

    try:
        max_eval = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--prune-max-steps must be an integer or 'auto'"
        ) from exc

    if max_eval <= 0:
        raise argparse.ArgumentTypeError(
            "--prune-max-steps must be positive or 'auto'"
        )

    return max_eval


def load_base_config(config_path: str | Path) -> ClassificationExperimentConfig:
    """Load base configuration from TOML file."""
    config_path = Path(config_path).expanduser().resolve()
    return load_experiment_config(config_path)


def create_trial_config(
    trial_params: Dict[str, Any],
    base_config: ClassificationExperimentConfig,
) -> ClassificationExperimentConfig:
    """Create configuration for a trial by modifying base config."""
    # Create a copy of the config
    import copy
    config = copy.deepcopy(base_config)

    # Override optimizer parameters
    config.optimizer.learning_rate = trial_params["learning_rate"]
    config.optimizer.weight_decay = trial_params["weight_decay"]

    # Override scheduler
    config.scheduler.name = trial_params["scheduler"]

    # Override model architecture parameters
    config.model.dropout = trial_params["dropout"]
    config.model.depth = trial_params["depth"]
    config.model.layer_repeats = trial_params["layer_repeats"]
    config.model.embedding_dim = trial_params["embedding_dim"]

    # Override batch size
    config.dataset.batch_size = trial_params["batch_size"]

    # Override label smoothing
    config.training.label_smoothing = trial_params["label_smoothing"]

    # Ensure we have intermediate validation for pruning
    if config.training.eval_strategy == "epoch":
        # Switch to steps-based evaluation for better pruning
        config.training.eval_strategy = "steps"
        config.training.eval_steps = 100

    # Enable TensorBoard logging for this trial
    config.training.tensorboard = True

    # Set a unique run name for this trial
    trial_num = trial_params.get("trial_number", 0)
    config.training.run_name = f"trial_{trial_num:03d}"

    return config


def run_training_trial(
    config: ClassificationExperimentConfig,
    trial: optuna.Trial,
    optimize_task: str = "combined",
) -> Dict[str, Any]:
    """Run a single training trial with given configuration.

    Executes a full training run with the trial's hyperparameters, computes task-weighted
    accuracy, and handles resource cleanup. Captures both the final weighted metric and
    individual per-task accuracies for analysis.

    Args:
        config: Configuration for this trial's training run
        trial: Optuna Trial object for reporting metrics and pruning
        optimize_task: Task to optimize for. 'combined' (default) optimizes the mean of all
                      task scores. Specify a task name (e.g., 'gender') to optimize only
                      that task's metric.

    Returns:
        Dictionary with trial results including weighted accuracy, loss, and per-task scores
    """
    try:
        logger.info(f"Starting trial {trial.number} with in-process training")

        if config.training.seed is None:
            raise ValueError(
                "HPO trials require config.training.seed to be set for deterministic runs. "
                "Please set [training] seed in the base configuration."
            )

        # Execute training with trial support (enables intermediate pruning)
        with preserve_hpo_rng_state():
            summary = train_from_config(config, trial=trial)

        # Extract final metrics from training summary
        if summary.history:
            final_epoch = summary.history[-1]
            val_loss = final_epoch.val_loss if final_epoch.val_loss is not None else float('inf')

            # Build task specs to identify task types (classification vs regression)
            task_specs = build_task_specs(config.dataset.name, config.dataset.tasks)
            task_spec_dict = {spec.name: spec for spec in task_specs}

            # Compute combined objective (accuracy for classification, 1/(1+mse) for regression)
            # No task weighting applied - all tasks contribute equally to the objective
            scores = []
            task_results = {}

            for task_name in config.dataset.tasks:
                spec = task_spec_dict.get(task_name)
                if spec is None:
                    continue

                if spec.task_type == "classification":
                    # Classification: use validation accuracy directly
                    accuracy = final_epoch.val_accuracy.get(task_name)
                    if accuracy is not None:
                        scores.append(float(accuracy))
                        task_results[task_name] = {
                            "type": "classification",
                            "accuracy": float(accuracy),
                        }
                else:
                    # Regression: normalize MSE to [0, 1] scale using 1/(1+mse)
                    mse = final_epoch.val_mse.get(task_name)
                    mae = final_epoch.val_mae.get(task_name)
                    if mse is not None:
                        mse_val = max(0.0, float(mse))
                        normalized_score = 1.0 / (1.0 + mse_val)
                        scores.append(normalized_score)
                        task_results[task_name] = {
                            "type": "regression",
                            "mse": mse_val,
                            "mae": float(mae) if mae is not None else None,
                            "normalized_score": normalized_score,
                        }

            # Select objective based on optimization target
            if optimize_task == "combined":
                # Current behavior: mean of all task scores
                combined_objective = sum(scores) / len(scores) if scores else 0.0
            else:
                # Specific task: use only that task's score
                if optimize_task not in task_results:
                    logger.warning(
                        f"Task '{optimize_task}' not found in trial results. "
                        f"Available tasks: {list(task_results.keys())}"
                    )
                    combined_objective = 0.0
                else:
                    result = task_results[optimize_task]
                    if result["type"] == "classification":
                        combined_objective = result["accuracy"]
                    else:
                        combined_objective = result["normalized_score"]
        else:
            # No training history (shouldn't happen with proper training)
            combined_objective = 0.0
            val_loss = float('inf')
            task_results = {}

        # Log trial results with both overall metric and per-task breakdown
        logger.info(f"Trial {trial.number} completed - Loss: {val_loss:.4f}, Combined Objective: {combined_objective:.4f}")
        for task_name, result in task_results.items():
            if result["type"] == "classification":
                logger.info(f"  {task_name} (classification): accuracy={result['accuracy']:.4f}")
            else:
                logger.info(f"  {task_name} (regression): mse={result['mse']:.4f}, score={result['normalized_score']:.4f}")

        # Explicitly release GPU memory to prevent accumulation across sequential trials
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "combined_objective": combined_objective,
            "val_loss": val_loss,
            "task_results": task_results,
            "best_metric": summary.best_metric,
            "best_epoch": summary.best_epoch,
            "success": True,
        }

    except optuna.TrialPruned:
        # Trial was pruned due to poor intermediate performance
        # Return the most recent objective value reported before pruning
        recent_objective = 0.0
        pruned_step = 0

        try:
            # Access intermediate values from the study's trial record
            current_trial = trial.study.trials[trial.number]
            if current_trial.intermediate_values:
                # Get the most recent step (highest step number)
                pruned_step = max(current_trial.intermediate_values.keys())
                recent_objective = current_trial.intermediate_values[pruned_step]
        except (AttributeError, IndexError, KeyError):
            # Fallback if we can't access intermediate values
            pass

        logger.warning(f"Trial {trial.number} was pruned (last objective: {recent_objective:.4f})")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "combined_objective": recent_objective,
            "val_loss": float('inf'),
            "success": False,
            "pruned": True,
            "pruned_at_step": pruned_step,
        }

    except Exception as e:
        # Catch any other training failure
        logger.error(f"Trial {trial.number} failed with error: {e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "combined_objective": 0.0,
            "val_loss": float('inf'),
            "success": False,
            "error": str(e),
        }


def create_objective(base_config: ClassificationExperimentConfig, optimize_task: str = "combined"):
    """Create closure that returns an Optuna-compatible objective function.

    The returned objective function samples hyperparameters from SEARCH_SPACE,
    runs training, and returns the objective for optimization. The objective can be
    either the combined metric (mean of all tasks) or a specific task's metric.

    Args:
        base_config: Base configuration to clone and modify per trial
        optimize_task: Task to optimize for. 'combined' (default) optimizes the mean of all
                      task scores. Specify a task name (e.g., 'gender') to optimize only
                      that task's metric.

    Returns:
        Objective function suitable for study.optimize()
    """

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function for hyperparameter optimization.

        Samples hyperparameters, runs training, and returns the objective score:
        - If optimize_task == 'combined': unweighted mean of all task scores
          - For classification tasks: uses validation accuracy directly
          - For regression tasks: uses 1/(1+mse) to normalize MSE to [0, 1]
        - If optimize_task is a specific task name: uses only that task's metric

        Args:
            trial: Optuna Trial object for sampling and reporting

        Returns:
            Objective score (higher is better, range ~[0, 1])
        """
        # Sample hyperparameters from search space with log scaling for rates
        lr_min, lr_max = _get_float_range("learning_rate")
        learning_rate = trial.suggest_float(
            "learning_rate",
            lr_min,
            lr_max,
            log=True,
        )
        wd_min, wd_max = _get_float_range("weight_decay")
        weight_decay = trial.suggest_float(
            "weight_decay",
            wd_min,
            wd_max,
            log=True,
        )
        scheduler = trial.suggest_categorical("scheduler", SEARCH_SPACE["scheduler"])
        dropout_min, dropout_max = _get_float_range("dropout")
        dropout = trial.suggest_float(
            "dropout",
            dropout_min,
            dropout_max,
            log=False,
        )
        depth_min, depth_max = _get_int_range("depth")
        depth = trial.suggest_int(
            "depth",
            depth_min,
            depth_max,
        )
        repeats_min, repeats_max = _get_int_range("layer_repeats")
        layer_repeats = trial.suggest_int(
            "layer_repeats",
            repeats_min,
            repeats_max,
        )
        embedding_dim = trial.suggest_categorical(
            "embedding_dim",
            SEARCH_SPACE["embedding_dim"],
        )
        batch_size = trial.suggest_categorical(
            "batch_size",
            SEARCH_SPACE["batch_size"],
        )
        ls_min, ls_max = _get_float_range("label_smoothing")
        label_smoothing = trial.suggest_float(
            "label_smoothing",
            ls_min,
            ls_max,
            log=False,
        )

        trial_params = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "scheduler": scheduler,
            "dropout": dropout,
            "depth": depth,
            "layer_repeats": layer_repeats,
            "embedding_dim": embedding_dim,
            "batch_size": batch_size,
            "label_smoothing": label_smoothing,
            "trial_number": trial.number,
        }

        logger.info(f"Starting trial {trial.number} with params: {trial_params}")

        # Create trial-specific config by modifying base config with sampled hyperparameters
        config = create_trial_config(trial_params, base_config)

        # Execute training with this trial's configuration
        results = run_training_trial(config, trial, optimize_task)

        # Store all results as trial user attributes for post-hoc analysis
        trial.set_user_attr("combined_objective", results["combined_objective"])
        trial.set_user_attr("val_loss", results["val_loss"])
        trial.set_user_attr("success", results["success"])

        # Store individual task results (both classification and regression) for debugging
        if "task_results" in results:
            for task_name, task_result in results["task_results"].items():
                if task_result["type"] == "classification":
                    trial.set_user_attr(f"task_{task_name}_accuracy", task_result["accuracy"])
                else:
                    trial.set_user_attr(f"task_{task_name}_mse", task_result["mse"])
                    if task_result.get("mae") is not None:
                        trial.set_user_attr(f"task_{task_name}_mae", task_result["mae"])
                    trial.set_user_attr(f"task_{task_name}_normalized_score", task_result["normalized_score"])

        # Store training metadata
        if "best_metric" in results:
            trial.set_user_attr("best_metric", results["best_metric"])
        if "best_epoch" in results:
            trial.set_user_attr("best_epoch", results["best_epoch"])

        # Return combined objective for optimization
        # Optuna will maximize this value (higher is better)
        return results["combined_objective"]

    return objective


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for classification training"
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        default="config/classification/baseline.toml",
        help="Path to base configuration TOML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for all trial runs",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of trials to run",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for the entire study (0 for unlimited)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="classification_hpo",
        help="Name of the Optuna study",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--prune-min-steps",
        type=int,
        default=3,
        help="Minimum number of steps before pruning (HyperbandPruner min_resource)",
    )
    parser.add_argument(
        "--prune-max-steps",
        type=_parse_prune_max,
        default=_parse_prune_max("auto"),
        help="Maximum number of steps (HyperbandPruner max_resource: int or 'auto')",
    )
    parser.add_argument(
        "--reduction-factor",
        type=int,
        default=3,
        help="Reduction factor for HyperbandPruner",
    )
    parser.add_argument(
        "--optimize-task",
        type=str,
        default="combined",
        help="Task to optimize for: 'combined' (default, mean of all tasks) or specific task name (e.g., 'gender', 'emotion')",
    )
    return parser.parse_args()


def main():
    """Main function to run hyperparameter tuning."""
    # Parse arguments
    args = parse_args()

    logger.info(f"Using base configuration from: {args.config_path}")

    # Load base configuration
    base_config = load_base_config(args.config_path)

    # Validate optimize_task argument
    if args.optimize_task != "combined":
        available_tasks = set(base_config.dataset.tasks)
        if args.optimize_task not in available_tasks:
            logger.error(
                f"Task '{args.optimize_task}' not found in config tasks. "
                f"Available tasks: {sorted(available_tasks)}"
            )
            sys.exit(1)
        logger.info(f"Optimizing for task: {args.optimize_task}")
    else:
        logger.info("Optimizing for combined objective (mean of all tasks)")

    # Override output directory if specified
    if args.output_dir:
        base_config.training.output_dir = Path(args.output_dir).resolve()
        logger.info(f"Overriding output directory to: {base_config.training.output_dir}")

    # Create Optuna study with TPE sampler and Hyperband pruner
    sampler = TPESampler(seed=args.seed)
    pruner = HyperbandPruner(
        min_resource=args.prune_min_steps,
        max_resource=args.prune_max_steps,
        reduction_factor=args.reduction_factor,
    )

    study = optuna.create_study(
        direction="maximize",  # Maximize accuracy
        sampler=sampler,
        pruner=pruner,
        study_name=args.study_name,
        load_if_exists=True,
    )

    logger.info(
        f"Starting hyperparameter tuning with up to {args.n_trials} trials "
        f"using TPE sampler"
    )
    logger.info("Pruning will be performed by HyperbandPruner based on intermediate results")

    # Create objective function with base config and task specification
    objective = create_objective(base_config, args.optimize_task)

    # Convert timeout=0 to None (unlimited)
    timeout = None if args.timeout == 0 else args.timeout

    # Run optimization
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=timeout,
    )

    # Log summary of hyperparameter optimization results
    logger.info("Hyperparameter tuning completed!")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best combined objective: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")

    # Log individual per-task metrics for the best performing trial
    # This helps understand which tasks benefited from the optimal hyperparameters
    best_attrs = study.best_trial.user_attrs
    for attr_name in sorted(best_attrs.keys()):
        attr_value = best_attrs[attr_name]
        if attr_name.startswith("task_") and attr_name.endswith("_accuracy"):
            task_name = attr_name.replace("task_", "").replace("_accuracy", "")
            logger.info(f"  {task_name} (classification) accuracy: {attr_value:.4f}")
        elif attr_name.startswith("task_") and attr_name.endswith("_mse"):
            task_name = attr_name.replace("task_", "").replace("_mse", "")
            logger.info(f"  {task_name} (regression) mse: {attr_value:.4f}")
        elif attr_name.startswith("task_") and attr_name.endswith("_normalized_score"):
            task_name = attr_name.replace("task_", "").replace("_normalized_score", "")
            logger.info(f"  {task_name} (regression) normalized_score: {attr_value:.4f}")

    # Save complete HPO results to JSON for further analysis
    results_path = Path("hpo_study_results.json")
    with results_path.open("w") as f:
        json.dump(
            {
                "best_trial": study.best_trial.number,
                "best_combined_objective": study.best_value,
                "best_params": study.best_params,
                "best_user_attrs": study.best_trial.user_attrs,
                # Include all trials for trend analysis and debugging
                "all_trials": [
                    {
                        "trial": t.number,
                        "value": t.value,
                        "params": t.params,
                        "user_attrs": t.user_attrs,
                        "state": str(t.state),
                    }
                    for t in study.trials
                ],
            },
            f,
            indent=2,
        )

    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
