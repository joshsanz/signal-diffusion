"""
Hyperparameter tuning for classification training using Optuna.
Uses TPESampler and HyperbandPruner for efficient hyperparameter optimization.
"""

import os
import sys
import gc
import json
import argparse
import logging
import coloredlogs
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

from signal_diffusion.training.classification import (
    load_experiment_config,
    train_from_config,
    ClassificationExperimentConfig,
)

# Setup logging
coloredlogs.install(level='INFO', fmt='%(levelname)s:%(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Hyperparameter search space
SEARCH_SPACE = {
    "learning_rate": [1e-5, 1e-2],  # log scale range
    "weight_decay": [1e-5, 1e-1],   # log scale range
    "scheduler": ["constant", "linear", "cosine"],
    "dropout": [0.1, 0.8],
    "depth": [2, 4],
    "layer_repeats": [1, 2],
    "embedding_dim": [32, 64, 128, 192,],
}


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
) -> Dict[str, Any]:
    """Run a single training trial with given configuration.

    Executes a full training run with the trial's hyperparameters, computes task-weighted
    accuracy, and handles resource cleanup. Captures both the final weighted metric and
    individual per-task accuracies for analysis.

    Args:
        config: Configuration for this trial's training run
        trial: Optuna Trial object for reporting metrics and pruning

    Returns:
        Dictionary with trial results including weighted accuracy, loss, and per-task scores
    """
    try:
        logger.info(f"Starting trial {trial.number} with in-process training")

        # Execute training with trial support (enables intermediate pruning)
        summary = train_from_config(config, trial=trial)

        # Extract final metrics from training summary
        if summary.history:
            final_epoch = summary.history[-1]
            val_loss = final_epoch.val_loss if final_epoch.val_loss is not None else float('inf')

            # Compute task-weighted mean accuracy for optimization objective
            # This respects task_weights from config to handle imbalanced tasks
            task_weights = config.training.task_weights or {}
            weighted_sum = 0.0
            weight_total = 0.0
            task_accuracies = {}

            for task_name, accuracy in final_epoch.val_accuracy.items():
                if accuracy is not None:
                    # Each task contributes to weighted sum proportional to its weight
                    weight = float(task_weights.get(task_name, 1.0))
                    weighted_sum += accuracy * weight
                    weight_total += weight
                    task_accuracies[task_name] = accuracy

            # Compute final weighted accuracy: Σ(accuracy_i × weight_i) / Σ(weight_i)
            mean_accuracy = weighted_sum / weight_total if weight_total > 0 else 0.0
        else:
            # No training history (shouldn't happen with proper training)
            mean_accuracy = 0.0
            val_loss = float('inf')
            task_accuracies = {}

        # Log trial results with both overall metric and per-task breakdown
        logger.info(f"Trial {trial.number} completed - Loss: {val_loss:.4f}, Weighted Accuracy: {mean_accuracy:.4f}")
        for task_name, accuracy in task_accuracies.items():
            weight = float(config.training.task_weights.get(task_name, 1.0)) if config.training.task_weights else 1.0
            logger.info(f"  {task_name}: accuracy={accuracy:.4f}, weight={weight:.4f}")

        # Explicitly release GPU memory to prevent accumulation across sequential trials
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "mean_accuracy": mean_accuracy,
            "val_loss": val_loss,
            "task_accuracies": task_accuracies,
            "best_metric": summary.best_metric,
            "best_epoch": summary.best_epoch,
            "success": True,
        }

    except optuna.TrialPruned:
        # Trial was pruned due to poor intermediate performance
        # Return the most recent objective value (accuracy) reported before pruning
        recent_accuracy = 0.0
        pruned_step = 0

        try:
            # Access intermediate values from the study's trial record
            current_trial = trial.study.trials[trial.number]
            if current_trial.intermediate_values:
                # Get the most recent step (highest step number)
                pruned_step = max(current_trial.intermediate_values.keys())
                recent_accuracy = current_trial.intermediate_values[pruned_step]
        except (AttributeError, IndexError, KeyError):
            # Fallback if we can't access intermediate values
            pass

        logger.warning(f"Trial {trial.number} was pruned (last accuracy: {recent_accuracy:.4f})")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "mean_accuracy": recent_accuracy,
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
            "mean_accuracy": 0.0,
            "val_loss": float('inf'),
            "success": False,
            "error": str(e),
        }


def create_objective(base_config: ClassificationExperimentConfig):
    """Create closure that returns an Optuna-compatible objective function.

    The returned objective function samples hyperparameters from SEARCH_SPACE,
    runs training, and returns the task-weighted accuracy for optimization.

    Args:
        base_config: Base configuration to clone and modify per trial

    Returns:
        Objective function suitable for study.optimize()
    """

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function for hyperparameter optimization.

        Samples hyperparameters, runs training, and returns the weighted accuracy
        metric for Optuna to maximize.

        Args:
            trial: Optuna Trial object for sampling and reporting

        Returns:
            Weighted mean accuracy (higher is better)
        """
        # Sample hyperparameters from search space with log scaling for rates
        learning_rate = trial.suggest_float(
            "learning_rate",
            SEARCH_SPACE["learning_rate"][0],
            SEARCH_SPACE["learning_rate"][1],
            log=True,
        )
        weight_decay = trial.suggest_float(
            "weight_decay",
            SEARCH_SPACE["weight_decay"][0],
            SEARCH_SPACE["weight_decay"][1],
            log=True,
        )
        scheduler = trial.suggest_categorical("scheduler", SEARCH_SPACE["scheduler"])
        dropout = trial.suggest_float(
            "dropout",
            SEARCH_SPACE["dropout"][0],
            SEARCH_SPACE["dropout"][1],
            log=False,
        )
        depth = trial.suggest_int(
            "depth",
            SEARCH_SPACE["depth"][0],
            SEARCH_SPACE["depth"][1],
        )
        layer_repeats = trial.suggest_int(
            "layer_repeats",
            SEARCH_SPACE["layer_repeats"][0],
            SEARCH_SPACE["layer_repeats"][1],
        )
        embedding_dim = trial.suggest_categorical(
            "embedding_dim",
            SEARCH_SPACE["embedding_dim"],
        )

        trial_params = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "scheduler": scheduler,
            "dropout": dropout,
            "depth": depth,
            "layer_repeats": layer_repeats,
            "embedding_dim": embedding_dim,
            "trial_number": trial.number,
        }

        logger.info(f"Starting trial {trial.number} with params: {trial_params}")

        # Create trial-specific config by modifying base config with sampled hyperparameters
        config = create_trial_config(trial_params, base_config)

        # Execute training with this trial's configuration
        results = run_training_trial(config, trial)

        # Store all results as trial user attributes for post-hoc analysis
        trial.set_user_attr("weighted_accuracy", results["mean_accuracy"])
        trial.set_user_attr("val_loss", results["val_loss"])
        trial.set_user_attr("success", results["success"])

        # Store individual task accuracies for debugging and analysis
        if "task_accuracies" in results:
            for task_name, accuracy in results["task_accuracies"].items():
                trial.set_user_attr(f"task_{task_name}_accuracy", accuracy)

        # Store training metadata
        if "best_metric" in results:
            trial.set_user_attr("best_metric", results["best_metric"])
        if "best_epoch" in results:
            trial.set_user_attr("best_epoch", results["best_epoch"])

        # Return weighted mean accuracy for optimization
        # Optuna will maximize this value (higher accuracy is better)
        return results["mean_accuracy"]

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
        "--prune-min-evals",
        type=int,
        default=3,
        help="Minimum number of evaluations before pruning (HyperbandPruner min_resource)",
    )
    parser.add_argument(
        "--prune-max-evals",
        type=int,
        default=30,
        help="Maximum number of evaluations (HyperbandPruner max_resource)",
    )
    parser.add_argument(
        "--reduction-factor",
        type=int,
        default=3,
        help="Reduction factor for HyperbandPruner",
    )
    return parser.parse_args()


def main():
    """Main function to run hyperparameter tuning."""
    # Parse arguments
    args = parse_args()

    logger.info(f"Using base configuration from: {args.config_path}")

    # Load base configuration
    base_config = load_base_config(args.config_path)

    # Create Optuna study with TPE sampler and Hyperband pruner
    sampler = TPESampler(seed=args.seed)
    pruner = HyperbandPruner(
        min_resource=args.prune_min_evals,
        max_resource=args.prune_max_evals,
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

    # Create objective function with base config
    objective = create_objective(base_config)

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
    logger.info(f"Best weighted accuracy: {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")

    # Log individual per-task accuracies for the best performing trial
    # This helps understand which tasks benefited from the optimal hyperparameters
    best_attrs = study.best_trial.user_attrs
    for attr_name, attr_value in sorted(best_attrs.items()):
        if attr_name.startswith("task_") and attr_name.endswith("_accuracy"):
            task_name = attr_name.replace("task_", "").replace("_accuracy", "")
            logger.info(f"  {task_name} accuracy: {attr_value:.4f}")

    # Save complete HPO results to JSON for further analysis
    results_path = Path("hpo_study_results.json")
    with results_path.open("w") as f:
        json.dump(
            {
                "best_trial": study.best_trial.number,
                "best_mean_accuracy": study.best_value,
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
