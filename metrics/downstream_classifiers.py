"""Train downstream classifiers on real/synthetic data and evaluate cross-splits."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from signal_diffusion.classification import (
    ClassifierConfig,
    build_classifier,
    build_dataset,
    build_task_specs,
)
from signal_diffusion.classification.tasks import available_tasks
from signal_diffusion.config import load_settings
from signal_diffusion.log_setup import get_logger
from signal_diffusion.training.classification import (
    ClassificationExperimentConfig,
    load_experiment_config,
    train_from_config,
)


LOGGER = get_logger(__name__)

SPEC_TYPES = ("db-only", "db-polar", "db-iq", "timeseries")

SPEC_TO_CONFIG = {
    "db-only": "config/classification/baseline.toml",
    "db-polar": "config/classification/baseline-db-polar.toml",
    "db-iq": "config/classification/baseline-db-iq.toml",
    "timeseries": "config/classification/baseline-timeseries.toml",
}

REQUIRED_TASKS = ("gender", "health", "age")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate downstream classifiers on real/synthetic datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-type",
        required=True,
        choices=SPEC_TYPES,
        help="Input data type (db-only, db-iq, db-polar, timeseries)",
    )
    parser.add_argument("--real-dataset", required=True, help="Path to real dataset (HF load_dataset dir)")
    parser.add_argument("--synthetic-dataset", required=True, help="Path to synthetic dataset (HF load_dataset dir)")
    parser.add_argument(
        "--hpo-summary",
        required=True,
        help="Path to HPO summary JSON (e.g., hpo_results/hpo_summary.json)",
    )
    parser.add_argument("--output", required=True, help="Output JSON filename for results")
    return parser.parse_args()


def _load_hpo_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _select_hpo_entry(summary: Mapping[str, Any], spec_type: str) -> tuple[str, dict[str, Any]]:
    # Prefer the run with the best gender accuracy when multiple task objectives exist.
    candidates = {
        name: value for name, value in summary.items() if name.startswith(f"{spec_type}_")
    }
    if not candidates:
        raise ValueError(f"No HPO entries found for spec_type '{spec_type}' in summary")

    def _gender_accuracy(entry: Mapping[str, Any]) -> float:
        metrics = entry.get("task_metrics", {})
        gender = metrics.get("gender", {}) if isinstance(metrics, Mapping) else {}
        value = gender.get("accuracy") if isinstance(gender, Mapping) else None
        return float(value) if value is not None else float("-inf")

    best_name = max(candidates.keys(), key=lambda name: _gender_accuracy(candidates[name]))
    best_entry = candidates[best_name]
    best_acc = _gender_accuracy(best_entry)
    if best_acc == float("-inf"):
        LOGGER.warning(
            "Gender accuracy missing for all HPO entries in %s; using %s based on ordering",
            spec_type,
            best_name,
        )
    return best_name, best_entry


def _apply_hpo_params(config: ClassificationExperimentConfig, hpo_params: Mapping[str, Any]) -> None:
    # Apply HPO parameters onto the loaded base config in-place.
    if "learning_rate" in hpo_params:
        config.optimizer.learning_rate = float(hpo_params["learning_rate"])
    if "weight_decay" in hpo_params:
        config.optimizer.weight_decay = float(hpo_params["weight_decay"])
    if "scheduler" in hpo_params:
        config.scheduler.name = str(hpo_params["scheduler"])
    if "dropout" in hpo_params:
        config.model.dropout = float(hpo_params["dropout"])
    if "depth" in hpo_params:
        config.model.depth = int(hpo_params["depth"])
    if "layer_repeats" in hpo_params:
        config.model.layer_repeats = int(hpo_params["layer_repeats"])
    if "embedding_dim" in hpo_params:
        config.model.embedding_dim = int(hpo_params["embedding_dim"])
    if "batch_size" in hpo_params:
        config.dataset.batch_size = int(hpo_params["batch_size"])


def _configure_dataset(
    config: ClassificationExperimentConfig,
    *,
    dataset_path: Path,
    train_split: str,
    val_split: str,
    tasks: tuple[str, ...],
) -> None:
    # Keep dataset settings aligned with the requested HF dataset path and splits.
    config.dataset.name = str(dataset_path)
    config.dataset.train_split = train_split
    config.dataset.val_split = val_split
    config.dataset.tasks = tasks


def _apply_data_overrides(config: ClassificationExperimentConfig):
    # Mirror the training loop behavior for settings overrides.
    settings = load_settings(config.settings_path)
    if config.data_overrides:
        if "output_type" in config.data_overrides:
            settings.output_type = str(config.data_overrides["output_type"])
        if "data_type" in config.data_overrides:
            settings.data_type = str(config.data_overrides["data_type"])
    return settings


def _prepare_run_config(
    base_config: ClassificationExperimentConfig,
    *,
    dataset_path: Path,
    train_split: str,
    val_split: str,
    tasks: tuple[str, ...],
    run_label: str,
    runs_root: Path,
) -> ClassificationExperimentConfig:
    config = copy.deepcopy(base_config)
    _configure_dataset(
        config,
        dataset_path=dataset_path,
        train_split=train_split,
        val_split=val_split,
        tasks=tasks,
    )
    # Route all run artifacts under a shared output directory for traceability.
    config.training.output_dir = runs_root
    config.training.run_name = run_label
    return config


def _build_classifier_from_config(config: ClassificationExperimentConfig) -> ClassifierConfig:
    task_specs = build_task_specs(config.dataset.name, config.dataset.tasks)
    return ClassifierConfig(
        backbone=config.model.backbone,
        input_channels=config.model.input_channels,
        tasks=task_specs,
        embedding_dim=config.model.embedding_dim,
        dropout=config.model.dropout,
        activation=config.model.activation,
        depth=config.model.depth,
        layer_repeats=config.model.layer_repeats,
        extras=config.model.extras,
    )


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model(checkpoint_path: Path, classifier_config: ClassifierConfig) -> torch.nn.Module:
    device = _select_device()
    model = build_classifier(classifier_config)
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _build_loader(
    settings,
    *,
    dataset_name: str,
    split: str,
    tasks: tuple[str, ...],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    extras: dict[str, Any],
    shuffle: bool,
) -> DataLoader:
    dataset = build_dataset(
        settings,
        dataset_name=dataset_name,
        split=split,
        tasks=tasks,
        target_format="dict",
        extras=extras,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def _evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    task_specs: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    # Compute metrics and confusion matrices in a single pass over the dataset.
    classification_stats = {
        name: {
            "correct": 0,
            "total": 0,
            "confusion": np.zeros((spec.output_dim, spec.output_dim), dtype=np.int64),
        }
        for name, spec in task_specs.items()
        if spec.task_type == "classification"
    }
    regression_stats = {
        name: {"abs_error_sum": 0.0, "squared_error_sum": 0.0, "total": 0}
        for name, spec in task_specs.items()
        if spec.task_type == "regression"
    }

    with torch.no_grad():
        for batch in loader:
            if "signal" in batch:
                inputs = batch["signal"].to(device).contiguous()
            elif "image" in batch:
                inputs = batch["image"].to(device).contiguous()
            else:
                raise KeyError("Batch must contain 'signal' or 'image'")

            outputs = model(inputs)
            for name, spec in task_specs.items():
                targets = batch["targets"][name].to(device)
                if spec.task_type == "classification":
                    targets = targets.long()
                    logits = outputs[name]
                    preds = logits.argmax(dim=1)
                    correct = (preds == targets).sum().item()
                    classification_stats[name]["correct"] += int(correct)
                    classification_stats[name]["total"] += int(targets.numel())

                    targets_np = targets.detach().cpu().numpy().astype(np.int64, copy=False)
                    preds_np = preds.detach().cpu().numpy().astype(np.int64, copy=False)
                    np.add.at(classification_stats[name]["confusion"], (targets_np, preds_np), 1)
                else:
                    logits = outputs[name].view(-1)
                    targets = targets.float().view(-1)
                    error = logits - targets
                    regression_stats[name]["abs_error_sum"] += float(torch.abs(error).sum().item())
                    regression_stats[name]["squared_error_sum"] += float((error ** 2).sum().item())
                    regression_stats[name]["total"] += int(targets.numel())

    metrics: dict[str, Any] = {}
    for name, stats in classification_stats.items():
        total = max(stats["total"], 1)
        metrics[name] = {
            "accuracy": stats["correct"] / total,
            "confusion_matrix": stats["confusion"].tolist(),
        }
    for name, stats in regression_stats.items():
        total = max(stats["total"], 1)
        metrics[name] = {
            "mae": stats["abs_error_sum"] / total,
            "mse": stats["squared_error_sum"] / total,
        }
    metrics["num_examples"] = int(
        sum(stats["total"] for stats in classification_stats.values())
        + sum(stats["total"] for stats in regression_stats.values())
    )
    return metrics


def _format_results_table(results: dict[str, Any]) -> str:
    evaluations = results.get("evaluations", {})
    lines = []
    lines.append("\n" + "=" * 120)
    lines.append("DOWNSTREAM CLASSIFIER RESULTS")
    lines.append("=" * 120)
    lines.append(
        f"{'Evaluation':<32} {'Gender Acc':<12} {'Health Acc':<12} {'Age MAE':<12} {'Age MSE':<12}"
    )
    lines.append("-" * 120)

    for name, payload in evaluations.items():
        gender_acc = _safe_metric(payload, "gender", "accuracy")
        health_acc = _safe_metric(payload, "health", "accuracy")
        age_mae = _safe_metric(payload, "age", "mae")
        age_mse = _safe_metric(payload, "age", "mse")
        lines.append(
            f"{name:<32} {gender_acc:<12} {health_acc:<12} {age_mae:<12} {age_mse:<12}"
        )

    lines.append("=" * 120)
    lines.append("Confusion matrices (rows=true, cols=pred):")
    for name, payload in evaluations.items():
        for task in ("gender", "health"):
            matrix = payload.get(task, {}).get("confusion_matrix")
            if matrix is not None:
                lines.append(f"  {name} {task}: {matrix}")
    lines.append("=" * 120)
    return "\n".join(lines)


def _safe_metric(payload: Mapping[str, Any], task: str, key: str) -> str:
    value = payload.get(task, {}).get(key)
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _serialize_training_config(training) -> dict[str, Any]:
    data = asdict(training)
    for key in ("output_dir", "log_dir", "metrics_summary_path"):
        if data.get(key) is not None:
            data[key] = str(data[key])
    if data.get("wandb_tags") is not None:
        data["wandb_tags"] = list(data["wandb_tags"])
    return data


def print_results_tables(results_path: Path) -> None:
    with results_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    print(_format_results_table(data))


def main(args: argparse.Namespace) -> None:
    spec_type = args.data_type
    real_path = Path(args.real_dataset).expanduser().resolve()
    synth_path = Path(args.synthetic_dataset).expanduser().resolve()
    hpo_path = Path(args.hpo_summary).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    hpo_summary = _load_hpo_summary(hpo_path)
    hpo_key, hpo_entry = _select_hpo_entry(hpo_summary, spec_type)

    base_config_path = Path(SPEC_TO_CONFIG[spec_type]).expanduser().resolve()
    base_config = load_experiment_config(base_config_path)
    _apply_hpo_params(base_config, hpo_entry.get("best_params", {}))

    available = set(available_tasks(str(real_path)))
    missing = [task for task in REQUIRED_TASKS if task not in available]
    if missing:
        raise ValueError(f"Missing required tasks {missing} in dataset {real_path}")
    tasks = tuple(REQUIRED_TASKS)

    runs_root = output_path.parent / "downstream_runs" / spec_type
    runs_root.mkdir(parents=True, exist_ok=True)

    real_train_config = _prepare_run_config(
        base_config,
        dataset_path=real_path,
        train_split="train",
        val_split=base_config.dataset.val_split,
        tasks=tasks,
        run_label=f"downstream_real_{spec_type}",
        runs_root=runs_root,
    )
    synth_train_config = _prepare_run_config(
        base_config,
        dataset_path=synth_path,
        train_split="train",
        val_split="train",
        tasks=tasks,
        run_label=f"downstream_synth_{spec_type}",
        runs_root=runs_root,
    )

    LOGGER.info("Training classifier on real data...")
    real_summary = train_from_config(real_train_config)
    LOGGER.info("Training classifier on synthetic data...")
    synth_summary = train_from_config(synth_train_config)

    settings = _apply_data_overrides(base_config)
    device = _select_device()
    task_specs = {spec.name: spec for spec in build_task_specs(str(real_path), tasks)}

    real_classifier_config = _build_classifier_from_config(real_train_config)
    synth_classifier_config = _build_classifier_from_config(synth_train_config)

    real_model = _load_model(real_summary.best_checkpoint, real_classifier_config)
    synth_model = _load_model(synth_summary.best_checkpoint, synth_classifier_config)

    extras = dict(real_train_config.dataset.extras)
    pin_memory = real_train_config.dataset.pin_memory and torch.cuda.is_available()

    real_test_loader = _build_loader(
        settings,
        dataset_name=str(real_path),
        split="test",
        tasks=tasks,
        batch_size=real_train_config.dataset.batch_size,
        num_workers=real_train_config.dataset.num_workers,
        pin_memory=pin_memory,
        extras=extras,
        shuffle=False,
    )
    synth_train_loader = _build_loader(
        settings,
        dataset_name=str(synth_path),
        split="train",
        tasks=tasks,
        batch_size=synth_train_config.dataset.batch_size,
        num_workers=synth_train_config.dataset.num_workers,
        pin_memory=pin_memory,
        extras=extras,
        shuffle=False,
    )

    results = {
        "meta": {
            "data_type": spec_type,
            "real_dataset": str(real_path),
            "synthetic_dataset": str(synth_path),
            "hpo_summary": str(hpo_path),
            "hpo_selection": {
                "config_name": hpo_key,
                "gender_accuracy": hpo_entry.get("task_metrics", {}).get("gender", {}).get("accuracy"),
            },
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "base_config": str(base_config_path),
            "training": {
                "real": _serialize_training_config(real_train_config.training),
                "synthetic": _serialize_training_config(synth_train_config.training),
            },
        },
        "evaluations": {},
    }

    LOGGER.info("Evaluating real-trained classifier on real test split...")
    results["evaluations"]["real_train__real_test"] = _evaluate_model(
        real_model,
        real_test_loader,
        task_specs,
        device,
    )

    LOGGER.info("Evaluating synthetic-trained classifier on real test split...")
    results["evaluations"]["synth_train__real_test"] = _evaluate_model(
        synth_model,
        real_test_loader,
        task_specs,
        device,
    )

    LOGGER.info("Evaluating real-trained classifier on synthetic train split...")
    results["evaluations"]["real_train__synth_train"] = _evaluate_model(
        real_model,
        synth_train_loader,
        task_specs,
        device,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    LOGGER.info("Saved downstream classifier results to %s", output_path)

    print_results_tables(output_path)


if __name__ == "__main__":
    main(parse_args())
