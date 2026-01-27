#!/usr/bin/env python3
"""Profile classification training with the PyTorch profiler."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from signal_diffusion.classification.config import load_classification_config
from signal_diffusion.training.classification import train_from_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile a short classification training run."
    )
    parser.add_argument("config", type=Path, help="Path to classification TOML config.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/profile"),
        help="Output directory for run artifacts.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum training steps to execute.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Optional warmup steps to run before profiling.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override training device (e.g., 'cuda', 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--record-shapes",
        action="store_true",
        help="Record tensor shapes in the profiler output.",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Record memory usage in the profiler output.",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="cuda_time_total",
        help="Sort key for profiler table output.",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=30,
        help="Number of rows to print from profiler table.",
    )
    parser.add_argument(
        "--trace-path",
        type=Path,
        default=None,
        help="Optional path to export a Chrome trace JSON.",
    )
    return parser.parse_args()


def _apply_profile_overrides(
    experiment,
    output_dir: Path,
    max_steps: int,
    device: str | None,
) -> None:
    # Keep runs short and avoid extra validation overhead.
    experiment.training.output_dir = output_dir.resolve()
    experiment.training.max_steps = max_steps
    experiment.training.epochs = 1
    experiment.training.eval_strategy = "none"
    experiment.training.log_every_batches = 0

    # Avoid periodic checkpointing during the profiling window.
    experiment.training.checkpoint_strategy = "steps"
    experiment.training.checkpoint_steps = max_steps + 1

    if device is not None:
        experiment.training.device = device


def main() -> None:
    args = _parse_args()

    experiment = load_classification_config(args.config)
    _apply_profile_overrides(
        experiment,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        device=args.device,
    )

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available() and (args.device is None or "cuda" in args.device):
        activities.append(ProfilerActivity.CUDA)

    if args.warmup_steps > 0:
        warmup_experiment = copy.deepcopy(experiment)
        warmup_dir = args.output_dir / "warmup"
        _apply_profile_overrides(
            warmup_experiment,
            output_dir=warmup_dir,
            max_steps=args.warmup_steps,
            device=args.device,
        )
        train_from_config(warmup_experiment)

    with profile(
        activities=activities,
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
    ) as prof:
        # Run the training loop inside the profiler context.
        train_from_config(experiment)

    table = prof.key_averages().table(
        sort_by=args.sort_by,
        row_limit=args.row_limit,
    )
    print(table)

    if args.trace_path is not None:
        args.trace_path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(args.trace_path))
        print(f"Chrome trace exported to {args.trace_path}")


if __name__ == "__main__":
    main()
