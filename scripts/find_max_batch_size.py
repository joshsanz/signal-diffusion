#!/usr/bin/env python3
"""
Find the maximum batch size that can run a short training loop for a config.

Batch sizes are restricted to 1, 2, 4, and multiples of 4. The search starts
at batch size 8, expands by doubling or halving to bracket success/failure,
then uses binary search to find the best working size.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import sys
import time
import traceback
from pathlib import Path
import torch

from signal_diffusion.config import read_toml, update_config_for_quick_test, write_toml

START_BATCH_SIZE = 8


def build_batch_sizes(max_batch_size: int) -> list[int]:
    """Return allowed batch sizes up to max_batch_size."""
    if max_batch_size < 1:
        raise ValueError("max_batch_size must be >= 1")

    sizes: list[int] = []
    # Enforce the allowed sequence: 1, 2, 4, then multiples of 4.
    if max_batch_size >= 1:
        sizes.append(1)
    if max_batch_size >= 2:
        sizes.append(2)
    if max_batch_size >= 4:
        sizes.append(4)
    if max_batch_size >= 8:
        sizes.extend(range(8, max_batch_size + 1, 4))

    return sizes


def clamp_start_batch_size(allowed_sizes: list[int]) -> int:
    """Clamp the starting batch size to the closest allowed size <= start."""
    for size in reversed(allowed_sizes):
        if size <= START_BATCH_SIZE:
            return size
    return allowed_sizes[0]


def resolve_train_callable(module_path: str):
    """Resolve the training function from a module path."""
    module = importlib.import_module(module_path)
    train_fn = getattr(module, "train", None)
    if train_fn is None:
        raise ValueError(
            f"Training module '{module_path}' does not expose a 'train' function."
        )
    return train_fn


# update_config_for_test moved to signal_diffusion.config.toml_utils
# as update_config_for_quick_test()


def test_batch_size(
    base_config_path: Path,
    train_fn,
    batch_size: int,
    eval_batch_size: int,
    max_steps: int,
    temp_dir: Path,
) -> bool:
    """Test a single batch size against the given config."""
    config = read_toml(base_config_path)
    update_config_for_quick_test(
        config,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        max_steps=max_steps,
    )

    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_config = temp_dir / f"test_batch_{batch_size}.toml"
    log_file = temp_dir / f"test_batch_{batch_size}.txt"
    output_dir = temp_dir / f"batch_{batch_size}"

    try:
        write_toml(config, temp_config)

        # Invoke the training function directly to catch CUDA OOM errors.
        train_fn(
            config_path=temp_config,
            output_dir=output_dir,
            resume_from_checkpoint=None,
            max_train_steps=max_steps,
        )
        return True
    except torch.cuda.OutOfMemoryError as exc:
        with open(log_file, "w") as f:
            f.write("CUDA OutOfMemoryError\n")
            f.write(repr(exc))
        print("  (CUDA out of memory)")
        return False
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" in message:
            with open(log_file, "w") as f:
                f.write("RuntimeError: out of memory\n")
                f.write("".join(traceback.format_exception(*sys.exc_info())))
            print("  (CUDA out of memory)")
            return False
        raise
    except Exception:
        with open(log_file, "w") as f:
            f.write("".join(traceback.format_exception(*sys.exc_info())))
        raise
    finally:
        if temp_config.exists():
            temp_config.unlink()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def find_max_batch_size(
    base_config_path: Path,
    module: str,
    batch_sizes: list[int],
    eval_batch_size: int | None,
    max_steps: int,
) -> int | None:
    """Return the largest batch size that succeeds for the given config."""
    temp_dir = Path("runs/model-size-tests/batch-size")
    train_fn = resolve_train_callable(module)

    def run_trial(batch_size: int) -> bool:
        attempt_start = time.monotonic()
        print("=" * 80)
        resolved_eval_batch = eval_batch_size or batch_size
        print(
            f"\nTesting batch_size={batch_size}, eval_batch_size={resolved_eval_batch}"
        )
        success = test_batch_size(
            base_config_path,
            train_fn,
            batch_size,
            resolved_eval_batch,
            max_steps,
            temp_dir,
        )
        elapsed = time.monotonic() - attempt_start
        print("✓ SUCCESS" if success else "✗ FAILED")
        print(f"Elapsed: {elapsed:.1f}s")
        print("=" * 80)
        return success

    start_size = clamp_start_batch_size(batch_sizes)
    start_index = batch_sizes.index(start_size)

    print(f"Starting at batch_size={start_size}")
    start_success = run_trial(start_size)

    if start_success:
        low_idx = start_index
        high_idx = None
        current_idx = start_index

        while True:
            next_target = batch_sizes[current_idx] * 2
            next_idx = None
            for i, size in enumerate(batch_sizes):
                if size >= next_target:
                    next_idx = i
                    break
            if next_idx is None or next_idx == current_idx:
                return batch_sizes[current_idx]

            if run_trial(batch_sizes[next_idx]):
                current_idx = next_idx
                low_idx = current_idx
            else:
                high_idx = next_idx
                break

        if high_idx is None:
            return batch_sizes[low_idx]
    else:
        high_idx = start_index
        low_idx = None
        current_idx = start_index

        while current_idx > 0:
            next_target = batch_sizes[current_idx] // 2
            next_idx = None
            for i in range(current_idx - 1, -1, -1):
                if batch_sizes[i] <= next_target:
                    next_idx = i
                    break
            if next_idx is None:
                next_idx = 0

            if next_idx == current_idx:
                break

            if run_trial(batch_sizes[next_idx]):
                low_idx = next_idx
                break

            current_idx = next_idx
            high_idx = current_idx

        if low_idx is None:
            return None

    while high_idx - low_idx > 1:
        mid_idx = (low_idx + high_idx) // 2
        if run_trial(batch_sizes[mid_idx]):
            low_idx = mid_idx
        else:
            high_idx = mid_idx

    return batch_sizes[low_idx]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find the maximum batch size for a model config."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the TOML config file.",
    )
    parser.add_argument(
        "--trainer-module",
        default="signal_diffusion.training.diffusion",
        help="Python module exposing a train(config_path, output_dir, resume_from_checkpoint, max_train_steps) function.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=128,
        help="Largest batch size to try (default: 128).",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Eval batch size to use (default: batch size).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Number of steps required for success (default: 10).",
    )
    args = parser.parse_args()
    batch_sizes = build_batch_sizes(args.max_batch_size)

    print("=" * 80)
    print("BATCH SIZE FINDER")
    print(f"Config: {args.config}")
    print(f"Trainer module: {args.trainer_module}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Start batch size: {START_BATCH_SIZE}")
    print("=" * 80)

    max_ok = find_max_batch_size(
        args.config,
        args.trainer_module,
        batch_sizes,
        args.eval_batch_size,
        args.max_steps,
    )

    print("\n" + "=" * 80)
    if max_ok is None:
        print("RESULT: No batch size succeeded.")
    else:
        print(f"RESULT: max batch_size = {max_ok}")
    print("=" * 80)


if __name__ == "__main__":
    main()
