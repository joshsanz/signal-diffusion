#!/usr/bin/env python3
"""
Find the maximum batch size that can run a short training loop for a config.

Batch sizes are tested in this order: 1, 2, 4, then multiples of 4.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Iterable

try:
    import tomli_w
except ImportError:
    print("ERROR: tomli_w not found. Install with: uv pip install tomli-w", file=sys.stderr)
    sys.exit(1)


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


def extract_steps(output: str) -> int:
    """Extract completed steps from training output."""
    # First try the explicit completion line.
    match = re.search(r"Completed training after (\d+) steps", output)
    if match:
        return int(match.group(1))

    # Fallback to scanning for step numbers in logs.
    matches = re.findall(r"step[:\s]+(\d+)", output, re.IGNORECASE)
    if matches:
        return max(int(m) for m in matches)

    return 0


def update_config_for_test(
    config: dict,
    batch_size: int,
    eval_batch_size: int,
    max_steps: int,
) -> None:
    """Mutate config for a short batch-size test."""
    # Ensure dataset section exists before setting sizes.
    config.setdefault("dataset", {})
    config["dataset"]["batch_size"] = batch_size
    config["dataset"]["eval_batch_size"] = eval_batch_size

    # Trim training to a quick sanity run.
    config.setdefault("training", {})
    config["training"]["max_train_steps"] = max_steps
    config["training"]["epochs"] = 1
    config["training"]["checkpoint_interval"] = max_steps + 1
    config["training"]["eval_strategy"] = "no"

    # Disable common logging to reduce overhead in quick tests.
    logging_cfg = config.get("logging")
    if isinstance(logging_cfg, dict):
        logging_cfg["tensorboard"] = False
        logging_cfg["wandb"] = False


def run_training(
    module: str,
    config_path: Path,
    output_dir: Path,
    timeout: int | None,
) -> subprocess.CompletedProcess[str]:
    """Run the training module with a config path."""
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        module,
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]

    # Keep logs in-process for parsing and file output.
    return subprocess.run(
        cmd,
        timeout=timeout,
        capture_output=True,
        text=True,
    )


def test_batch_size(
    base_config_path: Path,
    module: str,
    batch_size: int,
    eval_batch_size: int,
    max_steps: int,
    timeout: int | None,
    temp_dir: Path,
) -> bool:
    """Test a single batch size against the given config."""
    with open(base_config_path, "rb") as f:
        config = tomllib.load(f)

    update_config_for_test(config, batch_size, eval_batch_size, max_steps)

    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_config = temp_dir / f"test_batch_{batch_size}.toml"
    log_file = temp_dir / f"test_batch_{batch_size}.txt"

    try:
        with open(temp_config, "wb") as f:
            tomli_w.dump(config, f)

        result = run_training(module, temp_config, temp_dir / "output", timeout)

        with open(log_file, "w") as f:
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n\n=== STDERR ===\n")
            f.write(result.stderr)

        if result.returncode != 0:
            return False

        steps_completed = extract_steps(result.stdout + result.stderr)
        return steps_completed >= max_steps
    except subprocess.TimeoutExpired:
        print(f"  (timed out after {timeout}s)")
        return False
    finally:
        if temp_config.exists():
            temp_config.unlink()


def find_max_batch_size(
    base_config_path: Path,
    module: str,
    batch_sizes: Iterable[int],
    eval_batch_size: int | None,
    max_steps: int,
    timeout: int | None,
) -> int | None:
    """Return the largest batch size that succeeds for the given config."""
    temp_dir = Path("runs/model-size-tests/batch-size")
    max_ok: int | None = None

    for batch_size in batch_sizes:
        resolved_eval_batch = eval_batch_size or batch_size
        print(f"\nTesting batch_size={batch_size}, eval_batch_size={resolved_eval_batch}")

        if test_batch_size(
            base_config_path,
            module,
            batch_size,
            resolved_eval_batch,
            max_steps,
            timeout,
            temp_dir,
        ):
            print("✓ SUCCESS")
            max_ok = batch_size
        else:
            print("✗ FAILED")
            break

    return max_ok


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
        help="Python module to execute for training.",
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
        help="Eval batch size to use (default: same as batch size).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Number of steps required for success (default: 10).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per test in seconds (default: 300).",
    )
    parser.add_argument(
        "--no-timeout",
        action="store_true",
        help="Disable timeout (let tests run indefinitely).",
    )
    args = parser.parse_args()

    timeout = None if args.no_timeout else args.timeout
    batch_sizes = build_batch_sizes(args.max_batch_size)

    print("=" * 80)
    print("BATCH SIZE FINDER")
    print(f"Config: {args.config}")
    print(f"Trainer module: {args.trainer_module}")
    print(f"Batch sizes: {batch_sizes}")
    if timeout:
        print(f"Timeout: {timeout}s per test")
    else:
        print("Timeout: DISABLED")
    print("=" * 80)

    max_ok = find_max_batch_size(
        args.config,
        args.trainer_module,
        batch_sizes,
        args.eval_batch_size,
        args.max_steps,
        timeout,
    )

    print("\n" + "=" * 80)
    if max_ok is None:
        print("RESULT: No batch size succeeded.")
    else:
        print(f"RESULT: max batch_size = {max_ok}")
    print("=" * 80)


if __name__ == "__main__":
    main()
