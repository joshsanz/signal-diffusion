#!/usr/bin/env python3
"""
Find the largest LocalMamba and Hourglass model configurations that can
successfully train for at least 10 steps with specified batch sizes.

This script systematically tests model configurations from largest to smallest
and reports the first (largest) working configuration for each model type.
By default, it uses batch_size=1 and eval_batch_size=1, but these can be
customized via command line arguments to find max model sizes for different
batch configurations.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Iterator

try:
    import tomli_w
except ImportError:
    print("ERROR: tomli_w not found. Install with: uv pip install tomli-w", file=sys.stderr)
    sys.exit(1)


def generate_localmamba_configs() -> Iterator[dict]:
    """Generate LocalMamba configs from largest to smallest.

    Strategy:
    1. Vary depths with base dims/mlp_ratio
    2. Try reduced mlp_ratio (2/3.0 instead of 4.0)
    3. Reduce dims to smaller multiples of 64
    """
    base_dims = [96, 192, 384, 768]

    # Strategy 2: Try reduced mlp_ratio
    for mlp_ratio in [4.0, 3.0, 2.0]:
        # Strategy 1: Vary depths with base dims/mlp_ratio
        for depths in [
            [2, 2, 9, 2],
            [2, 2, 7, 2],
            [2, 2, 5, 2],
            [2, 2, 4, 2],
            [2, 2, 3, 2],
            [2, 2, 2, 2],
            [2, 2, 2],  # Drop last level
        ]:
            yield {"depths": depths, "dims": base_dims[: len(depths)], "mlp_ratio": mlp_ratio}

    # Strategy 3: Reduce dims (multiples of 64) with small depths and reduced mlp ratio
    for dims_scale in [[64, 128, 256, 512], [64, 128, 256, 384], [64, 128, 192, 256]]:
        for depths in [[2, 2, 2, 2], [2, 2, 2]]:
            yield {"depths": depths, "dims": dims_scale[: len(depths)], "mlp_ratio": 3.0}


def generate_hourglass_configs() -> Iterator[dict]:
    """Generate Hourglass configs from largest to smallest.

    Strategy:
    1. Reduce depths (early levels first)
    2. Reduce widths to smaller multiples of 64
    """
    base_widths = [128, 256, 512]

    # Strategy 1: Reduce depths (early levels first)
    for depths in [
        [4, 4, 4],
        [3, 4, 4],
        [2, 4, 4],
        [2, 3, 4],
        [2, 2, 4],
        [2, 2, 3],
        [2, 2, 2],
    ]:
        yield {"depths": depths, "widths": base_widths}

    # Strategy 2: Reduce widths (multiples of 64)
    for widths in [[64, 128, 256], [64, 128, 192], [64, 96, 128]]:
        for depths in [[4, 4, 4], [3, 3, 3], [2, 2, 3], [2, 2, 2]]:
            yield {"depths": depths, "widths": widths}


class ConfigTester:
    """Test model configurations by running training with modified parameters."""

    def __init__(self, base_config_path: Path, model_type: str):
        """Initialize the tester.

        Args:
            base_config_path: Path to the base TOML config file
            model_type: "localmamba" or "hourglass"
        """
        self.base_config_path = base_config_path
        self.model_type = model_type
        self.temp_dir = Path(f"runs/model-size-tests/{model_type}")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def test_config(self, params: dict, batch_size: int = 1, eval_batch_size: int = 1, timeout: int | None = 300) -> bool:
        """Test if a config can train for 10 steps.

        Args:
            params: Model parameters to test (depths, dims/widths, mlp_ratio)
            batch_size: Batch size to use for training (default: 1)
            eval_batch_size: Eval batch size to use for training (default: 1)
            timeout: Timeout in seconds, or None to disable timeout

        Returns:
            True if training succeeded for at least 10 steps
        """
        # 1. Load base TOML
        with open(self.base_config_path, "rb") as f:
            config = tomllib.load(f)

        # 2. Apply modifications
        config["dataset"]["batch_size"] = batch_size
        config["dataset"]["eval_batch_size"] = eval_batch_size
        config["training"]["max_train_steps"] = 10
        config["training"]["epochs"] = 1
        config["training"]["checkpoint_interval"] = 1000
        config["training"]["eval_strategy"] = "no"
        config["training"]["initial_eval"] = False
        config["logging"]["tensorboard"] = False
        config["optimizer"]["name"] = "adamw_8bit"

        # 2b. For LocalMamba: disable latent space and set channels to 3
        if self.model_type == "localmamba":
            config["model"]["latent_space"] = False
            config["model"]["extras"]["in_channels"] = 3
            config["model"]["extras"]["out_channels"] = 3

        # 3. Update model.extras with test parameters
        for key, value in params.items():
            config["model"]["extras"][key] = value

        # 4. Write temp config
        temp_config = self.temp_dir / f"test_config_{hash(str(params))}.toml"
        log_file = self.temp_dir / f"test_log_{hash(str(params))}.txt"

        try:
            with open(temp_config, "wb") as f:
                tomli_w.dump(config, f)

            # 5. Run training
            cmd = [
                "uv",
                "run",
                "python",
                "-m",
                "signal_diffusion.training.diffusion",
                str(temp_config),
                "--output-dir",
                str(self.temp_dir / "output"),
            ]

            result = subprocess.run(
                cmd,
                timeout=timeout,  # None = no timeout
                capture_output=True,
                text=True,
            )

            # Save logs for debugging
            with open(log_file, "w") as f:
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n\n=== STDERR ===\n")
                f.write(result.stderr)

            # 6. Check for success
            success = result.returncode == 0

            # Parse logs to verify steps completed
            if success:
                steps_completed = self._extract_steps(result.stdout + result.stderr)
                success = steps_completed >= 10

            return success

        except subprocess.TimeoutExpired:
            print(f"  (timed out after {timeout}s)")
            return False  # Timeout = failure
        except Exception as e:
            print(f"  Error testing config: {e}")
            return False
        finally:
            # Cleanup temp files
            if temp_config.exists():
                temp_config.unlink()

    def _extract_steps(self, output: str) -> int:
        """Extract completed steps from training output.

        Args:
            output: Combined stdout and stderr from training

        Returns:
            Number of steps completed
        """
        # Look for "Completed training after N steps"
        match = re.search(r"Completed training after (\d+) steps", output)
        if match:
            return int(match.group(1))

        # Alternative: look for step count in logs
        matches = re.findall(r"step[:\s]+(\d+)", output, re.IGNORECASE)
        if matches:
            return max(int(m) for m in matches)

        return 0


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Find largest model configs that can train with specified batch sizes"
    )
    parser.add_argument(
        "--model",
        choices=["localmamba", "hourglass", "both"],
        default="both",
        help="Which model to test (default: both)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per test in seconds (default: 300)",
    )
    parser.add_argument(
        "--no-timeout",
        action="store_true",
        help="Disable timeout (let tests run indefinitely)",
    )
    args = parser.parse_args()

    timeout = None if args.no_timeout else args.timeout

    print("=" * 80)
    print("MODEL SIZE FINDER")
    print(f"Finding largest models that train with batch_size={args.batch_size}, eval_batch_size={args.eval_batch_size}")
    print("Config: db-iq, latent_space=false, adamw_8bit optimizer")
    if timeout:
        print(f"Timeout: {timeout}s per test")
    else:
        print("Timeout: DISABLED")
    print("=" * 80)

    # Test LocalMamba
    localmamba_result = None
    if args.model in ["localmamba", "both"]:
        print("\nTesting LocalMamba configurations...")
        print("-" * 80)

        localmamba_tester = ConfigTester(
            Path("config/diffusion/localmamba-db-iq.toml"), "localmamba"
        )

        for i, params in enumerate(generate_localmamba_configs(), 1):
            attempt_start = time.monotonic()
            print("=" * 80)
            print(f"[{i}] Testing: {params}")
            if localmamba_tester.test_config(params, args.batch_size, args.eval_batch_size, timeout=timeout):
                elapsed = time.monotonic() - attempt_start
                print("✓ SUCCESS - Found largest working config!")
                print(f"Elapsed: {elapsed:.1f}s")
                print("=" * 80)
                localmamba_result = params
                break
            else:
                elapsed = time.monotonic() - attempt_start
                print("✗ FAILED")
                print(f"Elapsed: {elapsed:.1f}s")
                print("=" * 80)

    # Test Hourglass
    hourglass_result = None
    if args.model in ["hourglass", "both"]:
        print("\n\nTesting Hourglass configurations...")
        print("-" * 80)

        hourglass_tester = ConfigTester(
            Path("config/diffusion/hourglass-db-iq.toml"), "hourglass"
        )

        for i, params in enumerate(generate_hourglass_configs(), 1):
            attempt_start = time.monotonic()
            print("=" * 80)
            print(f"[{i}] Testing: {params}")
            if hourglass_tester.test_config(params, args.batch_size, args.eval_batch_size, timeout=timeout):
                elapsed = time.monotonic() - attempt_start
                print("✓ SUCCESS - Found largest working config!")
                print(f"Elapsed: {elapsed:.1f}s")
                print("=" * 80)
                hourglass_result = params
                break
            else:
                elapsed = time.monotonic() - attempt_start
                print("✗ FAILED")
                print(f"Elapsed: {elapsed:.1f}s")
                print("=" * 80)

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    if args.model in ["localmamba", "both"]:
        if localmamba_result:
            print("\nLocalMamba largest config:")
            print(f"  depths: {localmamba_result['depths']}")
            print(f"  dims: {localmamba_result['dims']}")
            print(f"  mlp_ratio: {localmamba_result['mlp_ratio']}")
        else:
            print("\nLocalMamba: No successful configuration found")

    if args.model in ["hourglass", "both"]:
        if hourglass_result:
            print("\nHourglass largest config:")
            print(f"  depths: {hourglass_result['depths']}")
            print(f"  widths: {hourglass_result['widths']}")
        else:
            print("\nHourglass: No successful configuration found")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
