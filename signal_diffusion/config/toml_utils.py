"""TOML configuration file reading and writing utilities.

Provides functions for reading TOML files using tomllib (Python 3.11+)
and writing TOML files using tomli_w.
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from typing import Any

try:
    import tomli_w
except ImportError:
    print(
        "ERROR: tomli_w not found. Install with: uv pip install tomli-w",
        file=sys.stderr,
    )
    sys.exit(1)


def read_toml(path: Path | str) -> dict[str, Any]:
    """Read a TOML file and return its contents as a dictionary.

    Args:
        path: Path to the TOML file

    Returns:
        Dictionary containing the TOML file contents

    Raises:
        FileNotFoundError: If the file doesn't exist
        tomllib.TOMLDecodeError: If the file is not valid TOML

    Examples:
        >>> config = read_toml("config/default.toml")
        >>> config["data"]["root"]
        '/data/data/signal-diffusion'
    """
    path = Path(path)
    with path.open("rb") as f:
        return tomllib.load(f)


def write_toml(config: dict[str, Any], path: Path | str) -> None:
    """Write a dictionary to a TOML file.

    Note: This will not preserve comments or formatting from the original file.
    For structure-preserving edits, use scripts/edit_config.py instead.

    Args:
        config: Dictionary to write as TOML
        path: Path to write the TOML file to

    Examples:
        >>> config = {"data": {"root": "/path/to/data"}}
        >>> write_toml(config, "config/test.toml")
    """
    path = Path(path)
    with path.open("wb") as f:
        tomli_w.dump(config, f)


def update_config_for_quick_test(
    config: dict[str, Any],
    *,
    batch_size: int | None = None,
    eval_batch_size: int | None = None,
    max_steps: int = 10,
    disable_logging: bool = True,
) -> dict[str, Any]:
    """Update a config dictionary for quick training tests.

    Modifies the config in-place and returns it for convenience.
    Useful for batch size testing or model size finding.

    Args:
        config: Configuration dictionary to modify
        batch_size: Training batch size (if None, leaves unchanged)
        eval_batch_size: Eval batch size (if None, uses batch_size or leaves unchanged)
        max_steps: Maximum training steps (default: 10)
        disable_logging: Whether to disable TensorBoard and wandb (default: True)

    Returns:
        The modified config dictionary (same object as input)

    Examples:
        >>> config = read_toml("config/diffusion/baseline.toml")
        >>> update_config_for_quick_test(config, batch_size=4, max_steps=10)
        >>> write_toml(config, "config/test.toml")
    """
    # Update dataset settings
    config.setdefault("dataset", {})
    if batch_size is not None:
        config["dataset"]["batch_size"] = batch_size
        if eval_batch_size is None:
            eval_batch_size = batch_size
    if eval_batch_size is not None:
        config["dataset"]["eval_batch_size"] = eval_batch_size

    # Update training settings for quick test
    config.setdefault("training", {})
    config["training"]["max_train_steps"] = max_steps
    config["training"]["epochs"] = 1
    config["training"]["checkpoint_interval"] = max_steps + 1
    config["training"]["eval_strategy"] = "no"
    config["training"]["initial_eval"] = False
    if eval_batch_size is not None:
        config["training"]["eval_batch_size"] = eval_batch_size

    # Disable logging if requested
    if disable_logging:
        logging_cfg = config.get("logging")
        if isinstance(logging_cfg, dict):
            logging_cfg["tensorboard"] = False
            logging_cfg.pop("wandb_project", None)
            logging_cfg.pop("wandb_entity", None)

    return config
