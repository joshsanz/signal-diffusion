"""Centralized logging setup for the signal_diffusion project."""
import logging
import os
import coloredlogs

# Basic configuration for logging
log_level = "DEBUG" if os.environ.get("DEBUG") and os.environ.get("DEBUG") != "0" else "INFO"
coloredlogs.install(level=log_level, fmt="%(asctime)s %(name)s:%(levelname)s - %(message)s")

def get_logger(name: str) -> logging.Logger:
    """Returns a logger with the given name."""
    return logging.getLogger(name)