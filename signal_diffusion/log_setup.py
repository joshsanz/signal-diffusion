"""Centralized logging setup for the signal_diffusion project."""
import logging
import os
import coloredlogs

# Basic configuration for logging
log_level = "DEBUG" if os.environ.get("DEBUG") and os.environ.get("DEBUG") != "0" else "INFO"
coloredlogs.install(level=log_level)

logger = logging.getLogger(__name__)
