# Configuration System

The Signal Diffusion project uses a flexible configuration system based on TOML files. This document explains how the configuration system works and how to use it.

## Configuration Files

The main configuration files are located in the `config` directory. The project uses a default configuration file named `default.toml`, which can be overridden by setting the `SIGNAL_DIFFUSION_CONFIG` environment variable.

### `default.toml`

This file contains the default configuration for the project, including data paths, model parameters, and training settings. You can create a `default-local.toml` to override the default settings for your local environment.

### Experiment-specific Configurations

The `config` directory also contains subdirectories for different types of experiments, such as `classification` and `diffusion`. These directories contain TOML files with specific configurations for each experiment.

## Loading Configuration

The configuration is loaded using the `signal_diffusion.config.load_settings()` function. This function loads the default configuration and merges it with any experiment-specific configuration files.

## Configuration Options

The following are the main configuration sections and options available:

-   **`[data]`**: Contains settings related to data paths and preprocessing.
-   **`[datasets]`**: Contains settings for specific datasets, such as `math`, `parkinsons`, and `seed`.
-   **`[model]`**: Contains settings for the model architecture, such as the backbone, number of channels, and embedding dimension.
-   **`[training]`**: Contains settings for the training process, such as the number of epochs, batch size, learning rate, and optimizer.
-   **`[logging]`**: Contains settings for logging, such as TensorBoard and Weights & Biases.
-   **`[inference]`**: Contains settings for the inference process, such as the number of denoising steps and the CFG scale.
