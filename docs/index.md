# Signal Diffusion Project Overview

This document provides a high-level overview of the Signal Diffusion project, a Python-based research project focused on EEG signal processing.

## Project Goals

The main goal of this project is to explore the use of diffusion models for generating synthetic EEG data. This includes:

-   Preprocessing real EEG data from various datasets.
-   Training and evaluating multi-task EEG classifiers.
-   Training and evaluating diffusion models for generating synthetic EEG spectrograms.
-   Evaluating the quality of the generated spectrograms.

## Project Structure

The project is organized into the following main components:

-   **`signal_diffusion`**: The core Python package containing the main logic for data processing, classification, and diffusion models.
-   **`eeg_classification`**: Contains notebooks and scripts for training and evaluating EEG classifiers.
-   **`fine_tuning`**: Contains scripts for training text-to-image diffusion models.
-   **`metrics`**: Provides tools for evaluating the quality of generated spectrograms.
-   **`config`**: Contains TOML configuration files for different experiments.
-   **`scripts`**: Contains various utility scripts.
-   **`docs`**: Contains project documentation.

## Getting Started

To get started with the project, please refer to the following documents:

-   **`docs/data_layer.md`**: Explains how to preprocess and work with EEG datasets.
-   **`docs/classifier_scaffolding.md`**: Explains how to build, train, and evaluate EEG classifiers.
-   **`docs/diffusion-training.md`**: Explains how to train diffusion models.
