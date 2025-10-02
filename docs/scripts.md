# Scripts and Utilities

This document provides an overview of the scripts and utilities available in the `scripts` and `common` directories.

## `scripts` Directory

The `scripts` directory contains a collection of Python scripts for performing various tasks, such as:

-   **`preprocess_data.py`**: Preprocesses EEG data from the supported datasets.
-   **`run_classification_training.sh`**: A shell script to run classification training experiments.
-   **`run_diffusion_training.sh`**: A shell script to run diffusion training experiments.
-   **`gen_weighted_dataset.py`**: Generates a class-balanced dataset on disk.
-   **`subdataset_weighting_analysis.py`**: A script for analyzing the weighting of subdatasets.

## `common` Directory

The `common` directory contains a collection of Python modules with shared utilities, such as:

-   **`gen_multichannel_stft.py`**: A script for generating multi-channel STFTs.
-   **`multichannel_spectrograms.py`**: A module with utilities for working with multi-channel spectrograms.
-   **`ear_eeg_support_scripts`**: A directory with scripts for supporting in-ear EEG data.
