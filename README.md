# Diffusion for Signals

## Shared Data Layer

Core dataset utilities now live under `signal_diffusion/`. Configuration is
managed via TOML files in `config/`, loaded with
`signal_diffusion.config.load_settings()`. Preprocessors and datasets for the
Math, Parkinsons, SEED, and CHB-MIT EEG collections can be found under
`signal_diffusion/data/`, with backwards-compatible shims remaining in
`data_processing/` for older scripts.

For a detailed overview see [`docs/data_layer.md`](./docs/data_layer.md).

## Riffusion Fine-Tuning
See [`fine_tuning`](./fine_tuning)

Managing and running training jobs is easiest with [SkyPilot](https://skypilot.readthedocs.io/en/latest/index.html), or you can look at the `sky-*.yaml` config files to see setup and run commands.

## EEG Classifier
See [`transformer_classification`](./transformer_classification)
