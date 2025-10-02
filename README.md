# Diffusion for Signals

The repository houses EEG spectrogram preprocessing, multi-task classifiers, and diffusion experiments. Tooling is now concentrated in the `signal_diffusion/` package with configuration handled through TOML files and all entry points runnable via `uv`.

## Environment Setup

1. Install [`uv`](https://github.com/astral-sh/uv) (`curl -LsSf https://astral.sh/uv/install.sh | sh` or `brew install uv`).
2. Create and activate a project environment:

   ```bash
   uv venv .venv
   source .venv/bin/activate
   ```

3. Sync dependencies from `pyproject.toml`:

   ```bash
   uv sync
   ```

   Optional extras:

   ```bash
   uv sync --group metrics      # evaluation-only utilities
   uv sync --group dev          # linting/formatting helpers
   ```

4. Launch scripts through `uv run` to ensure the environment stays consistent, e.g.

   ```bash
   uv run python -m signal_diffusion.training.diffusion config/diffusion/flowers.toml --output-dir runs/diffusion/flowers
   uv run python -m signal_diffusion.training.classification config/classification/test_gender_health_age.toml --output-dir runs/classification/baseline
   ```

## Project Layout

- `signal_diffusion/`: unified package for configs, data loaders, models, and training loops.
- `scripts/`: thin CLI wrappers for preprocessing, dataset weighting, and training launches.
- `config/`: TOML config files for datasets, diffusion runs, and classifier sweeps.
- `common/`: legacy utilities kept for backwards compatibility during the transition.
- `eeg_classification/`: archived standalone trainers plus historical TensorBoard runs.
- `metrics/`: evaluation scripts such as `calculate-metrics.py` and supporting helpers.
- `vae/`: STFT VAE prototypes, notebooks, and tensorboard logs.

## Configuration

Project configuration lives in `config/default.toml`. It defines dataset roots and output directories and is consumed by `signal_diffusion.config.load_settings()`:

```bash
export SIGNAL_DIFFUSION_CONFIG=/path/to/custom-config.toml
uv run python -c "from signal_diffusion.config import load_settings; print(load_settings())"
```

Each dataset receives a `root` (raw inputs) and `output` (preprocessed spectrograms). Create a copy of `config/default.toml` for local paths or new datasets.

## Dataset Processing Pipeline

1. **Configure paths** – copy `config/default.toml` and update raw/output roots as needed (`SIGNAL_DIFFUSION_CONFIG` governs which file is loaded).
2. **Preprocess spectrograms** – run `uv run python scripts/preprocess_data.py --overwrite` (optionally specify datasets) to materialise spectrograms and per-split metadata beneath each dataset’s configured output directory.
3. **Inspect metadata** – verify `{split}-metadata.csv` files under each dataset output (train/val/test folders plus aggregate `metadata.csv`). Each row carries canonical label codes (`gender` as `F`/`M`, `health` as `H`/`PD`, integer `age`) to simplify downstream joins.
4. **Generate balanced meta dataset** – execute `uv run python scripts/gen_weighted_dataset.py --preprocess --overwrite` (adjust flags as needed) to duplicate samples according to `MetaSampler` weights; the script materialises per-split directories (e.g. `train/`, `test/`), produces split-specific metadata files plus an aggregate `metadata.csv` containing `gender`, `health`, `age`, and an auto-generated caption, saves weight diagnostics, and writes a README with embedded Hugging Face dataset YAML detailing the preprocessing configuration and component counts.
5. **Point training scripts** – reference the new weighted dataset path in classifier or diffusion configs (update TOML output roots or CLI dataset arguments) before launching experiments.

## Diffusion Training

Launch diffusion experiments with the packaged entry point:

```bash
uv run python -m signal_diffusion.training.diffusion config/diffusion/flowers.toml --output-dir runs/diffusion/flowers
```

`scripts/run_diffusion_training.sh` wraps the invocation, resolves repository paths, and ensures the output directory exists.

## Classification Training

Classifier runs use the same module-based entry point:

```bash
uv run python -m signal_diffusion.training.classification config/classification/test_gender_health_age.toml --output-dir runs/classification/baseline
```

`scripts/run_classification_training.sh` demonstrates launching multiple datasets with a single template config.

## Metrics & Evaluation

Install the metrics extras to access evaluation tooling:

```bash
uv sync --group metrics
uv run python metrics/calculate-metrics.py --help
```

## Utility Scripts

- `scripts/preprocess_data.py` – builds spectrograms and metadata from configured datasets.
- `scripts/gen_weighted_dataset.py` – produces balanced meta datasets with diagnostic outputs.
- `scripts/run_classification_training.sh` – helper for classification smoke tests across datasets.
- `scripts/run_diffusion_training.sh` – convenience wrapper for diffusion training launches.

## Data Peculiarities

File `7_1_20180411.cnt` in the SEED V dataset has a broken header which causes errors in versions of `mne` newer than ~1.6. Something about the number of samples or size of data block is corrupted, breaking data size (bytes) inference or number of samples inference depending on the version.

