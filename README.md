# Diffusion for Signals

The repository houses EEG spectrogram preprocessing, multi-task classifiers, and diffusion experiments. Tooling now centers on a shared `signal_diffusion/` package with TOML-driven configuration and `uv` for environment management.

## Environment Setup

1. Install [`uv`](https://github.com/astral-sh/uv) (`curl -LsSf https://astral.sh/uv/install.sh | sh` or `brew install uv`).
2. Create and activate a project environment:

   ```bash
   uv venv .venv
   source .venv/bin/activate
   ```

3. Sync the core dependencies:

   ```bash
   uv sync
   ```

   Add optional groups as needed:

   ```bash
   uv sync --group classification            # classifiers only
   uv sync --group fine-tuning               # diffusion training stack
   uv sync --group metrics                   # metrics CLI utilities
   uv sync --group fine-tuning-lambda        # lambda-friendly diffusion deps
   uv sync --group notebooks                 # Jupyter + plotting helpers
   ```

4. Run project scripts through `uv` to ensure the lockfile is honoured, e.g.

   ```bash
   uv run python fine_tuning/train_text_to_image.py --help
   uv run python eeg_classification/transformer_classification_2.0.py
   ```

Legacy `requirements*.txt` files remain for pinned deployments that cannot yet move to `uv`.

## Configuration

Project configuration lives in `config/default.toml`. It defines dataset roots and output directories and is consumed by `signal_diffusion.config.load_settings()`:

```bash
export SIGNAL_DIFFUSION_CONFIG=/path/to/custom-config.toml
uv run python -c "from signal_diffusion.config import load_settings; print(load_settings())"
```

Each dataset receives a `root` (raw inputs) and `output` (preprocessed spectrograms). Create a copy of `config/default.toml` for local paths or new datasets.

## Shared Data Layer

Core dataset utilities live in `signal_diffusion/data/`. The shared base handles train/val/test splitting, metadata emission, and spectrogram persistence for:

- Math dataset
- Parkinsons dataset
- SEED dataset
- CHB-MIT (MIT) dataset

See [`docs/data_layer.md`](./docs/data_layer.md) for in-depth guidance on preprocessors, dataset classes, label registries, and the meta-dataset helpers that combine multiple sources.

## Dataset Processing Pipeline

1. **Configure paths** – copy `config/default.toml` and update raw/output roots as needed (`SIGNAL_DIFFUSION_CONFIG` governs which file is loaded).
2. **Preprocess spectrograms** – run `uv run python scripts/preprocess_data.py --overwrite` (optionally specify datasets) to materialise spectrograms and per-split metadata beneath each dataset’s configured output directory.
3. **Inspect metadata** – verify `{split}-metadata.csv` files under each dataset output (train/val/test folders plus aggregate `metadata.csv`). Each row now carries canonical label codes (`gender` as `F`/`M`, `health` as `H`/`PD`, integer `age`) to simplify downstream joins.
4. **Generate balanced meta dataset** – execute `uv run python scripts/gen_weighted_dataset.py --preprocess --overwrite` (adjust flags as needed) to duplicate samples according to `MetaSampler` weights; the script materialises per-split directories (e.g. `train/`, `test/`), produces split-specific metadata files plus an aggregate `metadata.csv` containing `gender`, `health`, `age`, and an auto-generated caption, saves weight diagnostics, and writes a README with embedded Hugging Face dataset YAML detailing the preprocessing configuration and component counts.
5. **Point training scripts** – reference the new weighted dataset path in classifier or diffusion configs (e.g. update TOML output roots or CLI dataset arguments) before launching experiments.

## EEG Classifier Stack

The multi-task classifier factory, model backbones, and dataset helpers live in `signal_diffusion/`.

- [`docs/classifier_scaffolding.md`](./docs/classifier_scaffolding.md) explains how to assemble new models and register additional targets.
- Baseline configs are stored in `config/classification/`.

## Diffusion & Fine-Tuning

Diffusion training scripts remain under `fine_tuning/`. Recommended workflow:

```bash
uv sync --group fine-tuning
uv run accelerate launch fine_tuning/train_text_to_image.py ...
```

SkyPilot configs (`fine_tuning/sky-*.yaml`) provide cloud launch examples.

## Metrics & Evaluation

Metrics tooling is moving into `signal_diffusion/metrics`. Install the metrics group to access the CLI utilities:

```bash
uv sync --group metrics
uv run python metrics/calculate-metrics.py --help
```

## Utility Scripts

- `uv run python scripts/gen_weighted_dataset.py --help` generates a re-weighted meta-dataset on disk by duplicating
  spectrograms according to `MetaSampler` weights, writes per-split diagnostics/metadata (with canonical `gender`, `health`,
  `age`, and caption fields), records the run configuration, and embeds a Hugging Face dataset card YAML at the top of the
  emitted README.

## Data Peculiarities

File `7_1_20180411.cnt` in the SEED V dataset has a broken header which causes errors in versions of `mne` newer than ~1.6. Something about the number of samples or size of data block is corrupted, breaking data size (bytes) inference or number of samples inference depending on the version.
