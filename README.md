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

4. Install NATTEN for Hourglass Diffusion Models

   ```bash
   uv run ./install-natten.sh
   # rm -rf NATTEN  # source may be removed if installation succeeds
   ```

5. Launch scripts through `uv run` to ensure the environment stays consistent, e.g.

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
2. **Preprocess spectrograms** – run `uv run python scripts/preprocess_spectrograms.py --overwrite` (optionally specify datasets) to materialise spectrograms and per-split metadata beneath each dataset's configured output directory.
3. **Inspect metadata** – verify `{split}-metadata.csv` files under each dataset output (train/val/test folders plus aggregate `metadata.csv`). Each row carries canonical label codes (`gender` as `F`/`M`, `health` as `H`/`PD`, integer `age`) to simplify downstream joins.
4. **Generate balanced meta dataset** – execute `uv run python scripts/gen_weighted_spectrogram_dataset.py --preprocess --overwrite --datasets math parkinsons seed longitudinal --tasks gender health age --output-dir data/weighted-dataset` to duplicate samples according to `MetaSampler` weights; the script materialises per-split directories (e.g. `train/`, `test/`), produces split-specific metadata files plus an aggregate `metadata.csv` containing `gender`, `health`, `age`, and an auto-generated caption, saves weight diagnostics, and writes a README with embedded Hugging Face dataset YAML detailing the preprocessing configuration and component counts.
5. **Point training scripts** – reference the new weighted dataset path in classifier or diffusion configs (update TOML output roots or CLI dataset arguments) before launching experiments.

## Diffusion Training

Launch diffusion experiments with the packaged entry point:

```bash
uv run python -m signal_diffusion.training.diffusion config/diffusion/flowers.toml --output-dir runs/diffusion/flowers
```

`scripts/run_diffusion_training.sh` wraps the invocation, resolves repository paths, and ensures the output directory exists.

## Caption-Conditioned Diffusion

Generate EEG spectrograms conditioned on text descriptions like "an EEG spectrogram of a healthy 73 year old female subject" or "an EEG spectrogram of a 55 year old male subject with parkinsons disease".

### Supported Models

| Model | Caption Support | Text Encoder |
|-------|----------------|--------------|
| **Hourglass** | ✅ Full | DualCLIP (2048D) |
| **LocalMamba** | ✅ Full | DualCLIP (2048D) |
| **Stable Diffusion 3.5** | ✅ Native | DualCLIP (2048D) |
| DiT | ❌ Not supported | - |

### Quick Start

**1. Prepare a caption dataset:**

Your dataset must include a caption column (e.g., "text"):

```python
# Example dataset structure
{
    "image": <PIL.Image>,
    "text": "an EEG spectrogram of a healthy 73 year old female subject"
}
```

**2. Configure caption conditioning:**

```toml
# config/diffusion/caption-example.toml
[dataset]
name = "path/to/dataset"
caption_column = "text"              # Column containing captions

[model]
name = "hourglass"                   # or "localmamba", "stable-diffusion-3.5-medium"
conditioning = "caption"             # Enable caption conditioning

[model.extras]
cfg_dropout = 0.1                    # 10% CFG dropout for guidance training
mapping_cond_dim = 2048              # DualCLIP output dimension
```

**3. Train with captions:**

```bash
uv run python -m signal_diffusion.training.diffusion config/diffusion/caption-example.toml --output-dir runs/caption
```

**4. Sample with prompts:**

```python
from signal_diffusion.diffusion.models.base import registry

adapter = registry.get("hourglass")
samples = adapter.generate_conditional_samples(
    conditioning="an EEG spectrogram of a healthy 73 year old female subject",
    guidance_scale=7.5,
    num_samples=4
)
```

### Architecture

Caption conditioning uses **DualCLIPTextEncoder** which combines two CLIP text encoders from Stable Diffusion 3.5:

```
Input: "healthy EEG signal"
  ↓
CLIP-L → 768D pooler_output
CLIP-G → 1280D text_embeds
  ↓
Concatenate → 2048D caption embedding
  ↓
Condition diffusion model
```

**Classifier-Free Guidance (CFG):**
- During training: 10% of captions randomly zeroed (cfg_dropout)
- During sampling: Use guidance_scale (5.0-7.5 recommended) to strengthen conditioning

### Configuration Tips

- **cfg_dropout**: 0.1 (10%) works well for most cases
- **guidance_scale**: Start with 7.5, adjust based on results
  - Lower (3.0-5.0): More diversity, weaker conditioning
  - Higher (10.0+): Stronger conditioning, less diversity
- **Latent space**: Optional for Hourglass/LocalMamba, native for SD 3.5
- **skip_t5**: Keep true for SD 3.5 to save ~11GB memory

## Classification Training

Classifier runs use the same module-based entry point:

```bash
uv run python -m signal_diffusion.training.classification config/classification/test_gender_health_age.toml --output-dir runs/classification/baseline
```

`scripts/run_classification_training.sh` demonstrates launching multiple datasets with a single template config.

### SWA (Stochastic Weight Averaging)

When `swa_enabled = true`, training appends extra SWA epochs after the base schedule completes:

- SWA epochs = base epochs × `swa_extra_ratio` (default 0.333, ~25% of total)
- The main LR scheduler (linear/cosine) completes during the base epochs only
- SWA phase uses `SWALR` with:
  - Linear anneal to `swa_lr = base_lr * swa_lr_frac` (default 0.25)
  - Anneal over 90% of one epoch's steps, then hold constant
  - Per-batch stepping (same cadence as the base scheduler)
- SWA weights are averaged and saved to `checkpoints/swa.pt`

Example: `epochs = 30` with defaults yields 30 base epochs + 10 SWA epochs = 40 total.

## Multi-Task Classification Example

To train a multi-task classifier on your weighted dataset:

```bash
uv run python -m signal_diffusion.training.classification config/classification/test_gender_health_age.toml --output-dir runs/classification/weighted-dataset
```

This config trains a model to predict:
- **Gender**: Binary classification (male/female)
- **Health**: Binary classification (healthy/Parkinson's)
- **Age**: Regression (continuous age prediction)

The configuration includes task weighting to balance the different loss scales between classification and regression tasks.

## Metrics & Evaluation

Install the metrics extras to access evaluation tooling:

```bash
uv sync --group metrics
uv run python metrics/calculate-metrics.py --help
```

## Utility Scripts

- `scripts/preprocess_spectrograms.py` – builds spectrograms and metadata from configured datasets.
- `scripts/gen_weighted_spectrogram_dataset.py` – produces balanced meta datasets with diagnostic outputs.
- `scripts/train_with_best_hpo.py` – trains classifier models using best hyperparameters from HPO studies.
- `scripts/find_max_model_size.py` – determines maximum model size that fits in available GPU memory.
- `scripts/prep-configs-for-sky.sh` – prepares configuration files for use with SkyPilot on, e.g., Lambda Cloud.
- `scripts/test_configs.sh` – makes sure all model & conditioning combos run without errors.
- `scripts/run_classification_training.sh` – helper for classification smoke tests across datasets.
- `scripts/run_diffusion_training.sh` – convenience wrapper for diffusion training launches.

## Additional Features

### HPO Integration
The codebase includes extensive Hyperparameter Optimization (HPO) support with Optuna:
- `hpo/classification_hpo.py` - HPO framework for classification models
- `hpo_results/` - Stores HPO study results and summaries
- `scripts/train_with_best_hpo.py` - Trains models using best hyperparameters from HPO studies

### SWA Support
Stochastic Weight Averaging is available for improved classifier model performance:
- Configurable SWA epochs and learning rate schedules
- Automatic weight averaging and checkpoint saving
- Integrated with existing training loops

### Time-Series Support
The codebase supports both spectrogram and time-series data formats:
- Configurable `data_type` parameter in settings
- Separate preprocessing scripts for each format
- Compatible with all model architectures

### Multiple Output Types
Support for different output formats:
- **db-only**: spectrogram magnitude in decibels
- **db-iq**: dB magnitude plus linear in-phase/quadrature components
- **db-polar**: dB magnitude plus polar representation with linear magnitude and phase

## Data Peculiarities

File `7_1_20180411.cnt` in the SEED V dataset has a broken header which causes errors in versions of `mne` newer than ~1.6. Something about the number of samples or size of data block is corrupted, breaking data size (bytes) inference or number of samples inference depending on the version.
