# Refactor Plan

After each step, gather changes into a git commit to separate them and enable easier rollbacks.

## 1. Repository Restructuring
- Create a top-level `signal_diffusion/` Python package.
- Subpackages: `data`, `models`, `training`, `inference`, `metrics`, `utils`.
- Keep legacy modules temporarily via shim imports (e.g., `data_processing/math.py` re-exporting new data-layer classes) to avoid breaking notebooks; document removal timeline.

## 2. Configuration & Tooling
- Standardize project settings under `config/` using TOML files (`config/default.toml`, environment overrides like `config/dev.toml`).
- Load config via a lightweight settings module that supports env-variable overrides.
- Migrate dependency/environment management to `uv`:
  - Generate `pyproject.toml`/`uv.lock`.
  - Replace `requirements*.txt` where feasible; keep for legacy until migration completes.
  - Update onboarding docs with `uv` commands.
- Retain existing testing approach (no new test dependencies yet).

## 3. Shared Data Layer Overhaul (Priority)
- Introduce `signal_diffusion/data/base.py` with a `BaseSpectrogramPreprocessor` handling:
  - Root path resolution (from TOML config).
  - Decimation/windowing/STFT param validation.
  - Chunk iteration and spectrogram generation via existing helpers.
  - Metadata emission and train/val/test split creation.
- Place dataset-specific logic in subclasses (`MathPreprocessor`, `ParkinsonsPreprocessor`, `SEEDPreprocessor`, future datasets) that implement hooks for loading raw data, channel maps, labels, captions.
- Co-locate dataset objects (`MathDataset`, `ParkinsonsDataset`, `SEEDDataset`) with preprocessors; make `__getitem__` return dicts supporting configurable targets.
- Define a `LabelSpec`/registry describing available tasks per dataset (gender, health, emotion, etc.); expose discovery utilities for training code.
- Fix existing issues while refactoring:
  - Typo `ages. append` in `data_processing/math.py`.
  - Invalid `self.task` reference in `data_processing/parkinsons.py`.
  - Harden metadata checks and error messages.
- Provide smoke scripts/tests that preprocess a tiny split and load a batch to catch regressions.

## 4. Classifier Scaffolding
- Build a classifier factory that assembles model trunks + dynamic heads based on selected targets from the data registry.
- Update training pipeline to accept task lists, automatically adjust loss weighting, logging, and checkpoint naming.
- Supply task config templates (TOML) under `config/classification/` for common scenarios.
- Document how to register new outputs and run minimal training.

## 5. Diffusion & VAE Alignment
- Extract shared accelerate/diffusers setup into `signal_diffusion/training/diffusion_utils.py`.
- Retire the experimental DoG optimizer (remove code paths and documentation).
- Consolidate dataset prep and inference helpers; ensure scripts consume TOML config paths.
- Refresh `fine_tuning/README.md` to reflect new structure and `uv` workflows.

## 6. Metrics & Evaluation
- Turn `metrics/calculate-metrics.py` into a module + CLI in `signal_diffusion/metrics/cli.py`.
- Support multiple feature extractors via config, optional caching, and structured JSON outputs.
- Document usage examples with new configuration.

## 7. Documentation & Notebooks
- Expand top-level `README.md` with clear project map, `uv` setup, config instructions, and links to module docs.
- Add `docs/` with guides for preprocessing, classifier training, diffusion workflows.
- Relocate notebooks into module-specific `notebooks/` directories, strip outputs, and update them to import from the new package.

## 8. Migration Checklist
- Sequence: data layer refactor → classifier scaffolding → training/metrics alignment → docs/tooling.
- Maintain change log in `REFACTOR.md` as tasks complete.
- Track removal dates for shims/notebooks relying on old paths.

## Progress
- [x] Shared data package scaffolding (`signal_diffusion/data/base.py`)
- [x] Migrated Math, Parkinsons, and SEED datasets to shared data layer
- [x] Added legacy shims and documentation for the new data structure
- [x] uv environment tooling in place (`pyproject.toml`, `uv.lock`, README/docs refresh)
- [x] Multi-task classifier training CLI with TOML-driven configs
- [x] Configurable evaluation scheduling for classifier training (eval_strategy & eval_steps)
- [x] Metrics logging with per-task losses (TensorBoard & W&B)
- [x] Enabled continuous age targets with healthy labels across classifiers and added a gender/health/age smoke config + script
- [x] Removed DoG optimizer artifacts from training and diffusion utilities
- [x] Added structured metrics summary export plus configurable best-checkpoint retention for classifier runs
- [x] Removed legacy optimizer restart tuning from historical EEG notebooks and helpers
- [x] Consolidated Stable Diffusion training helpers into `signal_diffusion/training/diffusion_utils.py` and updated training scripts
- [x] Ported weighted meta-dataset generator into `scripts/`, added per-split outputs + HF dataset card, and documented datasets and weights
## Next Steps
- Run the new smoke script once dataset paths are configured to validate regression logging
