# Repository Guidelines

## Agent Behavior

- Prefer to ask the user to approve shell commands (e.g. `git status`) rather than repeatedly run into sandbox guardrails.
- Commit changed code to git after each subtask is completed to keep logically grouped changes separated.
- Before beginning complex tasks, create a detailed plan and present it to the user for approval.
- Ask the user for clarification before making decisions that you are unsure about.
- Prefer using tools to modify/create files over workarounds like `echo` calls in shell commands. This makes it easier for the user to follow along with your actions.

## Project Structure & Module Organization

- `signal_diffusion/`: core package providing configs, dataloaders, models, and training loops.
- `scripts/`: CLI wrappers for preprocessing, dataset weighting, and trainer launches.
- `config/`: TOML configs for datasets, diffusion runs, and classification experiments.
- `common/`: legacy utilities retained while older pipelines migrate into `signal_diffusion/`.
- `metrics/`: evaluation scripts such as `calculate-metrics.py` plus VAE data tooling.
- `eeg_classification/`: historical CNN/transformer trainers and archived checkpoints under `bestmodels/`.
- `vae/`: STFT VAE prototypes (`train_vae.py`, decoder notebooks). Keep exploratory notebooks beside the modules they touch; store derived assets in `tensorboard_logs/` or dataset-specific folders.

## Build, Test, and Development Commands

- `uv venv && source .venv/bin/activate`: create and activate the Python 3.10+ virtualenv managed by `uv`.
- `uv sync [--group metrics|--group dev]`: install base dependencies plus optional extras.
- `uv run python scripts/preprocess_data.py --help`: inspect preprocessing options before generating spectrograms.
- `uv run python -m signal_diffusion.training.diffusion config/diffusion/flowers.toml --output-dir runs/diffusion/flowers`: launch diffusion training.
- `uv run python -m signal_diffusion.training.classification config/classification/test_gender_health_age.toml --output-dir runs/classification/baseline`: train the EEG classifier with the unified entry point.
- `uv run tensorboard --logdir runs`: monitor experiments and metric trends.

## Coding Style & Naming Conventions

Follow PEP 8 with 4-space indentation, group imports as stdlib/third-party/local, and prefer f-strings with explicit type hints in shared utilities. Use snake_case for files, variables, and functions; keep modules focused on a single responsibility. Parameterize data paths via CLI flags or config objects—avoid hard-coded absolute locations. Add comments in function bodies explaining what the code is doing.

## Testing Guidelines

No formal test suite exists yet; run sanity checks with short training loops (e.g., `--max_epochs 1`) to validate pipelines. When adding preprocessing logic, exercise it on a small dataset slice and capture observations in logs or TensorBoard. Document manual validation steps alongside code changes.

## Commit & Pull Request Guidelines

Use short, lower-case, imperative commit messages (e.g., `dataset utils`). Reference affected datasets/configs and record exact command lines executed. Pull requests should explain experiment intent, link relevant issues, note external requirements (SkyPilot, CUDA versions), and include representative plots or screenshots when available.

## Data & Access Notes

Keep large datasets outside the repo (e.g., `/data/shared/signal-diffusion/...`) and expose locations via environment variables. Do not commit generated spectrograms, TensorBoard logs, or checkpoints; update `.gitignore` when new artifacts appear.
