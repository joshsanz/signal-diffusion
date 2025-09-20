# Repository Guidelines

## Agent Behavior
- Prefer to ask the user to approve shell commands (e.g. `git status`) rather than repeatedly run into sandbox guardrails.
- Commit changed code to git after each subtask is completed to keep logically grouped changes separated.
- Before beginning complex tasks, create a detailed plan and present it to the user for approval. 
- Ask the user for clarification before making decisions that you are unsure about.

## Project Structure & Module Organization
- `fine_tuning/`: diffusion training, LoRA utilities, and launch configs (`run-cmd.sh` for reference commands).
- `eeg_classification/`: EEG CNN/transformer trainers, sweeps, TensorBoard logs, and saved checkpoints under `bestmodels/`.
- `common/` & `data_processing/`: shared preprocessing helpers, spectrogram builders, and metadata utilities.
- `metrics/`: evaluation scripts such as `calculate-metrics.py` plus VAE data tooling.
- `vae/`: STFT VAE prototypes (`train_vae.py`, decoder notebooks). Keep exploratory notebooks beside the modules they touch; store derived assets in `tensorboard_logs/` or dataset-specific folders.

## Build, Test, and Development Commands
- `uv venv && source .venv/bin/activate`: create and activate the Python 3.10+ virtualenv managed by `uv`.
- `uv pip install -r fine_tuning/requirements.txt`: install diffusion and LoRA dependencies (`metrics/requirements.txt` for metrics-only work).
- `uv run accelerate launch fine_tuning/train_text_to_image.py ...`: kick off full diffusion training; copy baseline flags from `run-cmd.sh`.
- `uv run python fine_tuning/infer_text_to_image_lora.py --help`: inspect LoRA inference options before running with custom weights.
- `uv run python eeg_classification/transformer_classification_2.0.py`: train the EEG transformer (ensure dataset paths are parameterized).
- `uv run tensorboard --logdir eeg_classification/tensorboard_logs`: monitor experiments and metric trends.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation, group imports as stdlib/third-party/local, and prefer f-strings with explicit type hints in shared utilities. Use snake_case for files, variables, and functions; keep modules focused on a single responsibility. Parameterize data paths via CLI flags or config objects—avoid hard-coded absolute locations.

## Testing Guidelines
No formal test suite exists yet; run smoke checks with short training loops (e.g., `--max_epochs 1`) to validate pipelines. When adding preprocessing logic, exercise it on a small dataset slice and capture observations in logs or TensorBoard. Document manual validation steps alongside code changes.

## Commit & Pull Request Guidelines
Use short, lower-case, imperative commit messages (e.g., `dataset utils`). Reference affected datasets/configs and record exact command lines executed. Pull requests should explain experiment intent, link relevant issues, note external requirements (SkyPilot, CUDA versions), and include representative plots or screenshots when available.

## Data & Access Notes
Keep large datasets outside the repo (e.g., `/data/shared/signal-diffusion/...`) and expose locations via environment variables. Do not commit generated spectrograms, TensorBoard logs, or checkpoints; update `.gitignore` when new artifacts appear.
