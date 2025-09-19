# Classifier Scaffolding

The new classifier stack lives under `signal_diffusion.models` and is designed
for multi-task EEG classification across any dataset that exposes a label
registry. It provides:

```bash
uv sync --group classification
```

Run the examples below with `uv run` to ensure the synced environment is used.

- reusable CNN/Transformer backbones that emit shared embeddings
- a `MultiTaskClassifier` wrapper with one head per task
- utilities to translate dataset label registries into classifier task specs
- dataset helpers that return PyTorch-ready datasets with consistent targets

## Building a Classifier

```python
from signal_diffusion.config import load_settings
from signal_diffusion.classification import build_dataset, build_task_specs
from signal_diffusion.models import ClassifierConfig, build_classifier

settings = load_settings()  # honours SIGNAL_DIFFUSION_CONFIG

train_tasks = build_task_specs("parkinsons", ["gender", "health"])
config = ClassifierConfig(
    backbone="cnn_light",
    input_channels=1,
    tasks=train_tasks,
    embedding_dim=256,
    dropout=0.3,
)
model = build_classifier(config)
```

`model(image)` now returns a dictionary of logits keyed by task name, or you can
request a specific head with `model(image, task="gender")`.

## Datasets

Use `signal_diffusion.classification.build_dataset` to construct datasets for a
particular split and task list:

```python
train_ds = build_dataset(
    settings,
    dataset_name="parkinsons",
    split="train",
    tasks=["gender", "health"],
)
```

Samples expose `sample["targets"]` as a dictionary mapping task names to label
indices, making it straightforward to compute multi-task losses.

A starter configuration is provided at
`configs/classification/baseline.toml`. It references the shared TOML settings,
selects the Parkinsons dataset, and trains a lightweight CNN backbone on the
`gender` and `health` tasks.

## Running Training

Launch experiments via the Typer CLI exposed by the training module. The
example below writes checkpoints and metrics into `runs/` by default:

```bash
uv run python -m signal_diffusion.training.classification train configs/classification/baseline.toml --output-dir runs
```

The CLI resolves dataset paths from the referenced TOML settings file, creates
task-specific checkpoints under `runs/<dataset>-<backbone>-<tasks>-<timestamp>/`,
and records per-epoch metrics in `history.json`. Override hyperparameters by
editing the configuration or adding additional files under
`configs/classification/`.


Set `[training.eval_strategy]` to control validation cadence (`epoch`, `steps`, or `none`). When using the `steps` strategy, provide `eval_steps` to specify how many optimizer updates should occur between evaluations.
