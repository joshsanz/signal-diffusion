# Data Layer Overview

The refactored data layer lives under the `signal_diffusion.data` package and
exposes dataset-specific preprocessors and PyTorch-ready datasets that share a
common scaffold.

## Configuration

Dataset locations are defined in TOML under `config/`. By default the project
loads `config/default.toml`, but you can override the configuration file by
setting the `SIGNAL_DIFFUSION_CONFIG` environment variable or by passing an
explicit path to `signal_diffusion.config.load_settings()`.

Each dataset entry supplies a `root` directory (raw data) and `output`
(preprocessed spectrograms).

```toml
[data]
root = "/data/shared/signal-diffusion"
output_root = "/data/shared/signal-diffusion/processed"

[datasets.math]
root = "eeg_math"
output = "eeg_math/stfts"
```

## Shared Preprocessor Base

`signal_diffusion.data.base.BaseSpectrogramPreprocessor` orchestrates common
behaviour:

- deriving deterministic train/val/test splits
- handling STFT parameters and output persistence
- writing per-split metadata (`{split}-metadata.csv`) and an aggregated
  `metadata.csv`

Dataset-specific preprocessors subclass the base class and implement
`subjects()` plus `generate_examples()` to yield
`signal_diffusion.data.base.SpectrogramExample` records.

## Available Datasets

| Dataset | Preprocessor | Dataset class | Tasks |
| ------- | ------------ | ------------- | ----- |
| Math | `signal_diffusion.data.math.MathPreprocessor` | `signal_diffusion.data.math.MathDataset` | gender, math_activity, math_condition |
| Parkinsons | `signal_diffusion.data.parkinsons.ParkinsonsPreprocessor` | `signal_diffusion.data.parkinsons.ParkinsonsDataset` | gender, health, parkinsons_condition |
| MIT (CHB-MIT) | `signal_diffusion.data.mit.MITPreprocessor` | `signal_diffusion.data.mit.MITDataset` | gender, seizure, mit_condition |
| SEED | `signal_diffusion.data.seed.SeedPreprocessor` | `signal_diffusion.data.seed.SeedDataset` | emotion, gender, seed_condition |

Tasks are looked up through dataset-specific label registries (e.g.
`signal_diffusion.data.math.MATH_LABELS`). These registries are used by the new
classifier scaffolding and by the updated shims under `data_processing/`.

## Legacy Compatibility

The historical entry points (`data_processing/math.py`,
`data_processing/parkinsons.py`, `data_processing/seed.py`) now wrap the new
implementations. Existing scripts that import those modules continue to receive
Tuples `(tensor, label)` and class-based samplers without modification.

## Getting Started

```python
from signal_diffusion.config import load_settings
from signal_diffusion.data.math import MathPreprocessor, MathDataset

settings = load_settings()  # honours SIGNAL_DIFFUSION_CONFIG
preprocessor = MathPreprocessor(settings, nsamps=2000, fs=125)
preprocessor.preprocess(overwrite=False)

train_ds = MathDataset(settings, split="train", tasks=("math_condition",))
example = train_ds[0]
print(example["targets"])
```

Use the same pattern for the Parkinsons and SEED datasets, adjusting task names
as needed.
