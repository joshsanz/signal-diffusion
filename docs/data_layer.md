# Data Layer Overview

The refactored data layer lives under the `signal_diffusion.data` package and
exposes dataset-specific preprocessors and PyTorch-ready datasets that share a
common scaffold.

Install dependencies with the `classification` group before running the
examples:

```bash
uv sync --group classification
```

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
| SEED | `signal_diffusion.data.seed.SEEDPreprocessor` | `signal_diffusion.data.seed.SEEDDataset` | emotion, gender, seed_condition |

Tasks are looked up through dataset-specific label registries (e.g.
`signal_diffusion.data.math.MATH_LABELS`). They feed into the classifier
scaffolding and the meta-dataset utilities under
`signal_diffusion.data.meta`, which expose `MetaPreprocessor`, `MetaDataset`,
and `MetaSampler` alongside backwards-compatible aliases named
`GeneralPreprocessor`, `GeneralDataset`, and `GeneralSampler`.

When you need a class-balanced dataset materialised on disk, run the utility
script:

```bash
uv run python scripts/gen_weighted_dataset.py --help
```

It computes duplication counts from `MetaSampler` weights, copies the sampled
spectrograms into per-split (`train/`, `test/`, etc.) folders below the
configured `output_root`, emits split-specific metadata files alongside an
aggregate `metadata.csv`, records the configuration, component datasets, and
weights in an auto-generated `README.md`, and embeds a Hugging Face dataset YAML
block at the top of that README for direct publishing.

### Metadata Schema

- **Per-dataset preprocessors** now persist canonical label columns: `gender`
  (`F`/`M`), `health` (`H`/`PD` where applicable), and integer `age`, alongside
  dataset-specific identifiers (e.g. SEED session/trial). Downstream code should
  rely on these normalised codes rather than the historical free-form strings.
- **MetaDataset / weighted exports** prune metadata to the core training fields:
  `file_name`, `gender`, `health`, `age`, `split`, and an auto-generated
  human-readable `caption` summarising the labels (e.g. “a spectrogram image of
  a 62 year old, healthy, male subject”).

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

For classifiers, see [`docs/classifier_scaffolding.md`](./classifier_scaffolding.md).
