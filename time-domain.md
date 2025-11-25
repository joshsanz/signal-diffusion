# Time-Domain EEG Implementation Plan - Revised

## Executive Summary

This plan implements time-domain EEG signal support alongside existing spectrogram processing. The revised approach addresses critical architectural issues discovered during codebase analysis:

1. **Normalization strategy** - Add per-channel z-score normalization with precomputed statistics
2. **Configuration location** - Use `data.data_type` (not model-level) since dimensionality is a dataset property
3. **Diffusion adapter shapes** - Dynamic noise tensor shapes based on data_type
4. **Transform pipeline** - Conditional branching for 1D vs 2D data
5. **Custom writers** - Leverage existing `SpectrogramExample.writer` mechanism for .npy files

## Design Principles

- **Backward compatibility**: All existing spectrogram workflows continue unchanged
- **Minimal duplication**: Reuse base classes, registries, and metadata systems
- **Type safety**: Validate data_type/model compatibility at config load time
- **Pragmatic inheritance**: Keep `BaseSpectrogramPreprocessor` despite naming mismatch
- **Centralized transforms**: Create new `signal_diffusion/data/transforms/` module
- **Separate storage**: Time-series data stored in `seed/timeseries/` (separate from spectrograms)
- **Clear semantics**: Use `'signal'` key (not `'image'`) in dataset returns for time-series
- **Asymmetric patching**: Add channel dimension (B, C, H, W) = (B, 1, n_channels, n_samples), then use asymmetric patches (1×8 instead of 2×2) for high aspect ratio signals

---

## Phase 1: Configuration System

### 1.1 Update Settings Class

**File**: `signal_diffusion/config/settings.py`

Add `data_type` field to `Settings` dataclass:

```python
@dataclass
class Settings:
    data_root: Path
    output_root: Path
    datasets: dict[str, DatasetSettings]
    max_sampling_weight: float = 5.0
    data_type: str = "spectrogram"  # NEW: "spectrogram" or "timeseries"

    def __post_init__(self):
        """Validate configuration."""
        if self.data_type not in {"spectrogram", "timeseries"}:
            raise ValueError(f"data_type must be 'spectrogram' or 'timeseries', got {self.data_type}")
```

### 1.2 Update Configuration Files

**Files**: `config/default.toml`, `config/classification/*.toml`, `config/diffusion/*.toml`

Add to `[data]` section in `default.toml`:
```toml
[data]
root = "/data/data/signal-diffusion"
output_root = "/data/data/signal-diffusion/processed"
max_sampling_weight = 5.0
data_type = "spectrogram"  # NEW: Default to existing behavior
```

Update dataset output paths to support separate directories:
```toml
[datasets.seed]
root = "seed"
output = "seed/stfts"           # Spectrograms
timeseries_output = "seed/timeseries"  # NEW: Time-series data
```

Create example time-domain configs:
- `config/classification/seed-timeseries.toml`
- `config/diffusion/dit-timeseries.toml`

Each should set `data_type = "timeseries"` in `[data]` section and use appropriate dataset names (e.g., `"seed_timeseries"`).

### 1.3 Add Validation

**File**: `signal_diffusion/diffusion/config.py`

Add validation method to `DiffusionConfig`:

```python
def validate(self):
    """Check for incompatible configurations."""
    if self.settings.data_type == "timeseries" and self.model.name == "stable-diffusion-v1-5":
        raise ValueError(
            "Stable Diffusion models require 2D image data. "
            "For timeseries data, use 'dit', 'hourglass', or 'localmamba'."
        )
```

Call this in `load_diffusion_config()` after parsing.

---

## Phase 2: Preprocessing Pipeline

### 2.1 Add Normalization Statistics Computation

**Files**: `signal_diffusion/data/{seed,parkinsons,math,longitudinal}.py`

For each dataset preprocessor, compute normalization statistics BEFORE preprocessing begins, then use them to normalize data during saving:

```python
class SEEDTimeSeriesPreprocessor(BaseSpectrogramPreprocessor):
    """Generate time-domain .npy files for SEED dataset."""

    def __init__(self, settings, *, nsamps, ovr_perc, fs, ...):
        super().__init__(settings, dataset_name="seed")
        self.nsamps = nsamps
        # ... other initialization

        # Load or compute normalization stats
        self.norm_stats = self._load_or_compute_normalization_stats()

    def _load_or_compute_normalization_stats(self) -> dict:
        """Load existing stats or compute from raw data."""
        # Use dataset-specific naming
        stats_path = self.dataset_settings.output / "seed_normalization_stats.json"

        if stats_path.exists():
            logger.info(f"Loading existing normalization stats from {stats_path}")
            with stats_path.open() as f:
                return json.load(f)

        logger.info("Computing normalization statistics from raw EEG data...")
        return self._compute_normalization_stats(stats_path)

    def _compute_normalization_stats(self, stats_path: Path) -> dict:
        """Compute per-channel mean/std from ALL raw data (train+val+test)."""
        n_channels = len(self.channel_indices)
        channel_sums = np.zeros(n_channels, dtype=np.float64)
        channel_sq_sums = np.zeros(n_channels, dtype=np.float64)
        total_samples = 0

        # Iterate through ALL subjects (not just train split)
        for subject_id in tqdm(self.subjects(), desc="Computing normalization stats"):
            for session_idx, file_path in self._get_session_files(subject_id):
                # Load raw EEG
                raw = mne.io.read_raw_cnt(file_path, preload=True, verbose="WARNING")
                data = raw.get_data()[self.channel_indices, :]

                # Decimate if needed (same as preprocessing)
                if self.decimation > 1:
                    data = decimate(data, self.decimation, axis=1, zero_phase=True)

                channel_sums += data.sum(axis=1)
                channel_sq_sums += (data ** 2).sum(axis=1)
                total_samples += data.shape[1]

        channel_means = channel_sums / total_samples
        channel_vars = channel_sq_sums / total_samples - channel_means ** 2
        channel_stds = np.sqrt(np.maximum(channel_vars, 1e-8))

        stats = {
            "channel_means": channel_means.tolist(),
            "channel_stds": channel_stds.tolist(),
            "n_eeg_channels": n_channels,  # Number of EEG channels (e.g., 62 for SEED)
            "n_samples_total": int(total_samples),  # Total samples across all data
        }

        # Save with dataset-specific name
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open('w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved normalization stats to {stats_path}")
        logger.info(f"  Mean range: [{channel_means.min():.2e}, {channel_means.max():.2e}]")
        logger.info(f"  Std range: [{channel_stds.min():.2e}, {channel_stds.max():.2e}]")

        return stats
```

### 2.2 Implement Time-Series Preprocessors

**Files**: `signal_diffusion/data/{seed,parkinsons,math,longitudinal}.py`

Modify `generate_examples()` to normalize data using precomputed stats before saving:

```python
def generate_examples(self, *, subject_id, split, resolution=None, hop_length=None):
    """Generate time-series examples (resolution/hop_length ignored)."""

    info = self._subject_metadata(subject_id)

    # Get normalization arrays
    means = np.array(self.norm_stats["channel_means"], dtype=np.float32).reshape(-1, 1)
    stds = np.array(self.norm_stats["channel_stds"], dtype=np.float32).reshape(-1, 1)

    for session_idx, file_path in self._get_session_files(subject_id):
        # Load raw EEG
        raw = mne.io.read_raw_cnt(file_path, preload=True, verbose="WARNING")
        data = raw.get_data()[self.channel_indices, :]

        # Decimate if needed
        if self.decimation > 1:
            data = decimate(data, self.decimation, axis=1, zero_phase=True)

        # Sliding window
        shift = self.nsamps - self.noverlap
        n_blocks = int(np.floor((data.shape[1] - self.nsamps) / shift)) + 1

        for block_idx in range(n_blocks):
            start = block_idx * shift
            end = start + self.nsamps
            block = data[:, start:end].astype(np.float32)

            # Apply z-score normalization using precomputed stats
            block_normalized = (block - means) / (stds + 1e-8)

            # Metadata (same as spectrogram version)
            metadata = {
                "session": session_idx,
                "block": block_idx,
                "gender": info.gender,
                "age": info.age,
                # ... task-specific labels
            }

            # Use custom writer for .npy (save normalized data)
            relative = Path(subject_id) / f"timeseries-s{session_idx:02d}-{block_idx:04d}.npy"
            block_copy = block_normalized.copy()  # Capture in closure

            yield SpectrogramExample(
                subject_id=subject_id,
                relative_path=relative,
                metadata=metadata,
                image=None,
                writer=lambda path, data=block_copy: np.save(path, data)
            )
```

**Key points**:
- Compute normalization stats ONCE before preprocessing (reuse if exists)
- Apply z-score normalization BEFORE saving (saves normalized data)
- Use dataset-specific naming: `seed_normalization_stats.json`
- Store as float32 (4 bytes per sample)
- Use `lambda` with default argument to capture block in closure
- Leverage existing `SpectrogramExample.writer` mechanism

---

## Phase 3: Dataset Loading

### 3.1 Create Transform Module

**New file**: `signal_diffusion/data/transforms/__init__.py`

```python
"""Transform utilities for EEG data."""

from .timeseries import (
    GaussianNoise,
    ChannelDropout,
    TemporalCrop,
)

__all__ = [
    "GaussianNoise",
    "ChannelDropout",
    "TemporalCrop",
]
```

**New file**: `signal_diffusion/data/transforms/timeseries.py`

```python
"""Transforms for time-domain EEG signals."""

import torch
import torch.nn as nn
from typing import Optional


class ZScoreNormalize(nn.Module):
    """Z-score normalization using precomputed statistics."""

    def __init__(self, means: torch.Tensor, stds: torch.Tensor):
        super().__init__()
        self.register_buffer("means", means.view(-1, 1))
        self.register_buffer("stds", stds.view(-1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply z-score: (x - mean) / std."""
        return (x - self.means) / (self.stds + 1e-8)


class GaussianNoise(nn.Module):
    """Add Gaussian noise for augmentation."""

    def __init__(self, std: float = 0.01):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return x + torch.randn_like(x) * self.std
        return x


class ChannelDropout(nn.Module):
    """Randomly zero out channels."""

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0:
            mask = torch.rand(x.shape[0], 1) > self.p
            return x * mask
        return x


class TemporalCrop(nn.Module):
    """Extract fixed-length window from signal."""

    def __init__(self, length: int, center: bool = False):
        super().__init__()
        self.length = length
        self.center = center

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.length:
            return x

        if self.center:
            start = (x.shape[-1] - self.length) // 2
        else:
            start = torch.randint(0, x.shape[-1] - self.length + 1, (1,)).item()

        return x[..., start:start + self.length]
```

### 3.2 Create Time-Domain Dataset Classes

**Files**: `signal_diffusion/data/{seed,parkinsons,math,longitudinal}.py`

Add dataset classes for time-domain loading:

```python
class SEEDTimeSeriesDataset(torch.utils.data.Dataset):
    """SEED time-domain dataset for classification/diffusion."""

    def __init__(
        self,
        settings: Settings,
        *,
        split: str,
        tasks: tuple[str, ...],
        transform: Optional[nn.Module] = None,
        target_format: str = "dict",
        expected_length: Optional[int] = None,  # NEW: Expected signal length
    ):
        self.settings = settings
        self.dataset_settings = settings.dataset("seed")
        self.split = split
        self.tasks = tasks
        self.target_format = target_format
        self.expected_length = expected_length

        # Load metadata
        metadata_path = self.dataset_settings.output / f"{split}-metadata.csv"
        self.metadata = pd.read_csv(metadata_path)
        self.root = self.dataset_settings.output

        # Load normalization stats to get n_eeg_channels
        stats_path = self.dataset_settings.output / "seed_normalization_stats.json"
        with stats_path.open() as f:
            norm_stats = json.load(f)
        self.n_eeg_channels = norm_stats["n_eeg_channels"]

        # Validate expected_length against actual data
        if self.expected_length is not None:
            self._validate_signal_length()

        # Transform (data is already normalized, just apply augmentation)
        self.transform = transform

    def _validate_signal_length(self):
        """Warn if configured resolution doesn't match actual signal length."""
        # Sample first file to check length
        first_file = self.root / self.metadata.iloc[0]["file_name"]
        sample_data = np.load(first_file)
        actual_length = sample_data.shape[1]

        if actual_length != self.expected_length:
            logger.warning(
                f"Expected signal length {self.expected_length} but found {actual_length} "
                f"in {first_file.name}. This may cause issues during training."
            )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int):
        row = self.metadata.iloc[index]
        data_path = self.root / row["file_name"]

        # Load .npy file (already normalized during preprocessing)
        data = np.load(data_path)  # (n_channels, n_samples) float32
        tensor = torch.from_numpy(data).float()

        # Apply transform (optional augmentation only, NO normalization)
        if self.transform:
            tensor = self.transform(tensor)

        # Encode targets (identical to spectrogram datasets)
        targets = {name: SEED_LABELS.encode(name, row) for name in self.tasks}

        sample = {
            "signal": tensor,  # Use 'signal' for time-series (clearer semantics)
            "targets": targets,
            "metadata": row.to_dict(),
        }

        if self.target_format == "tuple":
            if len(self.tasks) == 1:
                return tensor, targets[self.tasks[0]]
            return tensor, targets

        return sample
```

### 3.3 Update Dataset Registry and Populate extras

**File**: `signal_diffusion/classification/datasets.py`

Register time-domain datasets and ensure `dataset.extras` is populated with `n_eeg_channels` and `sequence_length`:

```python
_DATASET_CLS: Mapping[str, type] = {
    "math": MathDataset,
    "parkinsons": ParkinsonsDataset,
    "seed": SEEDDataset,
    "mit": MITDataset,
    # NEW: Time-domain variants
    "seed_timeseries": SEEDTimeSeriesDataset,
    "parkinsons_timeseries": ParkinsonsTimeSeriesDataset,
    "math_timeseries": MathTimeSeriesDataset,
    "longitudinal_timeseries": LongitudinalTimeSeriesDataset,
}

def build_dataset(...):
    """Build dataset and populate extras for time-series."""
    dataset_cls = _DATASET_CLS[dataset_name]
    dataset = dataset_cls(...)

    # For time-series datasets, populate extras with required metadata
    if "timeseries" in dataset_name and hasattr(dataset, 'n_eeg_channels'):
        # These are read from normalization stats during dataset init
        config.dataset.extras["n_eeg_channels"] = dataset.n_eeg_channels
        # sequence_length should match config.dataset.resolution
        config.dataset.extras["sequence_length"] = config.dataset.resolution

    return dataset
```

**File**: `signal_diffusion/data/__init__.py`

Export new classes:

```python
from .seed import SEEDTimeSeriesDataset, SEEDTimeSeriesPreprocessor
# ... repeat for other datasets

__all__ = [
    # ... existing exports
    "SEEDTimeSeriesDataset",
    "SEEDTimeSeriesPreprocessor",
    # ...
]
```

---

## Phase 4: Classification Models

### 4.1 Implement 1D CNN Backbone

**File**: `signal_diffusion/classification/backbones.py`

Add a 1D CNN that mirrors the 2D backbone’s configuration knobs (depth, layer_repeats, activation, dropout, embedding_dim):

- Input conv width is `max(8, next_power_of_two(input_channels))` to avoid shrinking early layers (typical timeseries ≈20 chans).
- Growth factor is derived from the ratio to `embedding_dim`, clamped to {2, 4} and rounded, so channels never decrease after the input conv.
- Kernel sizes decrease by 2 per block with a floor of 3: `kernel = max(3, 1 + 2*depth - 2*block_idx)` which yields 7/5/3 for depth=3.
- Block structure matches the 2D version: input conv → activation → norm → dropout → avg pool, then repeated blocks with norm → repeated convs → dropout → avg pool, followed by adaptive pooling and a linear projection to `embedding_dim`.

### 4.2 Update Classifier Factory

**File**: `signal_diffusion/classification/factory.py`

Wire the new 1D backbone to reuse the same config fields as the 2D CNN (`activation`, `dropout`, `embedding_dim`, `depth`, `layer_repeats`). Both `cnn_1d` and `cnn_1d_light` variants call `CNNBackbone1D` with these shared parameters; light vs. full depth is determined by the `depth`/`layer_repeats` values passed in the config (no separate hardcoded channel tuples).

### 4.3 Update Training Loop to Handle Both Keys

**File**: `signal_diffusion/training/classification.py`

Update training loop to handle both 'image' (spectrograms) and 'signal' (time-series):

```python
def training_step(batch):
    # Get input data - support both 'image' and 'signal' keys
    if "signal" in batch:
        inputs = batch["signal"]
    elif "image" in batch:
        inputs = batch["image"]
    else:
        raise KeyError("Batch must contain 'image' or 'signal' key")

    targets = batch["targets"]

    # Forward pass
    outputs = model(inputs)
    # ... rest of training logic
```

Similarly update validation loop.

### 4.4 Validate Configuration

Add validation to ensure backbone matches data type:

```python
def validate_classifier_config(config: ClassifierConfig, settings: Settings):
    """Validate that backbone and data_type are compatible."""
    is_1d_backbone = "1d" in config.backbone.lower()
    is_timeseries = settings.data_type == "timeseries"

    if is_1d_backbone and not is_timeseries:
        raise ValueError(
            f"Backbone '{config.backbone}' requires data_type='timeseries', "
            f"but got '{settings.data_type}'"
        )

    if is_timeseries and not is_1d_backbone:
        logger.warning(
            f"Using 2D backbone '{config.backbone}' with timeseries data. "
            "Consider using 'cnn_1d' or 'cnn_1d_light' for better performance."
        )
```

Call this in `signal_diffusion/training/classification.py` after loading config.

---

## Phase 5: Diffusion Models

### 5.1 Add Dynamic Noise Shapes to Adapters

**Files**: `signal_diffusion/diffusion/models/{dit,hourglass,localmamba}.py`

Each adapter needs a helper to create noise tensors with appropriate shape. For time-series data, the shape is `(B, in_channels, n_eeg_channels, sequence_length)` where:
- `in_channels` comes from `model.extras.in_channels` (typically 1, the artificial channel dimension added by collate_fn)
- `n_eeg_channels` is retrieved from `dataset.extras["n_eeg_channels"]` (computed from preprocessing, e.g., 62 for SEED)
- `sequence_length` comes from `dataset.extras["sequence_length"]` (should match `dataset.resolution`)

```python
def _create_noise_tensor(
    self,
    config: DiffusionConfig,
    num_samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create noise tensor with shape appropriate for data type."""

    if config.settings.data_type == "timeseries":
        # Time-series: (B, in_channels, n_eeg_channels, sequence_length)
        # in_channels: from model config (typically 1, may be >1 if augmented)
        # n_eeg_channels: EEG channels (dataset-specific, e.g., 62 for SEED)
        # sequence_length: temporal dimension (from config.dataset.resolution)
        # This allows reusing 2D models with asymmetric patching
        in_channels = self._extras.in_channels
        n_eeg_channels = config.dataset.extras.get("n_eeg_channels")
        sequence_length = config.dataset.extras.get("sequence_length")

        if n_eeg_channels is None or sequence_length is None:
            raise ValueError(
                f"Time-series config missing required extras: "
                f"n_eeg_channels={n_eeg_channels}, sequence_length={sequence_length}. "
                f"Ensure dataset.extras is populated during build_dataset()."
            )

        return torch.randn(
            (num_samples, in_channels, n_eeg_channels, sequence_length),
            device=device,
            dtype=dtype,
        )
    else:
        # 2D image: (B, C, H, W)
        channels = self._extras.in_channels
        size = config.model.sample_size or config.dataset.resolution
        return torch.randn(
            (num_samples, channels, size, size),
            device=device,
            dtype=dtype,
        )
```

Update `generate_samples()` to use this:

```python
def generate_samples(
    self,
    accelerator: Accelerator,
    config: DiffusionConfig,
    modules: DiffusionModules,
    *,
    num_images: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate unconditional samples."""

    # Create noise with appropriate shape
    sample = self._create_noise_tensor(
        config,
        num_images,
        device=accelerator.device,
        dtype=modules.weight_dtype,
    )

    # Existing sampling loop
    for t in scheduler.timesteps:
        # ...
```

### 5.2 Update DiffusionBatch to Handle Both Keys

**File**: `signal_diffusion/diffusion/data.py`

Update batch construction to handle both 'image' and 'signal' keys:

```python
@dataclass
class DiffusionBatch:
    """Batch of data for diffusion training."""
    pixel_values: torch.Tensor  # Keep name for backward compatibility
    captions: Optional[torch.Tensor] = None
    class_labels: Optional[torch.Tensor] = None

def collate_fn(examples):
    """Collate function supporting both image and signal data."""
    # Support both 'image' and 'signal' keys
    if "signal" in examples[0]:
        pixel_values = torch.stack([ex["signal"] for ex in examples])
        # Add channel dimension: (B, C, L) -> (B, 1, C, L)
        pixel_values = pixel_values.unsqueeze(1)
    elif "image" in examples[0]:
        pixel_values = torch.stack([ex["image"] for ex in examples])
    else:
        raise KeyError("Examples must contain 'image' or 'signal' key")

    # Handle optional conditioning
    captions = None
    if "caption" in examples[0] and examples[0]["caption"] is not None:
        captions = torch.stack([ex["caption"] for ex in examples])

    class_labels = None
    if "class_label" in examples[0]:
        class_labels = torch.tensor([ex["class_label"] for ex in examples])

    return DiffusionBatch(
        pixel_values=pixel_values,
        captions=captions,
        class_labels=class_labels,
    )
```

### 5.3 Add Conditional Transform Pipeline

**File**: `signal_diffusion/diffusion/data.py`

Add conditional transform pipeline:

```python
def _build_transforms(config: DatasetConfig, settings: Settings, *, train: bool):
    """Build transforms based on data type."""

    if settings.data_type == "timeseries":
        return _build_timeseries_transforms(config, train=train)
    else:
        return _build_spectrogram_transforms(config, train=train)


def _build_timeseries_transforms(config: DatasetConfig, *, train: bool):
    """Transforms for 1D signals."""
    from signal_diffusion.data.transforms import GaussianNoise, TemporalCrop

    transforms = []

    # Note: Data is already normalized during preprocessing

    if train:
        # Data augmentation - configurable Gaussian noise
        noise_std = config.extras.get("gaussian_noise_std", 0.01)
        if noise_std > 0:
            transforms.append(GaussianNoise(std=noise_std))

    # Ensure fixed length (warn if mismatch detected during dataset init)
    if config.resolution:
        transforms.append(TemporalCrop(length=config.resolution, center=not train))

    return transforms_v2.Compose(transforms) if transforms else None


def _build_spectrogram_transforms(config: DatasetConfig, *, train: bool):
    """Transforms for 2D spectrograms (existing logic)."""
    resolution = config.resolution

    transforms = [
        transforms_v2.ToImage(),
        transforms_v2.Resize(resolution, interpolation=InterpolationMode.BILINEAR),
    ]

    if train:
        transforms.append(transforms_v2.RandomCrop(resolution))
    else:
        transforms.append(transforms_v2.CenterCrop(resolution))

    transforms.append(transforms_v2.ToDtype(torch.float32, scale=True))
    transforms.append(transforms_v2.Normalize([0.5], [0.5]))

    return transforms_v2.Compose(transforms)
```

Update `build_dataloaders()` to pass settings and use custom collate:

```python
def build_dataloaders(
    config: DiffusionConfig,
    settings: Settings,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val dataloaders."""

    # Build transforms
    train_transform = _build_transforms(config.dataset, settings, train=True)
    val_transform = _build_transforms(config.dataset, settings, train=False)

    # Build datasets with expected_length for validation
    expected_length = config.dataset.resolution if settings.data_type == "timeseries" else None

    train_dataset = build_dataset(
        ...,
        expected_length=expected_length,
    )
    val_dataset = build_dataset(
        ...,
        expected_length=expected_length,
    )

    # Use custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        collate_fn=collate_fn,  # Handles both image/signal keys
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.dataset.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )

    return train_loader, val_loader
```

### 5.4 Configure Asymmetric Patching for Time-Series

**Files**: `signal_diffusion/diffusion/models/{dit,hourglass,localmamba}.py`

For time-series data with shape (B, 1, n_channels, n_samples), use asymmetric patch sizes to match the high aspect ratio (e.g., 62×2000 ≈ 1:32 ratio). Configure models to use (1, 8) patches instead of square (2, 2) patches.

**DiT adapter changes**:

```python
def build_modules(self, accelerator, config, tokenizer=None):
    """Build DiT modules with appropriate patch size."""

    # Determine patch size based on data type
    if config.settings.data_type == "timeseries":
        # Asymmetric patches for high aspect ratio signals
        patch_size = (1, 8)  # (height_patch, width_patch)
        in_channels = 1
    else:
        # Square patches for images
        patch_size = self._extras.patch_size or 2
        in_channels = self._extras.in_channels

    denoiser = DiTTransformer2DModel(
        num_layers=self._extras.num_layers,
        patch_size=patch_size,
        in_channels=in_channels,
        # ... other params
    )
    # ...
```

**Hourglass/LocalMamba**: Similarly update to accept tuple `patch_size=(height, width)` and configure based on `data_type`.

**Key insight**: By adding a channel dimension (1×n_channels×n_samples), we can reuse 2D architectures without major changes, just adjusting patch size to respect temporal structure.

### 5.5 Add PSNR Metric

**New file**: `signal_diffusion/metrics/reconstruction.py`

```python
"""Reconstruction quality metrics."""

import torch


def compute_psnr(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    max_value: float = 1.0,
) -> float:
    """Compute Peak Signal-to-Noise Ratio.

    Args:
        original: Ground truth tensor
        reconstructed: Reconstructed tensor
        max_value: Maximum possible value (1.0 for normalized data)

    Returns:
        PSNR in dB
    """
    mse = torch.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')

    return 20 * torch.log10(torch.tensor(max_value) / torch.sqrt(mse)).item()


def compute_batch_psnr(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    max_value: float = 1.0,
) -> tuple[float, float]:
    """Compute mean and std PSNR across batch.

    Returns:
        (mean_psnr, std_psnr)
    """
    psnrs = []
    for orig, recon in zip(originals, reconstructions):
        psnrs.append(compute_psnr(orig, recon, max_value))

    psnrs = torch.tensor(psnrs)
    return psnrs.mean().item(), psnrs.std().item()
```

Update `signal_diffusion/metrics/__init__.py`:

```python
from .reconstruction import compute_psnr, compute_batch_psnr

__all__ = [
    # ... existing
    "compute_psnr",
    "compute_batch_psnr",
]
```

### 5.6 Update Diffusion Training Loop

**File**: `signal_diffusion/training/diffusion.py`

Add time-series specific validation:

```python
def validation_loop(
    accelerator: Accelerator,
    config: DiffusionConfig,
    modules: DiffusionModules,
    adapter: DiffusionAdapter,
    val_dataloader: DataLoader,
) -> dict[str, float]:
    """Run validation."""

    metrics = {}

    if config.settings.data_type == "timeseries":
        # Time-series specific metrics
        from signal_diffusion.metrics import compute_batch_psnr

        # Generate samples
        samples = adapter.generate_samples(accelerator, config, modules, num_images=16)

        # Get real samples
        real_batch = next(iter(val_dataloader))
        real_samples = real_batch.pixel_values[:16].to(accelerator.device)

        # Compute PSNR
        mean_psnr, std_psnr = compute_batch_psnr(real_samples, samples)
        metrics["timeseries/psnr"] = mean_psnr
        metrics["timeseries/psnr_std"] = std_psnr

        # Log statistics
        metrics["timeseries/mean"] = samples.mean().item()
        metrics["timeseries/std"] = samples.std().item()
        metrics["timeseries/min"] = samples.min().item()
        metrics["timeseries/max"] = samples.max().item()

    else:
        # Existing image metrics (KID, etc.)
        # ...

    return metrics
```

---

## Phase 6: Meta-Dataset for Time-Domain

### 6.1 Create Weighted Time-Domain Dataset Script

**New file**: `scripts/gen_weighted_timeseries_dataset.py`

```python
"""Generate weighted meta time-series dataset."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Array2D, Value
from tqdm import tqdm

from signal_diffusion.config import load_settings
from signal_diffusion.data.meta import MetaSampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    settings = load_settings(args.config)

    # Load metadata from each dataset
    all_metadata = []
    for dataset_name in args.datasets:
        dataset_settings = settings.dataset(dataset_name)

        for split in ["train", "val", "test"]:
            metadata_path = dataset_settings.output / f"{split}-metadata.csv"
            df = pd.read_csv(metadata_path)
            df["dataset"] = dataset_name
            df["split"] = split
            all_metadata.append(df)

    metadata = pd.concat(all_metadata, ignore_index=True)

    # Compute weights using MetaSampler
    sampler = MetaSampler(metadata, seed=args.seed)
    weights = sampler.compute_weights()

    # Generate weighted dataset
    args.output.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        split_metadata = metadata[metadata["split"] == split]
        split_weights = weights[metadata["split"] == split]

        # Assign copies based on weights
        copies = sampler.assign_copies(split_weights)

        # Duplicate samples
        weighted_samples = []
        for idx, n_copies in tqdm(enumerate(copies), desc=f"Processing {split}"):
            if n_copies == 0:
                continue

            row = split_metadata.iloc[idx]

            # Load time-series
            dataset_settings = settings.dataset(row["dataset"])
            data_path = dataset_settings.output / row["file_name"]
            timeseries = np.load(data_path)

            # Create n_copies
            for copy_idx in range(n_copies):
                weighted_samples.append({
                    "timeseries": timeseries,
                    "gender": row["gender"],
                    "age": row.get("age", -1),
                    "dataset": row["dataset"],
                    "original_file": row["file_name"],
                })

        # Create HF dataset
        features = Features({
            "timeseries": Array2D(shape=(None, None), dtype="float32"),
            "gender": Value("int32"),
            "age": Value("float32"),
            "dataset": Value("string"),
            "original_file": Value("string"),
        })

        split_dataset = Dataset.from_dict(
            {k: [s[k] for s in weighted_samples] for k in weighted_samples[0].keys()},
            features=features,
        )

        # Save as parquet
        split_dataset.to_parquet(args.output / f"{split}.parquet")

        print(f"Saved {len(split_dataset)} samples to {split}.parquet")

    print(f"Dataset saved to {args.output}")


if __name__ == "__main__":
    main()
```

### 6.2 Create Meta Time-Domain Dataset Class

**New file**: `signal_diffusion/data/meta_timeseries.py`

```python
"""Meta time-series dataset for multi-dataset training."""

import torch
from datasets import load_from_disk
from pathlib import Path
from typing import Optional

from signal_diffusion.data.specs import META_LABELS


class MetaTimeSeriesDataset(torch.utils.data.Dataset):
    """Meta dataset for time-series from multiple sources."""

    def __init__(
        self,
        data_dir: Path,
        split: str,
        tasks: tuple[str, ...],
        transform: Optional[torch.nn.Module] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.tasks = tasks
        self.transform = transform

        # Load HF dataset
        dataset_dict = load_from_disk(str(self.data_dir))
        self.dataset = dataset_dict[split]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        row = self.dataset[index]

        # Get time-series
        timeseries = torch.from_numpy(row["timeseries"]).float()

        # Apply transform
        if self.transform:
            timeseries = self.transform(timeseries)

        # Encode targets
        targets = {}
        for task_name in self.tasks:
            if task_name in META_LABELS:
                targets[task_name] = META_LABELS.encode(task_name, row)

        return {
            "signal": timeseries,  # Use 'signal' key for time-series
            "targets": targets,
            "metadata": {k: v for k, v in row.items() if k != "timeseries"},
        }
```

Register in `signal_diffusion/classification/datasets.py`:

```python
_DATASET_CLS["meta_timeseries"] = MetaTimeSeriesDataset
```

---

## Phase 7: Integration & Testing

### 7.1 Create Example Configs

**File**: `config/classification/seed-timeseries.toml`

```toml
[settings]
config = "config/default.toml"
trainer = "classification"

[data]
data_type = "timeseries"

[dataset]
name = "seed_timeseries"
train_split = "train"
val_split = "val"
tasks = ["emotion", "gender"]

[model]
backbone = "cnn_1d"
input_channels = 62
embedding_dim = 256
dropout = 0.3

[training]
epochs = 25
learning_rate = 3e-4
optimizer = "adamw"
weight_decay = 1e-4

[training.task_weights]
emotion = 1.0
gender = 1.0
```

**File**: `config/diffusion/dit-timeseries.toml`

```toml
[settings]
config = "config/default.toml"
trainer = "diffusion"

[data]
data_type = "timeseries"

[dataset]
name = "seed_timeseries"
train_split = "train"
val_split = "val"
batch_size = 32
resolution = 2000  # sequence length (must match preprocessing nsamps)
num_classes = 0

[dataset.extras]
# REQUIRED for time-series: populated from dataset during build_dataset()
# n_eeg_channels = 62  # Auto-populated from normalization stats
# sequence_length = 2000  # Auto-populated from resolution
gaussian_noise_std = 0.01  # Set to 0 to disable augmentation

[model]
name = "dit"
sample_size = 2000
conditioning = "none"

[model.extras]
in_channels = 1  # Artificial channel dimension added by collate_fn

[objective]
prediction_type = "vector_field"
flow_match_timesteps = 1000

[training]
epochs = 100
mixed_precision = "fp16"
eval_strategy = "epoch"
```

### 7.2 Update Documentation

**File**: `CLAUDE.md`

Add section:

```markdown
## Time-Domain Data Processing

Signal Diffusion supports both spectrogram and time-domain EEG data.

### Configuration

Set `data_type` in config:

```toml
[data]
data_type = "timeseries"  # or "spectrogram" (default)
```

### Preprocessing

```python
from signal_diffusion.data import SEEDTimeSeriesPreprocessor

preprocessor = SEEDTimeSeriesPreprocessor(settings, nsamps=512, fs=200)
preprocessor.preprocess(seed=42)
# Generates .npy files + normalization_stats.json
```

### Training

Classification:
```bash
uv run python -m signal_diffusion.training.classification \
    config/classification/seed-timeseries.toml \
    --output-dir runs/classification/timeseries
```

Diffusion:
```bash
uv run python -m signal_diffusion.training.diffusion \
    config/diffusion/dit-timeseries.toml \
    --output-dir runs/diffusion/timeseries
```

### Models

- **Classification**: Use `cnn_1d` or `cnn_1d_light` backbone
- **Diffusion**: Use `dit`, `hourglass`, or `localmamba` (NOT `stable-diffusion-v1-5`)
```

### 7.3 Testing Checklist

1. **Configuration**
   - [ ] Load config with `data_type = "timeseries"`
   - [x] Verify separate output paths (seed/timeseries/ vs seed/stfts/)
   - [ ] Validate Stable Diffusion raises error
   - [ ] Validate backbone/data_type mismatch detection

2. **Preprocessing**
   - [x] Generate SEED time-series .npy files in seed/timeseries/
   - [x] Verify seed_normalization_stats.json created with dataset-specific name
   - [x] Verify stats reused on second run (not recomputed)
   - [ ] Verify saved .npy files contain normalized data (mean≈0, std≈1)
   - [ ] Check metadata CSVs match spectrogram version
   - [x] Verify files NOT mixed with spectrograms

3. **Dataset Loading**
   - [x] Load SEEDTimeSeriesDataset with expected_length parameter
   - [x] Verify warning logged if resolution mismatch detected
   - [x] Verify shape: (n_channels, n_samples)
   - [ ] Verify data already normalized (mean≈0, std≈1)
   - [x] Verify 'signal' key (not 'image') in return dict
   - [ ] Verify targets dict matches label registry
   - [ ] Test Gaussian noise augmentation (std=0 disables it)

4. **Classification Training**
   - [ ] Train CNNBackbone1D on SEED time-series
   - [ ] Verify training loop handles 'signal' key
   - [ ] Verify loss decreases
   - [ ] Verify validation metrics logged
   - [ ] Verify checkpoint saved

5. **Diffusion Training**
   - [ ] Train DiT with asymmetric patches (1×8) on time-series
   - [ ] Verify shape (B, 1, n_channels, n_samples) fed to model
   - [ ] Generate samples (raw tensors)
   - [ ] Verify PSNR metric computed
   - [ ] Verify samples have correct shape (B, 1, n_channels, n_samples)
   - [ ] Verify inverse transform removes channel dimension

6. **Meta-Dataset**
   - [ ] Generate weighted time-series dataset
   - [ ] Verify parquet files created
   - [ ] Load MetaTimeSeriesDataset
   - [ ] Train classifier on meta dataset

7. **Backward Compatibility**
   - [ ] Train classifier with `data_type = "spectrogram"` and 'image' key
   - [ ] Train diffusion with `data_type = "spectrogram"` and square patches
   - [ ] Verify no regressions

---

## Implementation Summary

### New Files (11)

1. `signal_diffusion/data/transforms/__init__.py`
2. `signal_diffusion/data/transforms/timeseries.py`
3. `signal_diffusion/data/meta_timeseries.py`
4. `signal_diffusion/metrics/reconstruction.py`
5. `scripts/gen_weighted_timeseries_dataset.py`
6. `config/classification/seed-timeseries.toml`
7. `config/classification/parkinsons-timeseries.toml`
8. `config/diffusion/dit-timeseries.toml`
9. `config/diffusion/hourglass-timeseries.toml`
10. `config/diffusion/localmamba-timeseries.toml`

### Modified Files (16)

1. `signal_diffusion/config/settings.py` - Add `data_type` field
2. `signal_diffusion/data/seed.py` - Add `SEEDTimeSeriesPreprocessor`, `SEEDTimeSeriesDataset`
3. `signal_diffusion/data/parkinsons.py` - Add time-series variants
4. `signal_diffusion/data/math.py` - Add time-series variants
5. `signal_diffusion/data/longitudinal.py` - Add time-series variants
6. `signal_diffusion/data/__init__.py` - Export new classes
7. `signal_diffusion/classification/datasets.py` - Register time-series datasets
8. `signal_diffusion/classification/backbones.py` - Add `CNNBackbone1D`
9. `signal_diffusion/classification/factory.py` - Add validation, backbone selection
10. `signal_diffusion/training/classification.py` - Call validation, handle 'signal' key
11. `signal_diffusion/diffusion/config.py` - Add validation method
12. `signal_diffusion/diffusion/data.py` - Conditional transforms, handle 'signal' key
13. `signal_diffusion/diffusion/models/dit.py` - Dynamic noise shapes, asymmetric patches
14. `signal_diffusion/diffusion/models/hourglass.py` - Dynamic noise shapes, asymmetric patches
15. `signal_diffusion/diffusion/models/localmamba.py` - Dynamic noise shapes, asymmetric patches
16. `signal_diffusion/training/diffusion.py` - Time-series validation metrics, handle 'signal' key
17. `signal_diffusion/metrics/__init__.py` - Export PSNR
18. `config/default.toml` - Add `data_type = "spectrogram"`
19. `CLAUDE.md` - Document time-domain usage

### Key Design Decisions

1. **Normalization**: Per-channel z-score computed BEFORE preprocessing, applied during save
   - Stats saved with dataset-specific naming: `{dataset}_normalization_stats.json`
   - Stats include `n_eeg_channels` (actual EEG channel count, e.g., 62 for SEED)
   - Reuse existing stats if available (don't recompute)
   - Normalized data stored in .npy files (no runtime normalization needed)
2. **EEG Channel Count** (`n_eeg_channels`):
   - Computed during preprocessing and stored in normalization stats
   - Loaded by time-series dataset classes during `__init__`
   - Populated into `dataset.extras["n_eeg_channels"]` by `build_dataset()`
   - Used by diffusion adapters to create correct noise tensor shape
3. **Sequence Length** (`sequence_length`):
   - Matches `dataset.resolution` in config (e.g., 2000 samples)
   - Populated into `dataset.extras["sequence_length"]` by `build_dataset()`
   - Used by diffusion adapters to create correct noise tensor shape
4. **In-Channels** (`in_channels`):
   - Set in `[model.extras]` for diffusion configs (typically 1)
   - Set in `[model]` for classification configs as `input_channels` (matches `n_eeg_channels`)
5. **Inheritance**: Time-series preprocessors inherit from `BaseSpectrogramPreprocessor` (pragmatic)
6. **File format**: Uncompressed .npy (float32)
7. **Configuration**: `data_type` at settings level (not model level)
8. **Transforms**: New centralized module at `signal_diffusion/data/transforms/`
   - No TimeReverse (removed per feedback)
   - Configurable Gaussian noise std (set to 0 to disable)
9. **Storage**: Separate directories (`seed/timeseries/` vs `seed/stfts/`)
10. **Dataset keys**: Use `'signal'` key (not `'image'`) for time-series clarity
11. **Diffusion shapes**: Shape is `(B, in_channels, n_eeg_channels, sequence_length)` for time-series
12. **Patching**: Asymmetric patches (1×8) for high aspect ratio signals (e.g., 62×2000)
13. **Validation**:
    - Error on Stable Diffusion + time-series
    - Warn if configured resolution doesn't match actual signal length
    - Warn on backbone/data_type mismatches
    - Error if `dataset.extras` missing required keys (`n_eeg_channels`, `sequence_length`)

---

## Implementation Order

1. **Phase 1** (Configuration) - Foundation
2. **Phase 3.1** (Transforms module) - Required by datasets
3. **Phase 2** (Preprocessing) - Generate data
4. **Phase 3.2-3.3** (Dataset loading) - Consume data
5. **Phase 4** (Classification) - First training use case
6. **Phase 5.4** (PSNR metric) - Before diffusion validation
7. **Phase 5.1-5.3** (Diffusion) - Generative modeling
8. **Phase 6** (Meta-dataset) - Advanced use case
9. **Phase 7** (Testing & docs) - Validation

---

## Configuration Flow for Time-Series Data

### Data Flow from Preprocessing → Diffusion

```
1. Phase 2: Preprocessing
   SEEDTimeSeriesPreprocessor._compute_normalization_stats()
   └─> Writes: seed_normalization_stats.json with:
       {
         "channel_means": [...],
         "channel_stds": [...],
         "n_eeg_channels": 62,        ← Actual EEG channel count
         "n_samples_total": 123456
       }

2. Phase 3: Dataset Loading
   SEEDTimeSeriesDataset.__init__()
   └─> Reads: seed_normalization_stats.json
   └─> Stores: self.n_eeg_channels = 62

3. Phase 3.3: build_dataset()
   └─> Populates config.dataset.extras:
       {
         "n_eeg_channels": 62,        ← From dataset.n_eeg_channels
         "sequence_length": 2000,     ← From config.dataset.resolution
         "gaussian_noise_std": 0.01
       }

4. Phase 5.1: Diffusion _create_noise_tensor()
   └─> Reads from:
       - model.extras.in_channels → 1
       - dataset.extras["n_eeg_channels"] → 62
       - dataset.extras["sequence_length"] → 2000
   └─> Creates: torch.randn((B, 1, 62, 2000))
```

### Configuration Parameters by Source

| Parameter | Phase | Source | Used By |
|-----------|-------|--------|---------|
| `n_eeg_channels` | 2 (preprocessing) | Computed from raw EEG data | Stored in normalization stats |
| `n_eeg_channels` | 3 (dataset) | Loaded from normalization stats | Populated to `dataset.extras` |
| `n_eeg_channels` | 5 (diffusion) | Retrieved from `dataset.extras` | Noise tensor shape |
| `sequence_length` | Config | Set in `[dataset] resolution` | Populated to `dataset.extras` |
| `sequence_length` | 5 (diffusion) | Retrieved from `dataset.extras` | Noise tensor shape |
| `in_channels` | Config | Set in `[model.extras]` | Noise tensor shape, adapter initialization |
| `input_channels` | Config (classification) | Set in `[model]` | Classification backbone initialization |

---

## Critical Files to Review Before Implementation

1. `signal_diffusion/data/base.py:182-183` - Verify writer mechanism
2. `signal_diffusion/data/seed.py` - Understand existing generate_examples pattern
3. `signal_diffusion/classification/backbones.py` - Understand backbone interface
4. `signal_diffusion/diffusion/models/dit.py:245` - Understand noise tensor creation
5. `signal_diffusion/diffusion/data.py:68-82` - Understand transform pipeline
6. `scripts/gen_weighted_dataset.py` - Understand meta-dataset generation
