# Diffusion Configuration Files

Quick reference for running diffusion model training on EEG spectrograms and timeseries data.

## Available Models × Datasets

### DiT (Diffusion Transformer)
```bash
# Spectrogram variants (1-3 channels, 64×64)
uv run python -m signal_diffusion.training.diffusion config/diffusion/dit-db-only.toml --output-dir runs/dit-db-only
uv run python -m signal_diffusion.training.diffusion config/diffusion/dit-db-iq.toml --output-dir runs/dit-db-iq
uv run python -m signal_diffusion.training.diffusion config/diffusion/dit-db-polar.toml --output-dir runs/dit-db-polar

# Timeseries variant (1 channel, 2048 sequence)
uv run python -m signal_diffusion.training.diffusion config/diffusion/dit-timeseries.toml --output-dir runs/dit-timeseries
```

### Hourglass (Hierarchical Transformer)
```bash
# Spectrogram variants (1-3 channels, 64×64)
uv run python -m signal_diffusion.training.diffusion config/diffusion/hourglass-db-only.toml --output-dir runs/hourglass-db-only
uv run python -m signal_diffusion.training.diffusion config/diffusion/hourglass-db-iq.toml --output-dir runs/hourglass-db-iq
uv run python -m signal_diffusion.training.diffusion config/diffusion/hourglass-db-polar.toml --output-dir runs/hourglass-db-polar

# Timeseries variant (1 channel, 2048 sequence)
uv run python -m signal_diffusion.training.diffusion config/diffusion/hourglass-timeseries.toml --output-dir runs/hourglass-timeseries
```

### LocalMamba (State Space Model with Vision)
```bash
# Spectrogram variants (1-3 channels, 64×64)
uv run python -m signal_diffusion.training.diffusion config/diffusion/localmamba-db-only.toml --output-dir runs/localmamba-db-only
uv run python -m signal_diffusion.training.diffusion config/diffusion/localmamba-db-iq.toml --output-dir runs/localmamba-db-iq
uv run python -m signal_diffusion.training.diffusion config/diffusion/localmamba-db-polar.toml --output-dir runs/localmamba-db-polar

# Timeseries variant (1 channel, 2048 sequence)
uv run python -m signal_diffusion.training.diffusion config/diffusion/localmamba-timeseries.toml --output-dir runs/localmamba-timeseries
```

### Stable Diffusion 3.5 Medium
```bash
# Spectrogram variants (1-3 channels, 64×64, with VAE latent space)
uv run python -m signal_diffusion.training.diffusion config/diffusion/sd35-db-only.toml --output-dir runs/sd35-db-only
uv run python -m signal_diffusion.training.diffusion config/diffusion/sd35-db-iq.toml --output-dir runs/sd35-db-iq
uv run python -m signal_diffusion.training.diffusion config/diffusion/sd35-db-polar.toml --output-dir runs/sd35-db-polar

# Timeseries variant (1 channel, 2048 sequence, with VAE latent space)
uv run python -m signal_diffusion.training.diffusion config/diffusion/sd35-timeseries.toml --output-dir runs/sd35-timeseries
```

## Dataset Overview

| Name | Path | Resolution | Channels | Type |
|------|------|------------|----------|------|
| **db-only** | `/data/data/signal-diffusion/processed/reweighted_meta_dataset_log_n2048_fs125` | 64×64 | 1 (dB) | spectrogram |
| **db-iq** | `/data/data/signal-diffusion/processed-iq/reweighted_meta_dataset_log_n2048_fs125` | 64×64 | 3 (dB, I, Q) | spectrogram |
| **db-polar** | `/data/data/signal-diffusion/processed-polar/reweighted_meta_dataset_log_n2048_fs125` | 64×64 | 3 (dB, mag, phase) | spectrogram |
| **timeseries** | `/data/data/signal-diffusion/processed/reweighted_timeseries_meta_dataset_n2048_fs125` | 2048 | 1 (20 EEG channels) | timeseries |

## Model Key Parameters

| Model | Layers/Depths | Width/Dims | LR | MixP | Batch |
|-------|---|---|---|---|---|
| **DiT** | 12 | 72×16 attn heads | 1e-4 | bf16 | 32 |
| **Hourglass** | [2,2,3] | [64,128,256] | 1e-4 | bf16 | 64 |
| **LocalMamba** | [2,2,6,2] | [64,128,256,512] | 3e-4 | bf16 | 64 |
| **SD3.5** | transformer | 16 (latent) | 5e-5 | bf16 | 16 |

## Conditional Models (Future)

When ready to add conditioning (captions, gender/health/age):

```bash
# Example with caption conditioning for SD 3.5
# Edit config: conditioning = "caption", caption_column = "description"
uv run python -m signal_diffusion.training.diffusion config/diffusion/sd35-db-only.toml \
    --output-dir runs/sd35-db-only-caption

# Example with class conditioning for Hourglass
# Edit config: conditioning = "classes", num_classes = 5
uv run python -m signal_diffusion.training.diffusion config/diffusion/hourglass-db-only.toml \
    --output-dir runs/hourglass-db-only-class
```

## Configuration Customization

Common edits for your needs:

```toml
# Change batch size (limited by GPU memory)
[dataset]
batch_size = 32  # Reduce for smaller GPU, increase for larger GPU

# Change number of training epochs
[training]
epochs = 50  # Default varies by config (1-200)

# Enable CFG dropout for classifier-free guidance preparation
[model.extras]
cfg_dropout = 0.1  # 0.0 = disabled, 0.1-0.2 = typical

# Change checkpoint frequency
[training]
checkpoint_interval = 500  # Save every N steps
eval_num_steps = 500  # Evaluate every N steps

# Adjust mixed precision (memory/speed tradeoff)
[training]
mixed_precision = "bf16"  # Options: "no", "fp16", "bf16"
```

## Resuming Training

```bash
# Resume from checkpoint
uv run python -m signal_diffusion.training.diffusion config/diffusion/dit-db-only.toml \
    --output-dir runs/dit-db-only \
    --resume-from-checkpoint runs/dit-db-only/checkpoint-1000
```

## Monitoring Training

All configs use TensorBoard logging. View training progress:

```bash
tensorboard --logdir runs/diffusion/tensorboard/
```

Then open http://localhost:6006 in browser.

## Notes

- **Memory Requirements**: SD3.5 and LocalMamba (timeseries) are more memory-intensive
- **Training Time**: DiT (~few hours), Hourglass (~4-8 hours), LocalMamba (~8-12 hours), SD3.5 (~12-24 hours)
- **Dataset Splits**: All use "train" split for training, "validation" for evaluation
- **Channel Handling**: 1-channel inputs automatically broadcast to model input channels if needed

For detailed configuration documentation, see [CONFIG_SUMMARY.md](../CONFIG_SUMMARY.md)
