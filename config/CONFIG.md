# Configuration File Documentation

This document describes the configuration file layout and all available settings for Signal Diffusion training.

## Table of Contents

1. [Configuration System Overview](#configuration-system-overview)
2. [File Structure and Resolution](#file-structure-and-resolution)
3. [Common Settings](#common-settings)
4. [Classification Settings](#classification-settings)
5. [Diffusion Settings](#diffusion-settings)
6. [Complete Examples](#complete-examples)

---

## Configuration System Overview

Signal Diffusion uses TOML configuration files organized into two layers:

1. **Global settings** (`config/default.toml`): Dataset paths, output directories, model IDs
2. **Training configs** (`config/{classification,diffusion}/*.toml`): Task-specific hyperparameters

Training configs reference the global settings via the `[settings]` section.

---

## File Structure and Resolution

### Settings Resolution Order

The global settings file is resolved in this order:

1. Explicit path passed to `load_settings(path)`
2. `SIGNAL_DIFFUSION_CONFIG` environment variable
3. `config/default.toml` (default)

### Config File References

Training configs specify which settings file to use:

```toml
[settings]
config = "config/default.toml"
trainer = "diffusion"  # or "classification"
```

---

## Common Settings

These settings appear in `config/default.toml` and apply to all training runs.

### `[hf_models]` Section

HuggingFace model identifiers for shared resources.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `stable_diffusion_model_id` | string | No | HuggingFace model ID for Stable Diffusion VAE and text encoders (default: `"stabilityai/stable-diffusion-3.5-medium"`) |

**Example:**
```toml
[hf_models]
stable_diffusion_model_id = "stabilityai/stable-diffusion-3.5-medium"
```

### `[data]` Section

Global data configuration.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `root` | string (path) | Yes | Root directory for raw datasets |
| `output_root` | string (path) | Yes | Root directory for preprocessed outputs |
| `max_sampling_weight` | float | No | Maximum weight for weighted sampling (default: null) |
| `output_type` | string | Yes | Output format: `"db-only"`, `"db-iq"`, or `"db-polar"` |
| `data_type` | string | Yes | Data type: `"spectrogram"` or `"timeseries"` |

**Example:**
```toml
[data]
root = "/data/data/signal-diffusion"
output_root = "/data/data/signal-diffusion/processed"
output_type = "db-only"
data_type = "spectrogram"
max_sampling_weight = 5.0
```

### `[datasets.*]` Sections

Per-dataset configuration. Each dataset gets its own section: `[datasets.seed]`, `[datasets.mit]`, etc.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `root` | string (path) | Yes | Relative path from `data.root` to dataset directory |
| `output` | string (path) | Yes | Relative path from `data.output_root` for spectrograms |
| `timeseries_output` | string (path) | No | Relative path for timeseries outputs (if `data_type="timeseries"`) |
| `min_db` | float | No | Minimum dB value for spectrogram normalization |
| `max_db` | float | No | Maximum dB value for spectrogram normalization |

**Example:**
```toml
[datasets.seed]
root = "seed"
output = "seed/stfts"
timeseries_output = "seed/timeseries"
min_db = -119.0125
max_db = -5.0875

[datasets.parkinsons]
root = "parkinsons"
output = "parkinsons/stfts"
min_db = -120.0675
max_db = 2.9085
```

---

## Classification Settings

Configuration for multi-task classification training (`config/classification/*.toml`).

### `[settings]` Section

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `config` | string (path) | Yes | Path to global settings file |
| `trainer` | string | Yes | Must be `"classification"` |

### `[data]` Section (Optional Overrides)

Optional overrides for global data settings.

| Parameter | Type | Description |
|-----------|------|-------------|
| `output_type` | string | Override output format: `"db-only"`, `"db-iq"`, or `"db-polar"` |
| `data_type` | string | Override data type: `"spectrogram"` or `"timeseries"` |

### `[dataset]` Section

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | string | Yes | - | Dataset name or path to preprocessed dataset |
| `tasks` | list[string] | Yes | - | List of task names to train on (see available tasks below) |
| `train_split` | string | No | `"train"` | Training split name |
| `val_split` | string | No | `"val"` | Validation split name |
| `batch_size` | int | No | `32` | Training batch size |
| `num_workers` | int | No | `4` | DataLoader workers |
| `pin_memory` | bool | No | `true` | Pin memory for GPU transfer |
| `shuffle` | bool | No | `true` | Shuffle training data |

**Available Tasks by Dataset:**

| Dataset | Available Tasks |
|---------|-----------------|
| **seed** | `emotion` (5 classes), `gender`, `age`, `health` |
| **mit** | `seizure`, `gender`, `age` |
| **parkinsons** | `health`, `parkinsons_condition`, `gender`, `age` |
| **math** | `math_activity`, `gender`, `age` |

### `[model]` Section

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `backbone` | string | Yes | - | Backbone architecture: `"cnn_light"`, `"cnn"`, `"transformer"` |
| `input_channels` | int | Yes | - | Number of input channels (1 for db-only, 3 for db-iq/db-polar) |
| `embedding_dim` | int | No | `256` | Embedding dimension output by backbone |
| `dropout` | float | No | `0.3` | Dropout probability |
| `activation` | string | No | `"gelu"` | Activation function |
| `depth` | int | No | `3` | Number of layers/blocks |
| `layer_repeats` | int | No | `2` | Repetitions per layer/block |

### `[optimizer]` Section

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | string | No | `"adamw"` | Optimizer type: `"adamw"`, `"adam"`, `"sgd"` |
| `learning_rate` | float | No | `3e-4` | Learning rate |
| `weight_decay` | float | No | `1e-4` | Weight decay (L2 regularization) |
| `betas` | list[float, float] | No | `[0.9, 0.999]` | Adam beta parameters |

### `[scheduler]` Section

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | string | No | `"constant"` | Scheduler type: `"constant"`, `"linear"`, `"cosine"` |
| `warmup_steps` | int | No | `0` | Learning rate warmup steps |

**`[scheduler.kwargs]` Subsection:**

Additional scheduler-specific parameters (optional).

| Parameter | Scheduler | Description |
|-----------|-----------|-------------|
| `min_lr_ratio` | linear, cosine | Minimum learning rate as ratio of base LR |
| `num_cycles` | cosine | Number of cosine cycles |

### `[training]` Section

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `epochs` | int | No | `25` | Number of training epochs |
| `max_steps` | int | No | `-1` | Maximum training steps (-1 for unlimited) |
| `clip_grad_norm` | float | No | `1.0` | Gradient clipping max norm (null to disable) |
| `device` | string | No | null | Device to use (auto-detected if null) |
| `log_every_batches` | int | No | `10` | Log metrics every N batches |
| `eval_strategy` | string | No | `"epoch"` | Evaluation strategy: `"epoch"` or `"steps"` |
| `eval_steps` | int | No | null | Evaluate every N steps (if `eval_strategy="steps"`) |
| `output_dir` | string (path) | No | null | Output directory for checkpoints |
| `log_dir` | string (path) | No | null | Directory for logs |
| `tensorboard` | bool | No | `false` | Enable TensorBoard logging |
| `wandb_project` | string | No | null | Weights & Biases project name |
| `wandb_entity` | string | No | null | Weights & Biases entity/username |
| `wandb_tags` | list[string] | No | `[]` | W&B tags for the run |
| `run_name` | string | No | null | Run name for logging (auto-generated if null) |
| `checkpoint_total_limit` | int | No | null | Maximum number of checkpoints to keep |
| `checkpoint_strategy` | string | No | `"epoch"` | Checkpoint strategy: `"epoch"` or `"steps"` |
| `checkpoint_steps` | int | No | null | Save checkpoint every N steps (if `checkpoint_strategy="steps"`) |
| `use_amp` | bool | No | `false` | Use automatic mixed precision |
| `metrics_summary_path` | string (path) | No | null | Path to save metrics summary CSV |
| `max_best_checkpoints` | int | No | `1` | Number of best checkpoints to keep |
| `early_stopping` | bool | No | `false` | Enable early stopping |
| `early_stopping_patience` | int | No | `5` | Early stopping patience (epochs) |
| `compile_model` | bool | No | `true` | Use `torch.compile()` |
| `compile_mode` | string | No | `"default"` | Compile mode: `"default"`, `"reduce-overhead"`, `"max-autotune"` |
| `swa_enabled` | bool | No | `false` | Enable Stochastic Weight Averaging |
| `swa_extra_ratio` | float | No | `0.34` | SWA extra epochs ratio |
| `swa_lr_frac` | float | No | `0.25` | SWA learning rate fraction |

**`[training.task_weights]` Subsection:**

Per-task loss weights. Defaults to 1.0 for all tasks.

**Example:**
```toml
[training.task_weights]
emotion = 1.0
gender = 2.0
age = 0.05  # Lower weight for regression task to balance with classification
```

---

## Diffusion Settings

Configuration for diffusion model training (`config/diffusion/*.toml`).

### `[settings]` Section

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `config` | string (path) | Yes | Path to global settings file |
| `trainer` | string | Yes | Must be `"diffusion"` |

### `[data]` Section (Optional Overrides)

Optional overrides for global data settings.

| Parameter | Type | Description |
|-----------|------|-------------|
| `output_type` | string | Override output format: `"db-only"`, `"db-iq"`, or `"db-polar"` |
| `data_type` | string | Override data type: `"spectrogram"` or `"timeseries"` |

### `[dataset]` Section

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | string | Yes | - | Path to preprocessed dataset directory |
| `batch_size` | int | No | `4` | Training batch size |
| `eval_batch_size` | int | No | null | Evaluation batch size (defaults to `batch_size`) |
| `num_workers` | int | No | `4` | DataLoader workers |
| `resolution` | int | No | `256` | Image resolution (must match preprocessed data) |
| `train_split` | string | No | `"train"` | Training split name |
| `val_split` | string | No | null | Validation split name |
| `cache_dir` | string (path) | No | null | Cache directory for datasets |
| `center_crop` | bool | No | `false` | Center crop images |
| `random_flip` | bool | No | `false` | Random horizontal flip augmentation |
| `image_column` | string | No | `"image"` | Column name for images |
| `caption_column` | string | No | `"text"` | Column name for captions (required if `model.conditioning="caption"`) |
| `class_column` | string | No | null | Column name for class labels (required if `model.conditioning="classes"`) |
| `gender_column` | string | No | `"gender"` | Column name for gender (for `"gend_hlth_age"` conditioning) |
| `health_column` | string | No | `"health"` | Column name for health status (for `"gend_hlth_age"` conditioning) |
| `age_column` | string | No | `"age"` | Column name for age (for `"gend_hlth_age"` conditioning) |
| `num_classes` | int | No | `0` | Number of classes (required if `model.conditioning="classes"`) |
| `dataset_type` | string | No | `"auto"` | Dataset type (auto-detected) |
| `max_train_samples` | int | No | null | Maximum training samples to use |
| `max_eval_samples` | int | No | null | Maximum evaluation samples to use |

### `[model]` Section

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | string | Yes | - | Model: `"dit"`, `"hourglass"`, `"localmamba"`, `"stable-diffusion"`, `"stable-diffusion-35"` |
| `conditioning` | string | No | null | Conditioning: `"none"`, `"caption"`, `"classes"`, `"gend_hlth_age"` |
| `pretrained` | string | No | null | Pretrained model path or HuggingFace ID |
| `revision` | string | No | null | Model revision/branch |
| `sample_size` | int | No | null | Latent sample size |
| `vae_tiling` | bool | No | `false` | Enable VAE tiling for large images |

**`[model.lora]` Subsection:**

LoRA adapter configuration (optional).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Enable LoRA training |
| `rank` | int | `4` | LoRA rank |
| `alpha` | float | `4.0` | LoRA alpha scaling |
| `dropout` | float | `0.0` | LoRA dropout |
| `target_modules` | list[string] | `["to_q", "to_k", "to_v", "to_out.0"]` | Modules to apply LoRA |
| `bias` | string | `"none"` | Bias handling: `"none"`, `"all"`, `"lora_only"` |
| `scaling` | float | null | Custom LoRA scaling factor |

**`[model.extras]` Subsection:**

Model-specific hyperparameters (varies by model).

**Common Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `latent_space` | bool | `false` | Use VAE latent space training |
| `vae` | string | null | VAE model ID (required if `latent_space=true`) |
| `in_channels` | int | auto | Input channels (3 for RGB, 4/16 for latents) |
| `out_channels` | int | auto | Output channels (same as `in_channels`) |
| `cfg_dropout` | float | `0.0` | Classifier-free guidance dropout probability |

**DiT-Specific:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_attention_heads` | int | Number of attention heads |
| `attention_head_dim` | int | Attention head dimension |
| `num_layers` | int | Number of transformer layers |
| `patch_size` | int or list | Patch size (e.g., `2` or `[1, 8]` for timeseries) |
| `num_classes` | int | Number of classes (for class conditioning) |

**Hourglass/LocalMamba-Specific:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `block_out_channels` | list[int] | Channel dimensions per level (e.g., `[224, 448, 672, 896]`) |
| `num_res_blocks` | int | Residual blocks per level |
| `num_attention_heads` | int | Number of attention heads |
| `mapping_cond` | int | Caption embedding dimension (2048 for caption conditioning) |
| `d_state` | int | Mamba state dimension (LocalMamba only) |
| `d_conv` | int | Mamba convolution dimension (LocalMamba only) |
| `expand` | int | Mamba expansion factor (LocalMamba only) |

**Stable Diffusion 3.5-Specific:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_layers` | int | Number of transformer layers |
| `attention_head_dim` | int | Attention head dimension |
| `num_attention_heads` | int | Number of attention heads |
| `joint_attention_dim` | int | Joint attention dimension |
| `caption_projection_dim` | int | Caption projection dimension |
| `pooled_projection_dim` | int | Pooled projection dimension |

### `[objective]` Section

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prediction_type` | string | No | `"epsilon"` | What model predicts: `"epsilon"`, `"sample"`, `"vector_field"` |
| `scheduler` | string | No | `"ddim"` | Noise scheduler type |
| `num_timesteps` | int | No | `1000` | Timesteps for flow matching |

### `[optimizer]` Section

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | string | No | `"adamw"` | Optimizer type: `"adamw"`, `"adam"`, `"sgd"` |
| `learning_rate` | float | No | `1e-4` | Learning rate |
| `weight_decay` | float | No | `1e-2` | Weight decay |
| `betas` | list[float, float] | No | `[0.9, 0.999]` | Adam beta parameters |
| `eps` | float | No | `1e-8` | Adam epsilon |

### `[scheduler]` Section

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `name` | string | No | `"constant"` | Scheduler: `"constant"`, `"constant_with_warmup"`, `"linear"`, `"cosine"`, `"polynomial"` |
| `warmup_steps` | int | No | `0` | Warmup steps |

### `[training]` Section

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `seed` | int | No | `42` | Random seed |
| `output_dir` | string (path) | No | null | Output directory for checkpoints |
| `mixed_precision` | string | No | `"bf16"` | Mixed precision: `"no"`, `"fp16"`, `"bf16"` |
| `gradient_checkpointing` | bool | No | `false` | Enable gradient checkpointing |
| `gradient_accumulation_steps` | int | No | `1` | Gradient accumulation steps |
| `epochs` | int | No | `1` | Number of training epochs |
| `max_train_steps` | int | No | null | Maximum training steps (overrides epochs) |
| `log_every_steps` | int | No | `10` | Log metrics every N steps |
| `checkpoint_interval` | int | No | null | Save checkpoint every N steps |
| `checkpoint_total_limit` | int | No | null | Maximum checkpoints to keep |
| `resume` | string (path) | No | null | Checkpoint path to resume from |
| `gradient_clip_norm` | float | No | `1.0` | Gradient clipping max norm |
| `ema_decay` | float | No | `0.999` | EMA decay rate |
| `ema_power` | float | No | `0.75` | EMA power for warmup |
| `ema_inv_gamma` | float | No | `1.0` | EMA inverse gamma |
| `ema_update_after_step` | int | No | `5000` | EMA update after step |
| `ema_use_ema_warmup` | bool | No | `true` | Use EMA warmup |
| `allow_tf32` | bool | No | `true` | Allow TF32 on Ampere GPUs |
| `snr_gamma` | float | No | null | SNR gamma for loss weighting |
| `eval_num_examples` | int | No | `0` | Number of examples to generate during eval |
| `eval_mmd_samples` | int | No | `0` | Number of samples for MMD metric |
| `eval_mmd_fallback_ntrain` | int | No | `0` | Fallback training samples for MMD |
| `eval_strategy` | string | No | `"epoch"` | Evaluation strategy: `"epoch"` or `"steps"` |
| `eval_num_steps` | int | No | `0` | Evaluate every N steps (if `eval_strategy="steps"`) |
| `eval_batch_size` | int | No | `0` | Evaluation batch size |
| `initial_eval` | bool | No | `true` | Run evaluation before training |

### `[logging]` Section

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `tensorboard` | bool | No | `false` | Enable TensorBoard logging |
| `log_dir` | string (path) | No | null | Directory for logs |
| `wandb_project` | string | No | null | Weights & Biases project name |
| `wandb_entity` | string | No | null | W&B entity/username |
| `run_name` | string | No | null | Run name (auto-generated if null) |

### `[inference]` Section

Sampling configuration for validation and generation.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `denoising_steps` | int | No | `50` | Number of denoising steps |
| `cfg_scale` | float | No | `7.5` | Classifier-free guidance scale |

---

## Complete Examples

### Classification Example

**config/classification/baseline.toml:**
```toml
[settings]
config = "config/default.toml"
trainer = "classification"

[data]
output_type = "db-only"
data_type = "spectrogram"

[dataset]
name = "/data/data/signal-diffusion/processed/reweighted_meta_dataset_log_n2048_fs125"
train_split = "train"
val_split = "validation"
tasks = ["gender", "health", "age"]
batch_size = 32
num_workers = 2
shuffle = true

[model]
backbone = "cnn_light"
input_channels = 1
embedding_dim = 384
dropout = 0.43
depth = 3
layer_repeats = 3

[optimizer]
name = "adamw"
learning_rate = 0.0002
weight_decay = 0.009
betas = [0.9, 0.995]

[scheduler]
name = "cosine"
warmup_steps = 100

[scheduler.kwargs]
min_lr_ratio = 0.03

[training]
epochs = 15
max_steps = -1
clip_grad_norm = 1.0
log_every_batches = 5
tensorboard = true
eval_strategy = "steps"
eval_steps = 100
checkpoint_strategy = "steps"
checkpoint_steps = 100
checkpoint_total_limit = 3
compile_model = true
compile_mode = "default"
swa_enabled = false

[training.task_weights]
gender = 2.0
health = 1.0
age = 0.05
```

### Diffusion Example (DiT)

**config/diffusion/baseline.toml:**
```toml
[settings]
config = "config/default.toml"
trainer = "diffusion"

[data]
output_type = "db-only"
data_type = "spectrogram"

[dataset]
name = "/data/data/signal-diffusion/processed/reweighted_meta_dataset_log_n2048_fs125"
train_split = "train"
val_split = "val"
batch_size = 32
num_workers = 4
resolution = 64
center_crop = false
random_flip = false
image_column = "image"
caption_column = ""
num_classes = 0

[model]
name = "dit"
sample_size = 32
conditioning = "none"

[model.extras]
num_attention_heads = 16
attention_head_dim = 72
in_channels = 3
out_channels = 3
num_layers = 12
patch_size = 2
latent_space = false
num_classes = 0
cfg_dropout = 0.1

[model.lora]
enabled = false

[objective]
prediction_type = "vector_field"
flow_match_timesteps = 1000

[optimizer]
name = "adamw"
learning_rate = 1e-4
weight_decay = 1e-4
betas = [0.9, 0.999]

[scheduler]
name = "constant_with_warmup"
warmup_steps = 100

[training]
seed = 42
epochs = 1
mixed_precision = "bf16"
gradient_checkpointing = false
gradient_accumulation_steps = 1
gradient_clip_norm = 1.0
snr_gamma = 5.0
log_every_steps = 10
checkpoint_interval = 1000
checkpoint_total_limit = 3
eval_strategy = "steps"
eval_num_steps = 100
eval_num_examples = 8
eval_mmd_samples = 100
eval_mmd_fallback_ntrain = 1000
ema_use_ema_warmup = true
ema_inv_gamma = 1
ema_power = 0.75
ema_decay = 0.999
ema_update_after_step = 1000

[logging]
tensorboard = true
run_name = "baseline"
log_dir = "runs/diffusion/tensorboard/base"

[inference]
denoising_steps = 25
cfg_scale = 1.0
```

---

## Notes

1. **Path Resolution**:
   - Paths in `config/default.toml` are absolute or relative to the config file
   - Dataset paths in `[datasets.*]` are relative to `data.root` and `data.output_root`
   - Paths starting with `~/` are expanded to the user's home directory

2. **Auto-Configuration**:
   - `model.extras.in_channels`: Auto-detected from VAE config when `latent_space=true`
   - `scheduler.num_training_steps`: Calculated from epochs and dataset size (for diffusion)
   - `dataset.eval_batch_size`: Defaults to `batch_size` if not specified

3. **Conditioning Requirements**:
   - `conditioning="caption"`: Requires `dataset.caption_column`
   - `conditioning="classes"`: Requires `dataset.num_classes` and optionally `dataset.class_column`
   - `conditioning="gend_hlth_age"`: Requires `gender_column`, `health_column`, and `age_column`
   - `conditioning="none"` or null: No additional requirements

4. **Latent Space Training**:
   - SD 3.5 VAE: 16 channels, 8× spatial compression
   - SD 1.5 VAE: 4 channels, 8× spatial compression
   - Set `model.extras.latent_space=true` and `model.extras.vae="model-id"`

5. **Output Types**:
   - `"db-only"`: 1-channel grayscale (magnitude only)
   - `"db-iq"`: 3-channel (magnitude, I, Q)
   - `"db-polar"`: 3-channel (magnitude, phase-cos, phase-sin)

6. **Data Overrides**:
   - Training configs can override `output_type` and `data_type` in their `[data]` section
   - This allows using different data formats without modifying `config/default.toml`
