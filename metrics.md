# Metrics Measurement and Visualization Overview

## Summary

This document describes all locations in the Signal Diffusion codebase where metrics are measured, how they're calculated, and how visualizations are generated. The codebase measures two primary types of metrics:

1. **Fidelity Metrics (KID/MMD)**: Measure distributional similarity between real and generated/reconstructed EEG spectrograms
2. **Classification Accuracy**: Measure performance of multi-task classifiers on real and synthetic datasets

---

## 1. KID/MMD Fidelity Metrics

### Overview
- **KID** (Kernel Inception Distance) and **MMD** (Maximum Mean Discrepancy) are used interchangeably
- KID is an unbiased estimator of MMD²
- Used to evaluate quality of generated spectrograms and VAE reconstructions
- All implementations use the `torch-fidelity` library (no custom MMD implementation)

### 1.1 Core Metrics Module

**File**: `signal_diffusion/metrics/fidelity.py`

**Calculation Method**:
- Library: `torch-fidelity`
- Kernel: RBF (Radial Basis Function)
- Kernel sigma: 10.0
- Default subset size: 1000 samples
- Feature extractors:
  - Primary: `dinov2-vit-l-14`
  - Alternative: `clip-vit-l-14`
- Also calculates PRC (Precision-Recall-Coverage) metrics

**Key Function**:
```python
def calculate_pair_metrics(real, generated, feature_extractor, config, ...):
    metrics = tfid.calculate_metrics(
        kid=True,
        kid_kernel="rbf",
        kid_kernel_rbf_sigma=10.0,
        kid_subset_size=config.kid_subset_size,
        ...
    )
    return metrics  # Contains kernel_inception_distance_mean and _std
```

### 1.2 Diffusion Training Evaluation

**File**: `signal_diffusion/diffusion/eval_utils.py`

**Purpose**: Compute KID during diffusion model training for validation

**Calculation Method**:
- Called during validation steps in training loop
- Compares generated samples against reference dataset
- Feature extractor: `dinov2-vit-s-14` (smaller variant for speed)
- Returns KID mean and standard deviation

**Key Function**:
```python
def compute_kid_score(generated_samples, ref_dataset):
    metrics = calculate_metrics(
        input1=generated_dataset,
        input2=ref_dataset,
        kid=True,
        kid_subset_size=min(num_gen, num_ref, 1000),
        feature_extractor="dinov2-vit-s-14",
        ...
    )
    return metrics["kernel_inception_distance_mean"], metrics["kernel_inception_distance_std"]
```

### 1.3 CLI Metrics Calculation

**File**: `metrics/calculate-metrics.py`

**Purpose**: Command-line tool for computing dataset fidelity metrics

**Usage**:
```bash
python metrics/calculate-metrics.py \
    -g /path/to/generated \
    -r /path/to/real \
    --kid-subset-size 1000 \
    --batch-size 64 \
    --output metrics.json
```

**Features**:
- Evaluates generated datasets against real datasets
- Supports multiple feature extractors
- Can cache extracted features for reuse
- Supports random subsets for evaluation
- Outputs results to JSON file

### 1.4 VAE Baseline Generation

**File**: `metrics/vae-only-dataset.py`

**Purpose**: Generate VAE reconstructions for KID baseline comparisons

**Workflow**:
1. Load dataset with VAE model
2. Encode images to latents
3. Decode latents back to images (reconstructions)
4. Save reconstructed parquet dataset (metadata preserved, images replaced)
5. Use `calculate-metrics.py` to compute KID vs original dataset

**Usage**:
```bash
python metrics/vae-only-dataset.py \
    -d /path/to/parquet-dataset-dir \
    -m stabilityai/stable-diffusion-3.5-medium \
    -o /path/to/output \
    --split train \
    --image-key image
```
Outputs ` /path/to/output/train.parquet` containing all original columns with reconstructed images.

### 1.5 KID/MMD Visualization

**File**: `summary-plots.py`

**Purpose**: Generate publication-quality plots of KID metrics

**Plot Generated**:
- **Output file**: `kdd-bar.png`
- **Plot type**: Bar chart with error bars
- **Y-axis label**: "MMD↓" (lower is better)
- **X-axis**: Different model/dataset conditions
- **Data source**: JSON files in `metrics/` directory:
  - `metrics/metrics-sdv1.json`
  - `metrics/metrics-dis.json`
  - `metrics/metrics-self.json`
  - `metrics/metrics-sdxl-vae.json`
  - `metrics/metrics-sdv1-vae.json`

**Visualization Settings**:
- Font: Tahoma, DejaVu Sans
- Font size: 15
- Compares real_eeg vs generated datasets (dis, sdv1, sdxl, self)
- Shows results for both DINOv2 and CLIP feature extractors

---

## 2. Classification Accuracy Metrics

### Overview
- Measures performance of multi-task classifiers
- Used for real vs synthetic dataset comparisons
- Supports both classification (accuracy) and regression (MAE/MSE) tasks

### 2.1 Core Training Loop

**File**: `signal_diffusion/training/classification.py`

**Classification Accuracy Calculation** (Lines 1079-1327):
```python
# For each batch:
preds = logits.argmax(dim=1)  # Get predicted class
correct = (preds == targets[name]).sum().item()  # Count correct
total = targets[name].size(0)

# Per epoch:
accuracy = total_correct / total_samples
```

**Regression Metrics** (for continuous targets like age):
- **MAE** (Mean Absolute Error): Average of absolute differences
- **MSE** (Mean Squared Error): Average of squared differences

```python
# Lines 1239-1245
abs_error = torch.abs(pred - target)
squared_error = (pred - target) ** 2

# Lines 1301-1302
mae = abs_error_sum / num_examples
mse = squared_error_sum / num_examples
```

**Weighted Mean Accuracy** (Lines 1360-1388):
- Function: `_compute_weighted_mean_accuracy()`
- Computes task-weighted average across all classification tasks
- Used for checkpoint selection and early stopping

**Metrics Logged to TensorBoard/WandB/MLflow**:
- Per-task accuracy: `{phase}/accuracy/{task_name}`
- Per-task MAE: `{phase}/mae/{task_name}`
- Per-task MSE: `{phase}/mse/{task_name}`
- Overall loss: `{phase}/loss`

### 2.2 Metrics Logger

**File**: `signal_diffusion/classification/metrics.py`

**Logging Backends**:
1. **TensorBoard**: Scalar metrics and histograms
2. **Weights & Biases (WandB)**: Metrics, config, and artifacts
3. **MLflow**: Metrics, hyperparameters, and model artifacts

**Logged Metrics** (Lines 131-142):
- Loss (overall and per-task)
- Accuracy per classification task
- MAE per regression task
- MSE per regression task
- Gradient norms

### 2.3 Hyperparameter Optimization

**File**: `hpo/classification_hpo.py`

**Purpose**: Use accuracy as optimization objective for HPO

**Features**:
- Uses Optuna for hyperparameter search
- Optimizes based on validation accuracy
- Supports pruning of unpromising trials
- Combined objective for mixed classification/regression tasks

### 2.4 Synthetic Dataset Generation

**File**: `scripts/generate_synthetic_dataset.py`

**Purpose**: Generate synthetic EEG spectrograms from pretrained diffusion models

**Features**:
- Generates data with controlled attributes (gender, health, age)
- Supports multiple conditioning modes:
  - Caption conditioning
  - Class conditioning
  - Multi-attribute conditioning
- Outputs Parquet dataset with images and metadata

**Note**: This script only generates data; it does NOT measure accuracy. To evaluate synthetic data quality:
1. Generate synthetic dataset with this script
2. Train classifier on synthetic data using `signal_diffusion/training/classification.py`
3. Evaluate on real test set
4. Compare results manually or with custom analysis

### 2.5 Real vs Synthetic Comparison Visualization

**File**: `summary-plots.py`

**Plot Generated**: Classifier accuracy matrix and bar charts

**Data Source**: `eeg_classification/bestmodels/accuracy.csv` (from older experiments, not in current codebase)

**Plots**:
1. **Accuracy Matrix Heatmap** (`classifier-accuracy-matrix.png`):
   - Shows cross-dataset performance
   - Rows: Training sets
   - Columns: Test sets

2. **Accuracy Bar Chart** (`classifier-accuracy-bar.png`):
   - Grouped bar chart comparing training sets
   - Training sets: `['real_eeg', 'real_eeg EMA', 'dis', 'dis EMA', 'sdv1', 'sdv1 EMA']`
   - Test sets: `['real_eeg', 'dis', 'sdv1']`

**Visualization Settings**:
- Font: Tahoma, DejaVu Sans
- Font size: 15
- Library: matplotlib

---

## 3. Additional Visualizations

### 3.1 STFT Histogram Analysis

**File**: `scripts/stft_histograms.py`

**Purpose**: Comprehensive STFT dB histogram analysis with quantization metrics

**Plots Generated**:

1. **STFT Histogram (dB scale)** (`stft_hist_{dataset}.png`):
   - Bar chart with overlays for quantized distributions
   - Y-axis: log scale
   - Shows raw dB and quantized versions for JS divergence, Wasserstein, PSNR metrics

2. **STFT Histogram (Linear magnitude)** (`stft_hist_linear_{dataset}.png`):
   - Both X and Y axes: log scale

3. **CDF Plots**:
   - `stft_cdf_{dataset}.png`: Raw cumulative distribution
   - `stft_cdf_quantized_{dataset}.png`: Comparison of raw vs quantized CDFs

4. **Metric Contour Landscapes** (`stft_{metric}_contour_{dataset}.png`):
   - Heatmaps showing optimization surfaces
   - Metrics: entropy, JS divergence, Wasserstein distance, PSNR
   - X-axis: Lower percentile bound
   - Y-axis: Upper percentile bound
   - Colormap: viridis

**Features**:
- Quantization analysis for uint8 dB bins
- Multiple metrics: entropy, JS divergence, Wasserstein distance, PSNR
- Percentile sweep analysis (0-49% lower, 95.5-100% upper)

### 3.2 Dataset Distribution Analysis

**File**: `scripts/subdataset_weighting_analysis.py`

**Purpose**: Analyze dataset composition and compute resampling weights

**Plots Generated**:

1. **Normalized Distribution by Category** (`{category}_distribution.png`):
   - Stacked histogram showing proportion by dataset
   - Categories: dataset, gender, health, age

2. **Raw Counts by Category** (`{category}_raw_counts.png`):
   - Stacked bar charts with annotated segment heights

3. **Subject Prevalence Pie Chart** (`subject_prevalence.png`):
   - Color-coded by dataset and gender
   - Shows sample count per subject

**Libraries**: matplotlib, seaborn

### 3.3 Weighted Dataset Utilities

**File**: `scripts/weighted_dataset_utils.py`

**Plot Generated**: Scaled sample weights

**Output**: `weights.png` or `{split}_weights.png`
- Line plot showing weight distribution across samples
- Y-axis: Weight relative to minimum

### 3.4 Image Grid Generation

**File**: `scripts/generate_image_grid.py`

**Purpose**: Generate grids of samples from trained diffusion models

**Output**:
- `generated_grid.jpg` (configurable filename)
- Grid layout: reference images in first row, generated samples in subsequent rows
- Supports class-conditioned generation

### 3.5 Jupyter Notebooks

#### `data-vis.ipynb`
**Visualizations**:
- EEG timeseries plots using MNE
- Power spectral density plots
- VAE reconstruction comparisons
- Saves: `parkinsons-snippet.png`, `seed-snippet.png`, `eeg-spec.png`, `fm-spec.png`
- VAE reconstruction outputs: `sdxl_vae_orig_{n}.jpg`, `sdxl_vae_recon_{n}.jpg`

#### `noise-floor.ipynb`
**Visualizations**:
- Average power spectral density across dataset
- Noise floor estimation with augmentation noise overlay
- Frequency range: 0-62.5 Hz
- dB scale plots with grid

---

## 4. Workflow Summaries

### 4.1 KID/MMD Evaluation Workflow

1. **Generate data** (one of):
   - Generate synthetic spectrograms: `scripts/generate_synthetic_dataset.py`
   - Generate VAE reconstructions: `metrics/vae-only-dataset.py`

2. **Calculate metrics**:
   ```bash
   python metrics/calculate-metrics.py \
       -g /path/to/generated \
       -r /path/to/real \
       --output metrics/metrics-{name}.json
   ```

3. **Visualize results**:
   ```bash
   python summary-plots.py
   # Generates: kdd-bar.png
   ```

### 4.2 Classification Accuracy Workflow

1. **Generate synthetic dataset** (if needed):
   ```bash
   uv run python scripts/generate_synthetic_dataset.py \
       --checkpoint /path/to/model \
       --output /path/to/synthetic \
       --num-samples 5000
   ```

2. **Train classifier on synthetic data**:
   ```bash
   uv run python -m signal_diffusion.training.classification \
       config/classification/synthetic.toml
   ```
   (Update config to point to synthetic dataset path)

3. **Evaluate on real test set**:
   - Handled automatically during training
   - Metrics logged to TensorBoard/WandB/MLflow

4. **Compare results**:
   - Manual comparison of logged metrics
   - Or use `summary-plots.py` (requires CSV with accuracy data)

---

## 5. Key Libraries and Tools

| Library/Tool | Purpose | Files |
|--------------|---------|-------|
| **torch-fidelity** | KID/MMD computation | `fidelity.py`, `calculate-metrics.py`, `eval_utils.py` |
| **matplotlib** | Primary plotting library | All visualization scripts |
| **seaborn** | Statistical visualizations | `subdataset_weighting_analysis.py` |
| **TensorBoard** | Training metrics logging | `classification/metrics.py`, `training/` |
| **WandB** | Experiment tracking | `classification/metrics.py`, `training/` |
| **MLflow** | Model and metrics tracking | `classification/metrics.py`, `training/` |
| **Optuna** | Hyperparameter optimization | `hpo/classification_hpo.py` |

---

## 6. Critical Files Reference

### Metrics Calculation
- `signal_diffusion/metrics/fidelity.py`: Core fidelity metrics (KID, PRC)
- `signal_diffusion/diffusion/eval_utils.py`: Diffusion training evaluation
- `metrics/calculate-metrics.py`: CLI for dataset metrics
- `metrics/vae-only-dataset.py`: VAE reconstruction dataset generator
- `signal_diffusion/training/classification.py`: Classification accuracy calculation
- `signal_diffusion/classification/metrics.py`: Metrics logging infrastructure

### Visualization
- `summary-plots.py`: Publication-quality KID and accuracy plots
- `scripts/stft_histograms.py`: STFT distribution analysis
- `scripts/subdataset_weighting_analysis.py`: Dataset composition analysis
- `scripts/weighted_dataset_utils.py`: Sample weight visualization
- `scripts/generate_image_grid.py`: Sample grid generation
- `data-vis.ipynb`: Interactive VAE and EEG visualization
- `noise-floor.ipynb`: Power spectral density analysis

### Data Generation
- `scripts/generate_synthetic_dataset.py`: Synthetic spectrogram generation
- `scripts/gen_weighted_dataset.py`: Weighted dataset generation

---

## 7. Current Gaps

Based on the exploration, the codebase currently lacks:

1. **Automated real vs synthetic classifier evaluation script**: The workflow exists in pieces (synthetic generation + classifier training), but there's no single script that:
   - Trains classifiers on synthetic data
   - Evaluates on real test sets
   - Compares results systematically
   - Generates comparison visualizations

2. **Current accuracy CSV**: The file `eeg_classification/bestmodels/accuracy.csv` referenced in `summary-plots.py` doesn't exist in the current codebase (appears to be from older experiments)

3. **Integrated evaluation pipeline**: No script that runs the full pipeline:
   - Generate synthetic → Calculate KID → Train classifier → Compare accuracy
