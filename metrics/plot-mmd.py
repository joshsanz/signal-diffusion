# MMD plots for generative results
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

plt.rcParams['font.size'] = 15
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans',
                                   'Lucida Grande', 'Verdana']

# Find all metric JSON files in the metrics directory
metrics_dir = Path(__file__).parent
metrics_files = sorted(metrics_dir.glob('*-metrics.json'))
metrics_files.extend(sorted(metrics_dir.glob('synth-train.json')))

print(f"Found {len(metrics_files)} metric files:")
for mf in metrics_files:
    print(f"  - {mf.name}")

# Load the generative metrics
gen_metrics = pd.DataFrame()
for mf in metrics_files:
    df = pd.read_json(mf, orient='index')
    # Extract generator name from filename
    # e.g., "sdxl-vae-metrics.json" -> "sdxl-vae"
    # e.g., "synth-train.json" -> "synth-train"
    generator_name = mf.stem.replace('-metrics', '')
    df["Generator"] = [generator_name] * len(df)
    gen_metrics = pd.concat([gen_metrics, df])

gen_metrics = gen_metrics.reset_index(names="Featurizer")

# Replace featurizer names with prettier versions
gen_metrics = gen_metrics.replace("dinov2-vit-l-14", "DINOv2")
gen_metrics = gen_metrics.replace("clip-vit-l-14", "CLIP")

gen_metrics.set_index(["Generator", "Featurizer"], inplace=True)
print("\nLoaded metrics:")
print(gen_metrics)

# Plot MMD metrics
fig, ax = plt.subplots(figsize=(10, 6))
unstacked = gen_metrics.unstack()

unstacked.plot.bar(
    y="kernel_inception_distance_mean",
    ax=ax,
    yerr="kernel_inception_distance_std",
    capsize=5,
    legend=False
)

# Recolor bars: C1 for VAE, C0 for others
num_featurizers = len(gen_metrics.index.get_level_values('Featurizer').unique())
for i, patch in enumerate(ax.patches):
    gen_idx = i // num_featurizers
    generator_name = unstacked.index[gen_idx]
    if 'vae' in str(generator_name).lower():
        patch.set_color('C1')
    else:
        patch.set_color('C0')

plt.xticks(rotation=45, ha='right')
plt.ylabel(r"MMD$\downarrow$")
plt.xlabel("Generator")
plt.title("Maximal Mean Discrepancy")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(metrics_dir / 'mmd-bar.png', dpi=300)
print(f"\nSaved plot to {metrics_dir / 'mmd-bar.png'}")
plt.close()
