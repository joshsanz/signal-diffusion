
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from signal_diffusion.config import load_settings
from signal_diffusion.data.parkinsons import ParkinsonsDataset
from signal_diffusion.data.mit import MITDataset
from signal_diffusion.data.seed import SEEDDataset
from signal_diffusion.data.math import MathDataset

def load_metadata():
    """Loads metadata from all datasets into a single DataFrame."""
    settings = load_settings("config/default-mac.toml")
    
    parkinsons_df = ParkinsonsDataset(settings, split="train").metadata
    parkinsons_df["dataset"] = "parkinsons"
    
    math_df = MathDataset(settings, split="train").metadata
    math_df["dataset"] = "math"
    
    return pd.concat([parkinsons_df, math_df])

def plot_distribution(df, category, output_dir):
    """Plots the distribution of samples for a given category and calculates resampling weights."""
    plt.figure(figsize=(10, 6))
    counts = df[category].value_counts()
    normalized_counts = counts / len(df)
    weights = 1 / normalized_counts

    ax = sns.barplot(x=normalized_counts.index, y=normalized_counts.values)

    for i, p in enumerate(ax.patches):
        label = normalized_counts.index[i]
        weight = weights[label]
        ax.annotate(f"Weight: {weight:.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')

    plt.title(f"Normalized Sample Distribution by {category.capitalize()}")
    plt.xlabel(category.capitalize())
    plt.ylabel("Normalized Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / f"{category}_distribution.png")
    plt.close()

def main():
    """Main function to run the analysis."""
    output_dir = Path("analysis_plots")
    output_dir.mkdir(exist_ok=True)

    metadata_df = load_metadata()

    plot_distribution(metadata_df, "dataset", output_dir)
    plot_distribution(metadata_df, "gender", output_dir)
    plot_distribution(metadata_df, "health", output_dir)
    plot_distribution(metadata_df, "age", output_dir)

if __name__ == "__main__":
    main()
