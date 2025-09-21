
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from signal_diffusion.config import load_settings
from signal_diffusion.data.parkinsons import ParkinsonsDataset
from signal_diffusion.data.mit import MITDataset
from signal_diffusion.data.seed import SEEDDataset
from signal_diffusion.data.math import MathDataset


def load_metadata(settings, dataset_names):
    """Loads metadata from specified datasets into a single DataFrame."""
    all_dfs = []
    dataset_classes = {
        "parkinsons": ParkinsonsDataset,
        "math": MathDataset,
        "mit": MITDataset,
        "seed": SEEDDataset,
    }

    for name in dataset_names:
        if name not in dataset_classes:
            print(f"Warning: Unknown dataset '{name}'. Skipping.")
            continue

        try:
            dataset_class = dataset_classes[name]
            df = dataset_class(settings, split="train").metadata
            df["dataset"] = name
            # Ensure 'health' column exists for all datasets
            if "health" not in df.columns:
                df["health"] = "healthy" # Use "healthy" as a placeholder for datasets without health info
            all_dfs.append(df)
        except FileNotFoundError:
            print(f"Warning: {name.capitalize()} dataset metadata not found. Skipping.")
        except Exception as e:
            print(f"Error loading {name.capitalize()} dataset: {e}. Skipping.")

    combined_df = pd.concat(all_dfs)
    # TODO: This is a hack to standardize 'health' labels.
    # The ParkinsonsDataset metadata logic should be modified to
    # produce 'healthy' and 'parkinsons' directly when generating metadata.
    combined_df["health"] = combined_df["health"].replace({"Control": "healthy", "PD": "parkinsons"})
    return combined_df


def plot_distribution(df, category, output_dir):
    """Plots the distribution of samples for a given category and calculates resampling weights."""
    plt.figure(figsize=(10, 6))

    if category == "dataset":
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
        plt.ylabel("Normalized Count")
    else:
        # For other categories, plot stacked normalized distribution by dataset
        # Calculate overall weights based on the category distribution
        overall_counts = df[category].value_counts()
        overall_normalized_counts = overall_counts / len(df)
        weights = 1 / overall_normalized_counts # Keep weights for potential future use or just for understanding

        ax = sns.histplot(data=df, x=category, hue="dataset", multiple="stack", stat="proportion", common_norm=True)

        # The stacked bars visually represent the contribution of each subdataset.
        # Annotating individual proportions on stacked bars can be very cluttered.
        # The overall weight for the category value is still relevant, but hard to place cleanly on histplot.
        # For now, we rely on the visual representation and the title.

        plt.title(f"Normalized Sample Distribution by {category.capitalize()} (Stacked by Dataset)")
        plt.ylabel("Proportion of Total Samples")

    plt.xlabel(category.capitalize())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / f"{category}_distribution.png")
    plt.close()


def plot_raw_counts(df, category, output_dir):
    """Plots the raw number of samples for a given category."""
    plt.figure(figsize=(10, 6))

    if category == "dataset":
        counts = df[category].value_counts()
        ax = sns.barplot(x=counts.index, y=counts.values)

        for p in ax.patches:
            ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')
        plt.title(f"Raw Sample Counts by {category.capitalize()}")
        plt.ylabel("Raw Count")
    else:
        # For other categories, plot stacked raw counts by dataset
        ax = sns.histplot(data=df, x=category, hue="dataset", multiple="stack", stat="count")

        # Annotate each segment of the stacked bar
        for container in ax.containers:
            for p in container.patches:
                height = p.get_height()
                if height > 0: # Only annotate if there's a segment
                    ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., p.get_y() + height / 2.),
                                ha='center', va='center', fontsize=8, color='white')

        plt.title(f"Raw Sample Counts by {category.capitalize()} (Stacked by Dataset)")
        plt.ylabel("Count")

    plt.xlabel(category.capitalize())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / f"{category}_raw_counts.png")
    plt.close()

def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None, help="Path to the config file.")
    parser.add_argument("datasets", nargs="*", default=["math", "parkinsons", "seed", "mit"], help="Datasets to analyze.")
    args = parser.parse_args()

    output_dir = Path("analysis_plots")
    output_dir.mkdir(exist_ok=True)

    settings = load_settings(args.config)
    metadata_df = load_metadata(settings, args.datasets)
    plot_distribution(metadata_df, "dataset", output_dir)
    plot_raw_counts(metadata_df, "dataset", output_dir)
    plot_distribution(metadata_df, "gender", output_dir)
    plot_raw_counts(metadata_df, "gender", output_dir)
    plot_distribution(metadata_df, "health", output_dir)
    plot_raw_counts(metadata_df, "health", output_dir)
    plot_distribution(metadata_df, "age", output_dir)
    plot_raw_counts(metadata_df, "age", output_dir)


if __name__ == "__main__":
    main()
