
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from pathlib import Path

from signal_diffusion.config import load_settings
from signal_diffusion.data.parkinsons import ParkinsonsDataset
from signal_diffusion.data.seed import SEEDDataset
from signal_diffusion.data.math import MathDataset


def load_metadata(settings, dataset_names):
    """Loads metadata from specified datasets into a single DataFrame."""
    all_dfs = []
    dataset_classes = {
        "parkinsons": ParkinsonsDataset,
        "math": MathDataset,
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

def plot_subject_prevalence(df, output_dir):
    """Plots the prevalence of subjects, broken down by gender and dataset."""
    plt.figure(figsize=(12, 8))

    # Group by dataset, subject, and gender to get counts
    subject_gender_counts = df.groupby(['dataset', 'subject', 'gender']).size().reset_index(name='count')
    # Sort by dataset and then by gender
    subject_gender_counts = subject_gender_counts.sort_values(by=['dataset', 'gender'])

    # Prepare data for pie chart
    data = []
    labels = []
    colors = []

    # Define a color palette for datasets and genders
    # Using seaborn's palette for consistency
    dataset_names = df['dataset'].unique()
    gender_names = df['gender'].unique()

    # Create a color map for each dataset-gender combination
    # This is a simplified approach compared to the original's fixed colors
    # For better visual distinction, we might need a more robust color assignment
    color_palette = sns.color_palette("Paired", len(dataset_names) * len(gender_names))
    color_map = {}
    color_idx = 0
    for ds in dataset_names:
        for g in gender_names:
            color_map[(ds, g)] = color_palette[color_idx]
            color_idx += 1

    for _, row in subject_gender_counts.iterrows():
        data.append(row['count'])
        labels.append(f"{row['subject']} ({row['gender']})")
        colors.append(color_map[(row['dataset'], row['gender'])])

    # Creating plot
    fig, ax = plt.subplots(figsize=(10, 7))
    wp = {'linewidth': 1, 'edgecolor': "black"}
    tp = {'fontsize': 8} # Increased font size for better readability

    patches, texts = ax.pie(data, labels=labels, colors=colors, textprops=tp, wedgeprops=wp, startangle=90)

    # Create Legend
    legend_patches = []
    for ds in dataset_names:
        for g in gender_names:
            if (ds, g) in color_map: # Only add if combination exists
                legend_patches.append(mpatches.Patch(color=color_map[(ds, g)], label=f"{ds} - {g}"))

    plt.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    ax.set_title("Subject Prevalence by Dataset and Gender")
    plt.tight_layout()
    plt.savefig(output_dir / "subject_prevalence.png")
    plt.close()

def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None, help="Path to the config file.")
    parser.add_argument("datasets", nargs="*", default=["math", "parkinsons", "seed"], help="Datasets to analyze.")
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
    plot_subject_prevalence(metadata_df, output_dir)


if __name__ == "__main__":
    main()
