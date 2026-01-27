# Classification plots: accuracies and confusion matrices
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import seaborn as sns
from pathlib import Path
from itertools import product

plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans',
                                   'Lucida Grande', 'Verdana']

def format_eval_label(eval_key):
    """Convert 'real_train__synth_test' to 'Train: Real | Test: Synth'."""
    parts = eval_key.split('__')
    if len(parts) != 2:
        return eval_key

    def parse_split(raw_label):
        if raw_label.endswith('_validation'):
            split = 'Val'
            dataset = raw_label[:-11]
        elif raw_label.endswith('_val'):
            split = 'Val'
            dataset = raw_label[:-4]
        elif raw_label.endswith('_test'):
            split = 'Test'
            dataset = raw_label[:-5]
        elif raw_label.endswith('_train'):
            split = 'Train'
            dataset = raw_label[:-6]
        else:
            split = 'Eval'
            dataset = raw_label
        dataset = dataset.replace('_', ' ').strip().capitalize()
        return split, dataset

    train_split, train_dataset = parse_split(parts[0])
    eval_split, eval_dataset = parse_split(parts[1])
    return f"{train_split}: {train_dataset} | {eval_split}: {eval_dataset}"

metrics_dir = Path(__file__).parent
classification_files = sorted(metrics_dir.glob('classification-*.json'))

print(f"Found {len(classification_files)} classification files:")
for cf in classification_files:
    print(f"  - {cf.name}")

if not classification_files:
    print("No classification JSON files found!")
    exit(1)

# Load all classification results
all_results = {}
for cf in classification_files:
    # Extract model name from filename: classification-{model}.json
    model_name = cf.stem.replace('classification-', '')
    with open(cf) as f:
        all_results[model_name] = json.load(f)
    print(f"Loaded {model_name}")

# Build accuracy comparison dataframe
accuracy_data = []
age_data = []
for model_name, results in all_results.items():
    for eval_key, eval_results in results['evaluations'].items():
        for task, metrics in eval_results.items():
            if task == 'num_examples':
                continue
            if task == 'health' and 'macro_f1' in metrics:
                # Use F1 score for health instead of accuracy
                accuracy_data.append({
                    'Model': model_name,
                    'Evaluation': eval_key,
                    'Task': task,
                    'Accuracy': metrics['macro_f1']
                })
            elif task != 'health' and 'accuracy' in metrics:
                accuracy_data.append({
                    'Model': model_name,
                    'Evaluation': eval_key,
                    'Task': task,
                    'Accuracy': metrics['accuracy']
                })
            elif task == 'age' and ('mae' in metrics or 'mse' in metrics):
                age_data.append({
                    'Model': model_name,
                    'Evaluation': eval_key,
                    'MAE': metrics.get('mae', None),
                    'MSE': metrics.get('mse', None)
                })

accuracy_df = pd.DataFrame(accuracy_data)
age_df = pd.DataFrame(age_data)
print("\nAccuracy data:")
print(accuracy_df.head(10))
print("\nAge error data:")
print(age_df)

# Helper function to sort evaluation keys consistently
def get_eval_sort_key(eval_key):
    """Create sort key for consistent ordering of evaluation keys.

    Order: real-real < real-real_val < real-synth < synth-synth < synth-real < synth-real_val
    """
    parts = eval_key.split('__')
    if len(parts) != 2:
        return (999, eval_key)

    train_part, test_part = parts

    # Extract base names
    def get_base(s):
        for suffix in ['_train', '_test', '_validation']:
            if s.endswith(suffix):
                return s[:-len(suffix)]
        return s

    train_base = get_base(train_part)
    test_base = get_base(test_part)
    has_validation = 'validation' in test_part.lower()

    # Define priority for (train_base, test_base) pairs
    pair_priority = {
        ('real', 'real'): 0,
        ('real', 'synth'): 1,
        ('synth', 'synth'): 2,
        ('synth', 'real'): 3,
    }
    pair_order = pair_priority.get((train_base, test_base), 999)

    return (pair_order, has_validation, eval_key)

# Helper function to map evaluation keys to colors
def get_color_for_eval_key(eval_key):
    """Map evaluation key to color, creating lighter versions for validation sets."""
    parts = eval_key.split('__')
    if len(parts) != 2:
        return '#cccccc'  # Default gray

    train_part, test_part = parts

    # Extract base dataset names (remove _train, _test, _validation suffixes)
    def get_base(s):
        for suffix in ['_train', '_test', '_validation']:
            if s.endswith(suffix):
                return s[:-len(suffix)]
        return s

    train_base = get_base(train_part)
    test_base = get_base(test_part)
    is_validation = 'validation' in eval_key.lower()

    # Map to base colors
    if train_base == 'real' and test_base == 'real':
        base_color = 'C0'
    elif train_base == 'real' and test_base == 'synth':
        base_color = 'C1'
    elif train_base == 'synth' and test_base == 'real':
        base_color = 'C2'
    else:
        base_color = 'C7'

    if is_validation:
        # Create lighter version by interpolating 60% towards white
        rgb = plt.matplotlib.colors.to_rgb(base_color)
        lighter_rgb = tuple(c + 0.6 * (1 - c) for c in rgb)
        return lighter_rgb
    else:
        return base_color

# Plot 1: Accuracy comparison by task and evaluation
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
tasks = ['gender', 'health']
eval_keys = accuracy_df['Evaluation'].unique()

for task_idx, task in enumerate(tasks):
    task_data = accuracy_df[accuracy_df['Task'] == task]
    task_data_pivot = task_data.pivot(index='Model', columns='Evaluation', values='Accuracy')

    # Sort columns consistently
    eval_cols = sorted(task_data_pivot.columns, key=get_eval_sort_key)
    task_data_pivot = task_data_pivot[eval_cols]

    # Format column labels
    task_data_pivot.columns = [format_eval_label(col) for col in task_data_pivot.columns]

    ax = axes[task_idx]
    task_data_pivot.plot(kind='bar', ax=ax, width=0.8)

    # Apply colors to patches
    patches = ax.patches
    num_models = len(task_data_pivot.index)
    num_evals = len(eval_cols)

    for idx, patch in enumerate(patches):
        eval_idx = idx // num_models
        eval_key = eval_cols[eval_idx]
        color = get_color_for_eval_key(eval_key)
        patch.set_color(color)

    if task == 'health':
        ax.set_title(f'{task.capitalize()} F1 Score (macro)')
        ax.set_ylabel('F1 Score (macro)')
    else:
        ax.set_title(f'{task.capitalize()} Accuracy')
        ax.set_ylabel('Accuracy')
    ax.set_xlabel('Model')
    ax.set_ylim(0, 1.0)
    ax.legend(title='Evaluation', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(metrics_dir / 'classification-accuracy.png', dpi=300, bbox_inches='tight')
print(f"Saved accuracy plot to {metrics_dir / 'classification-accuracy.png'}")
plt.close()

# Plot 1b: Age errors (MAE and MSE)
if not age_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # MAE plot
    if 'MAE' in age_df.columns:
        mae_pivot = age_df.pivot(index='Model', columns='Evaluation', values='MAE')
        eval_cols_age = sorted(mae_pivot.columns, key=get_eval_sort_key)
        mae_pivot = mae_pivot[eval_cols_age]
        mae_pivot.columns = [format_eval_label(col) for col in mae_pivot.columns]

        # Build color list
        colors_age = [get_color_for_eval_key(eval_key) for eval_key in eval_cols_age]

        mae_pivot.plot(kind='bar', ax=axes[0], width=0.8)

        # Apply colors to patches
        patches = axes[0].patches
        num_models = len(mae_pivot.index)
        num_evals = len(eval_cols_age)

        for idx, patch in enumerate(patches):
            eval_idx = idx // num_models
            patch.set_color(colors_age[eval_idx])

        axes[0].set_title('Age Prediction - MAE')
        axes[0].set_ylabel('Mean Absolute Error (years)')
        axes[0].set_xlabel('Model')
        axes[0].legend(title='Evaluation', fontsize=9, loc='lower right')
        axes[0].grid(axis='y', alpha=0.3)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # RMS plot
    if 'MSE' in age_df.columns:
        mse_pivot = age_df.pivot(index='Model', columns='Evaluation', values='MSE')
        eval_cols_age = sorted(mse_pivot.columns, key=get_eval_sort_key)
        mse_pivot = mse_pivot[eval_cols_age]
        rms_pivot = np.sqrt(mse_pivot)
        rms_pivot.columns = [format_eval_label(col) for col in rms_pivot.columns]

        # Build color list
        colors_age = [get_color_for_eval_key(eval_key) for eval_key in eval_cols_age]

        rms_pivot.plot(kind='bar', ax=axes[1], width=0.8)

        # Apply colors to patches
        patches = axes[1].patches
        num_models = len(rms_pivot.index)
        num_evals = len(eval_cols_age)

        for idx, patch in enumerate(patches):
            eval_idx = idx // num_models
            patch.set_color(colors_age[eval_idx])

        axes[1].set_title('Age Prediction - RMS')
        axes[1].set_ylabel('Root Mean Squared Error (years)')
        axes[1].set_xlabel('Model')
        axes[1].legend(title='Evaluation', fontsize=9, loc='lower right')
        axes[1].grid(axis='y', alpha=0.3)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Tie y-axes together
    if 'MAE' in age_df.columns and 'MSE' in age_df.columns:
        y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
        axes[0].set_ylim(0, y_max)
        axes[1].set_ylim(0, y_max)

    plt.tight_layout()
    plt.savefig(metrics_dir / 'classification-age-errors.png', dpi=300, bbox_inches='tight')
    print(f"Saved age errors plot to {metrics_dir / 'classification-age-errors.png'}")
    plt.close()

# Plot 2: Confusion matrices
for model_name, results in all_results.items():
    eval_results = results['evaluations']

    # Get all evaluation types
    for eval_key in sorted(eval_results.keys()):
        eval_data = eval_results[eval_key]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{model_name.upper()} - {format_eval_label(eval_key)}', fontsize=14, fontweight='bold')

        for ax_idx, task in enumerate(['gender', 'health']):
            ax = axes[ax_idx]
            if task in eval_data and 'confusion_matrix' in eval_data[task]:
                cm = np.array(eval_data[task]['confusion_matrix'])
                accuracy = eval_data[task]['accuracy']

                # Normalize for visualization
                cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

                sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues', ax=ax,
                           cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
                ax.set_title(f'{task.capitalize()} (Accuracy: {accuracy:.2%})')
                ax.set_ylabel('True Label')
                ax.set_xlabel('Predicted Label')
                ax.set_xticklabels(['Negative', 'Positive'])
                ax.set_yticklabels(['Negative', 'Positive'])

        plt.tight_layout()
        # Create a filename-safe version of the eval label
        safe_eval_key = eval_key.replace('__', '-').replace('_', '')
        filename = f'confusion-{model_name}-{safe_eval_key}.png'
        plt.savefig(metrics_dir / filename, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {metrics_dir / filename}")
        plt.close()

print(f"\nAll plots saved to {metrics_dir}")
