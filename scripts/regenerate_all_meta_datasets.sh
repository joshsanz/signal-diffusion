#!/bin/bash
# Script to regenerate meta-weighted datasets for all output types (db-only, db-polar, db-iq)
# Usage: ./regenerate_all_meta_datasets.sh [-p] [-y]
#   -p: Enable preprocessing (generates spectrograms from raw data)
#   -y: Force overwrite without confirmation prompts

set -e  # Exit on any error

# Parse command-line arguments
PREPROCESS=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--preprocess)
            PREPROCESS=true
            shift
            ;;
        -y|--force)
            FORCE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [-p] [-y]"
            echo "  -p, --preprocess  Enable preprocessing (generate spectrograms from raw data)"
            echo "  -y, --force       Force overwrite without confirmation prompts"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_ROOT/config/default.toml"
BACKUP_CONFIG="${CONFIG_FILE}.backup.regenerate"

# Verify config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Create backup of original config
echo "Creating backup of config file..."
cp "$CONFIG_FILE" "$BACKUP_CONFIG"

# Function to restore config on exit (both success and error)
restore_config() {
    echo "Restoring original config file..."
    mv "$BACKUP_CONFIG" "$CONFIG_FILE"
}
trap restore_config EXIT

# Function to update config for a given output type
update_config_for_output_type() {
    local output_type=$1

    echo "Updating config for output_type: $output_type"

    # Use sed to update the config file
    # Update output_type
    sed -i.tmp "s/^output_type = .*/output_type = \"$output_type\"/" "$CONFIG_FILE"

    # Determine output_root based on output_type
    local output_root
    case $output_type in
        "db-only")
            output_root="/data/data/signal-diffusion/processed"
            ;;
        "db-polar")
            output_root="/data/data/signal-diffusion/processed-polar"
            ;;
        "db-iq")
            output_root="/data/data/signal-diffusion/processed-iq"
            ;;
        *)
            echo "Error: Unknown output_type: $output_type"
            return 1
            ;;
    esac

    # Update output_root - need to escape forward slashes for sed
    local escaped_output_root=$(echo "$output_root" | sed 's/\//\\\//g')
    sed -i.tmp "s/^output_root = .*/output_root = \"$escaped_output_root\"/" "$CONFIG_FILE"

    # Ensure data_type is spectrogram
    sed -i.tmp 's/^data_type = .*/data_type = "spectrogram"/' "$CONFIG_FILE"

    # Clean up temporary sed file if it exists
    rm -f "${CONFIG_FILE}.tmp"

    return 0
}

# Function to run weighted dataset generation
run_weighted_dataset_generation() {
    local output_type=$1
    local script=$2

    echo ""
    echo "=========================================="
    echo "Generating weighted dataset for: $output_type"
    echo "Script: $script"
    echo "=========================================="

    # Build command
    local cmd="uv run python $script --overwrite"

    if [[ "$PREPROCESS" == true ]]; then
        cmd="$cmd --preprocess"
    fi

    if [[ "$FORCE" == true ]]; then
        cmd="$cmd --force"
    fi

    echo "Running: $cmd"

    # Run the command
    if ! (cd "$PROJECT_ROOT" && eval "$cmd"); then
        echo "Error: Failed to generate weighted dataset for $output_type"
        return 1
    fi

    echo "Successfully generated weighted dataset for: $output_type"
    return 0
}

# Main execution
echo "Regenerating meta-weighted datasets..."
echo "Preprocess: $PREPROCESS"
echo "Force overwrite: $FORCE"
echo ""

# Array of output types to process
OUTPUT_TYPES=("db-only" "db-polar" "db-iq")

# Track success/failure
FAILED=()

for output_type in "${OUTPUT_TYPES[@]}"; do
    # Update config for this output type
    if ! update_config_for_output_type "$output_type"; then
        echo "Error: Failed to update config for $output_type"
        FAILED+=("$output_type")
        continue
    fi

    # Run spectrogram dataset generation
    spectrogram_script="$SCRIPT_DIR/gen_weighted_spectrogram_dataset.py"
    if ! run_weighted_dataset_generation "$output_type" "$spectrogram_script"; then
        FAILED+=("$output_type (spectrogram)")
        # Don't exit, continue with other types
    fi
done

echo ""
echo "=========================================="
echo "Dataset generation complete!"
echo "=========================================="

if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "All datasets generated successfully!"
    exit 0
else
    echo "The following datasets failed to generate:"
    for failed in "${FAILED[@]}"; do
        echo "  - $failed"
    done
    exit 1
fi
