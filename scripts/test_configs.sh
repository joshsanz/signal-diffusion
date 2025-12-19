#!/bin/bash

# Quick validation script to test all diffusion configurations
# Runs each config for a limited number of steps to exercise the code

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
MAX_STEPS=50
TIMEOUT=300  # 5 minutes per config
CONFIGS_DIR="config/diffusion"
OUTPUT_BASE="runs/config-tests"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEST_OUTPUT="$OUTPUT_BASE/test_${TIMESTAMP}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0
TIMEOUT_COUNT=0

# Associative array to store durations
declare -A DURATIONS

# Function to extract steps completed from log file
extract_steps() {
    local log_file=$1

    if [ ! -f "$log_file" ]; then
        echo "?"
        return
    fi

    # Look for step information in common formats
    # Match: "after 50 steps", "Step 45", "step 45", "steps: 45", "50/50"
    local steps=$(grep -oE "after [0-9]+ steps|Step [0-9]+|step [0-9]+|steps: [0-9]+|[0-9]+/[0-9]+" "$log_file" | tail -1 | grep -oE "[0-9]+" | tail -1) || true

    if [ -z "$steps" ]; then
        echo "?"
    else
        echo "$steps"
    fi
}

# Function to extract duration from log file
extract_duration() {
    local log_file=$1

    if [ ! -f "$log_file" ]; then
        echo "N/A"
        return
    fi

    # Get first and last timestamps from log
    local first_timestamp=$(grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}' "$log_file" | head -1)
    local last_timestamp=$(grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}' "$log_file" | tail -1)

    if [ -z "$first_timestamp" ] || [ -z "$last_timestamp" ]; then
        echo "N/A"
        return
    fi

    # Convert to seconds since epoch
    local start_sec=$(date -d "$first_timestamp" +%s 2>/dev/null || echo "")
    local end_sec=$(date -d "$last_timestamp" +%s 2>/dev/null || echo "")

    if [ -z "$start_sec" ] || [ -z "$end_sec" ]; then
        echo "N/A"
        return
    fi

    # Calculate duration in seconds
    local duration=$((end_sec - start_sec))

    # Format as mm:ss or Xm Ys
    if [ $duration -lt 60 ]; then
        echo "${duration}s"
    elif [ $duration -lt 3600 ]; then
        local minutes=$((duration / 60))
        local seconds=$((duration % 60))
        printf "%dm %ds" $minutes $seconds
    else
        local hours=$((duration / 3600))
        local minutes=$(((duration % 3600) / 60))
        printf "%dh %dm" $hours $minutes
    fi
}

# Create output directory
mkdir -p "$TEST_OUTPUT"

echo "================================================================================"
echo "                    DIFFUSION CONFIG VALIDATION TEST"
echo "================================================================================"
echo "Timestamp: $TIMESTAMP"
echo "Max Steps: $MAX_STEPS"
echo "Timeout: ${TIMEOUT}s per config"
echo "Test Output: $TEST_OUTPUT"
echo "================================================================================"
echo ""

# List of configs to test (4 models × 4 datasets)
CONFIGS=(
    # DiT variants
    # "dit-db-only"
    # "dit-db-iq"
    # "dit-db-polar"
    # "dit-timeseries"

    # Hourglass variants
    # "hourglass-db-only"
    # "hourglass-db-iq"
    # "hourglass-db-polar"
    # "hourglass-timeseries"

    # LocalMamba variants
    # "localmamba-db-only"
    # "localmamba-db-iq"
    # "localmamba-db-polar"
    "localmamba-timeseries"

    # Stable Diffusion 3.5 variants
    "sd35-db-only"
    "sd35-db-iq"
    "sd35-db-polar"
)

# Function to test a configuration with specific conditioning
test_config_with_conditioning() {
    local config_name=$1
    local conditioning=$2
    local config_file="${CONFIGS_DIR}/${config_name}.toml"
    local test_name="${config_name}-${conditioning}"
    local output_dir="${TEST_OUTPUT}/${test_name}"
    local log_file="${TEST_OUTPUT}/${test_name}.log"

    # Create temporary config file with conditioning override
    local temp_config=$(mktemp)
    trap "rm -f $temp_config" RETURN

    # Copy config and replace conditioning setting
    if [ -f "$config_file" ]; then
        # If conditioning line exists, replace it; otherwise add it under [model] section
        if grep -q "^conditioning = " "$config_file"; then
            sed "s/^conditioning = .*/conditioning = \"$conditioning\"/" "$config_file" > "$temp_config"
        else
            # Add conditioning after [model] line
            awk "/^\[model\]/{p=1} p && !done && /^[a-zA-Z]/{print \"conditioning = \\\"$conditioning\\\"\"; done=1} {print}" "$config_file" > "$temp_config"
        fi

        # If conditioning is "caption", also set caption_column to "caption" and num_classes to 0
        if [ "$conditioning" = "caption" ]; then
            if grep -q "^caption_column = " "$temp_config"; then
                sed -i "s/^caption_column = .*/caption_column = \"caption\"/" "$temp_config"
            else
                # Add caption_column after [dataset] line if it doesn't exist
                awk "/^\[dataset\]/{p=1} p && !done && /^[a-zA-Z]/{print \"caption_column = \\\"caption\\\"\"; done=1} {print}" "$temp_config" > "$temp_config.tmp" && mv "$temp_config.tmp" "$temp_config"
            fi

            # Set num_classes to 0 for caption conditioning
            if grep -q "^num_classes = " "$temp_config"; then
                sed -i "s/^num_classes = .*/num_classes = 0/" "$temp_config"
            else
                # Add num_classes after [dataset] line if it doesn't exist
                awk "/^\[dataset\]/{p=1} p && !done && /^[a-zA-Z]/{print \"num_classes = 0\"; done=1} {print}" "$temp_config" > "$temp_config.tmp" && mv "$temp_config.tmp" "$temp_config"
            fi
        fi
    else
        echo -e "${YELLOW}⊘ SKIPPED${NC} $test_name (config not found)"
        SKIPPED=$((SKIPPED + 1))
        return 1
    fi

    TOTAL=$((TOTAL + 1))

    echo -e "${BLUE}⟳ TESTING${NC} $test_name..."

    # Run training with max_steps limit
    if timeout $TIMEOUT uv run python -m signal_diffusion.training.diffusion \
        "$temp_config" \
        --output-dir "$output_dir" \
        --max-train-steps $MAX_STEPS \
        > "$log_file" 2>&1; then

        # Extract duration and step count from logs for successful runs
        local duration=$(extract_duration "$log_file")
        local steps_completed=$(extract_steps "$log_file")

        DURATIONS["$test_name"]="$duration"
        echo -e "${GREEN}✓ PASSED${NC} $test_name (completed $steps_completed steps in $duration)"
        PASSED=$((PASSED + 1))
        return 0
    else
        local exit_code=$?

        if [ $exit_code -eq 124 ]; then
            # Extract duration and step count from logs
            local duration=$(extract_duration "$log_file")
            local steps_completed=$(extract_steps "$log_file")

            DURATIONS["$test_name"]="$duration"
            echo -e "${YELLOW}⏱ TIMEOUT${NC} $test_name (exceeded ${TIMEOUT}s, completed $steps_completed steps in $duration)"
            echo "Test timed out after ${TIMEOUT}s (completed $steps_completed steps)" >> "$log_file"

            PASSED=$((PASSED + 1))
            TIMEOUT_COUNT=$((TIMEOUT_COUNT + 1))
            return 0
        else
            local duration=$(extract_duration "$log_file")
            local steps_completed=$(extract_steps "$log_file")
            DURATIONS["$test_name"]="$duration"
            echo -e "${RED}✗ FAILED${NC} $test_name (exit code: $exit_code, completed $steps_completed steps in $duration)"
            echo "Training failed with exit code: $exit_code" >> "$log_file"

            FAILED=$((FAILED + 1))

            # Show last 50 lines of log for debugging
            if [ -f "$log_file" ]; then
                echo "  Last log lines:"
                tail -50 "$log_file" | sed 's/^/    /'
            fi

            return 1
        fi
    fi
}

# Run all configurations with appropriate conditioning types
# SD3.5 models only support caption, others support both gend_hlth_age and caption
CONDITIONING_TYPES=("gend_hlth_age" "caption")
SD35_COUNT=$(printf '%s\n' "${CONFIGS[@]}" | grep -c "^sd35" || true)
OTHER_COUNT=$((${#CONFIGS[@]} - SD35_COUNT))
TOTAL_TESTS=$((OTHER_COUNT * ${#CONDITIONING_TYPES[@]} + SD35_COUNT))

echo "Testing ${#CONFIGS[@]} configurations (SD3.5: caption only, others: 2 conditioning types) = $TOTAL_TESTS tests..."
echo ""

for config in "${CONFIGS[@]}"; do
    # SD3.5 models only support caption conditioning
    if [[ "$config" == sd35* ]]; then
        test_config_with_conditioning "$config" "caption"
        echo ""
    else
        for conditioning in "${CONDITIONING_TYPES[@]}"; do
            test_config_with_conditioning "$config" "$conditioning"
            echo ""
        done
    fi
done

# Print summary
echo "================================================================================"
echo "                              TEST SUMMARY"
echo "================================================================================"
echo "Total:     $TOTAL"
echo -e "Passed:    ${GREEN}$PASSED${NC}"
echo -e "Timeouts:  ${YELLOW}$TIMEOUT_COUNT${NC}"
echo -e "Failed:    ${RED}$FAILED${NC}"
echo -e "Skipped:   ${YELLOW}$SKIPPED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed (timeouts do not count as failures)${NC}"
    EXIT_CODE=0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    EXIT_CODE=1
fi

echo ""
echo "Test output saved to: $TEST_OUTPUT"
echo "================================================================================"

# Print log file summary
echo ""
echo "Log Files:"
for config in "${CONFIGS[@]}"; do
    for conditioning in "${CONDITIONING_TYPES[@]}"; do
        log_file="${TEST_OUTPUT}/${config}-${conditioning}.log"
        if [ -f "$log_file" ]; then
            log_size=$(wc -l < "$log_file")
            echo "  ${config}-${conditioning}.log ($log_size lines)"
        fi
    done
done

exit $EXIT_CODE
