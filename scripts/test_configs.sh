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
    "dit-db-only"
    "dit-db-iq"
    "dit-db-polar"
    "dit-timeseries"

    # Hourglass variants
    "hourglass-db-only"
    "hourglass-db-iq"
    "hourglass-db-polar"
    "hourglass-timeseries"

    # LocalMamba variants
    "localmamba-db-only"
    "localmamba-db-iq"
    "localmamba-db-polar"
    "localmamba-timeseries"

    # Stable Diffusion 3.5 variants
    "sd35-db-only"
    "sd35-db-iq"
    "sd35-db-polar"
    "sd35-timeseries"
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

        echo -e "${GREEN}✓ PASSED${NC} $test_name"
        PASSED=$((PASSED + 1))
        return 0
    else
        local exit_code=$?

        if [ $exit_code -eq 124 ]; then
            echo -e "${RED}✗ TIMEOUT${NC} $test_name (exceeded ${TIMEOUT}s)"
            echo "Test timed out after ${TIMEOUT}s" >> "$log_file"
        else
            echo -e "${RED}✗ FAILED${NC} $test_name (exit code: $exit_code)"
            echo "Training failed with exit code: $exit_code" >> "$log_file"
        fi

        FAILED=$((FAILED + 1))

        # Show last 20 lines of log for debugging
        if [ -f "$log_file" ]; then
            echo "  Last log lines:"
            tail -20 "$log_file" | sed 's/^/    /'
        fi

        return 1
    fi
}

# Run all configurations with both conditioning types
CONDITIONING_TYPES=("classes" "caption")
TOTAL_TESTS=$((${#CONFIGS[@]} * ${#CONDITIONING_TYPES[@]}))

echo "Testing ${#CONFIGS[@]} configurations × ${#CONDITIONING_TYPES[@]} conditioning types = $TOTAL_TESTS tests..."
echo ""

for config in "${CONFIGS[@]}"; do
    for conditioning in "${CONDITIONING_TYPES[@]}"; do
        test_config_with_conditioning "$config" "$conditioning"
        echo ""
    done
done

# Print summary
echo "================================================================================"
echo "                              TEST SUMMARY"
echo "================================================================================"
echo "Total:    $TOTAL"
echo -e "Passed:   ${GREEN}$PASSED${NC}"
echo -e "Failed:   ${RED}$FAILED${NC}"
echo -e "Skipped:  ${YELLOW}$SKIPPED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
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
    log_file="${TEST_OUTPUT}/${config}.log"
    if [ -f "$log_file" ]; then
        log_size=$(wc -l < "$log_file")
        echo "  $config.log ($log_size lines)"
    fi
done

exit $EXIT_CODE
