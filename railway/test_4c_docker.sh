#!/bin/bash
# Step-by-step test script to verify 4C Docker image works
# NOTE: This script is for LOCAL testing. On Railway, we're already IN the 4C image!

echo "=========================================="
echo "4C Docker Image Test Script (Local)"
echo "=========================================="
echo ""
echo "NOTE: On Railway, we're already IN the 4C Docker image,"
echo "      so we call the binary directly (no 'docker run' needed)"
echo ""

# Step 1: Check if Docker is available (for local testing)
echo "Step 1: Checking Docker (for local testing)..."
if ! command -v docker &> /dev/null; then
    echo "⚠ Docker not found - this script requires Docker for local testing"
    echo "  On Railway, Docker is not needed (we're in the image)"
    exit 1
fi
echo "✓ Docker found: $(docker --version)"
echo ""

# Step 2: Pull/check 4C Docker image
echo "Step 2: Pulling 4C Docker image..."
docker pull ghcr.io/4c-multiphysics/4c:main
if [ $? -ne 0 ]; then
    echo "❌ Failed to pull 4C Docker image"
    exit 1
fi
echo "✓ 4C Docker image available"
echo ""

# Step 3: Test 4C binary in container
echo "Step 3: Testing 4C binary in container..."
echo "Running: docker run --rm ghcr.io/4c-multiphysics/4c:main /home/user/4C/build/4C --help"
docker run --rm ghcr.io/4c-multiphysics/4c:main /home/user/4C/build/4C --help 2>&1 | head -20
echo ""

# Step 4: List available test input files
echo "Step 4: Listing test input files in 4C image..."
echo "Available test files:"
docker run --rm ghcr.io/4c-multiphysics/4c:main find /home/user/4C/tests/input_files -name "*.4C.yaml" -type f 2>/dev/null | head -10
echo ""

# Step 5: Run a simple test simulation
echo "Step 5: Running a test simulation..."
echo "This will take 30-60 seconds..."
TEST_OUTPUT=$(mktemp -d)
docker run --rm \
    -v "$TEST_OUTPUT:/output" \
    -w /output \
    ghcr.io/4c-multiphysics/4c:main \
    /home/user/4C/build/4C \
    /home/user/4C/tests/input_files/tutorial_solid_vtu.4C.yaml \
    test_output 2>&1

if [ $? -eq 0 ]; then
    echo "✓ 4C simulation completed successfully!"
    echo "Output files:"
    ls -lh "$TEST_OUTPUT" 2>/dev/null | head -10
else
    echo "❌ 4C simulation failed"
    echo "Check the error messages above"
fi

rm -rf "$TEST_OUTPUT"
echo ""
echo "=========================================="
echo "Local test complete!"
echo ""
echo "On Railway: Use /test-4c-docker endpoint"
echo "  (calls binary directly, no docker run)"
echo "=========================================="

