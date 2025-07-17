#!/bin/bash

# Setup script for RL-based scx_flashyspark Scheduler Optimization
# This script helps install dependencies and verify the environment

set -e  # Exit on any error

echo "======================================================"
echo "RL-based scx_flashyspark Scheduler Optimization Setup"
echo "======================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
VENV_NAME="rl_scx_params"

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}



# Check if the virtual environment directory exists
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment '$VENV_NAME'..."
    python3 -m venv "$VENV_NAME"
else
    echo "Virtual environment '$VENV_NAME' already exists."
fi

# Activate the virtual environment
source "$VENV_NAME"/bin/activate

echo "Virtual environment '$VENV_NAME' activated."


# Check for pip
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 not found. Please install pip first."
    exit 1
fi

# Install Python dependencies
print_status "Installing Python dependencies..."
if [ -f "rl_experiments/requirements.txt" ]; then
    pip3 install -r rl_experiments/requirements.txt
    print_status "Python dependencies installed successfully"
else
    print_warning "requirements.txt not found, installing manually..."
    pip3 install torch gymnasium numpy pandas matplotlib seaborn
fi

# Check critical binaries
print_status "Checking required binaries..."

# Check scx_flashyspark
if [ -x "/home/nvidia/bin/scx_flashyspark" ]; then
    print_status "scx_flashyspark found at /home/nvidia/bin/scx_flashyspark"
else
    print_error "scx_flashyspark not found at /home/nvidia/bin/scx_flashyspark"
    print_error "Please compile and install scx_flashyspark first"
    exit 1
fi

# Check llama-bench
if [ -x "/home/nvidia/llama.cpp/build/bin/llama-bench" ]; then
    print_status "llama-bench found at /home/nvidia/llama.cpp/build/bin/llama-bench"
else
    print_error "llama-bench not found at /home/nvidia/llama.cpp/build/bin/llama-bench"
    print_error "Please compile llama.cpp with benchmark support first"
    exit 1
fi

# Check for models
print_status "Checking for models..."
models_dir="/home/nvidia/llama.cpp/models"
if [ -d "$models_dir" ]; then
    model_count=$(find "$models_dir" -name "*.gguf" | wc -l)
    if [ "$model_count" -gt 0 ]; then
        print_status "Found $model_count GGUF model(s) in $models_dir"
        find "$models_dir" -name "*.gguf" | head -3 | while read model; do
            print_status "  - $(basename "$model")"
        done
        if [ "$model_count" -gt 3 ]; then
            print_status "  ... and $((model_count - 3)) more"
        fi
    else
        print_warning "No GGUF models found in $models_dir"
        print_warning "Please download a model file for benchmarking"
    fi
else
    print_warning "Models directory $models_dir not found"
fi

# Check sudo access
print_status "Checking sudo access..."
if sudo -n true 2>/dev/null; then
    print_status "Sudo access available (no password required)"
elif sudo -v 2>/dev/null; then
    print_status "Sudo access available (password may be required)"
else
    print_error "No sudo access. Scheduler loading requires root privileges."
    exit 1
fi

# Check for existing scheduler processes
print_status "Checking for running scheduler processes..."
if pgrep -f "scx_" > /dev/null; then
    print_warning "Found running sched_ext processes:"
    ps aux | grep -E "scx_[a-zA-Z]" | grep -v grep
    print_warning "You may need to stop these before running the RL optimization"
    print_warning "Use: sudo pkill -f scx_"
else
    print_status "No conflicting scheduler processes found"
fi

# Create output directory
output_dir="/home/nvidia/rl_experiments"
print_status "Creating output directory..."
mkdir -p "$output_dir"
if [ -w "$output_dir" ]; then
    print_status "Output directory ready: $output_dir"
else
    print_error "Cannot write to output directory: $output_dir"
    exit 1
fi

# Test Python imports
print_status "Testing Python imports..."
python3 -c "
import sys
try:
    import torch
    import gymnasium
    import numpy
    import pandas
    import matplotlib
    import seaborn
    print('All required Python packages imported successfully')
except ImportError as e:
    print(f'Import error: {e}')
    sys.exit(1)
" || {
    print_error "Python import test failed"
    exit 1
}

# Final verification
print_status "Running final verification..."
if python3 -c "
import sys
sys.path.insert(0, '/home/nvidia/scripts')
try:
    from parse_benchmark_results import BenchmarkResultsParser
    print('Benchmark parser available')
except ImportError:
    print('Warning: Benchmark parser not available (optional)')
"; then
    print_status "All verifications completed"
fi

echo ""
echo "======================================================"
print_status "Setup completed successfully!"
echo "======================================================"
echo ""
echo "Quick start:"
echo "  # Test installation with a short run"
echo "  python rl_experiments/main.py --episodes 5 --baseline-runs 1"
echo ""
echo "  # Start full optimization"
echo "  python rl_experiments/main.py --algorithm ppo --episodes 100"
echo ""
echo "  # Get help"
echo "  python rl_experiments/main.py --help"
echo ""
echo "Check the README.md for detailed usage instructions."
