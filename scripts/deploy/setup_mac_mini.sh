#!/bin/bash
# Setup script for deploying Pythia on Mac Mini
# Run this once during initial deployment

set -e  # Exit on error

echo "========================================"
echo "Pythia Mac Mini Setup"
echo "========================================"
echo ""

# Find project root (2 levels up from this script)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$PROJECT_DIR"
echo "Project directory: $PROJECT_DIR"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.12"

if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    echo "ERROR: Python $REQUIRED_VERSION or higher required. Found: $PYTHON_VERSION"
    echo "Install miniforge: brew install --cask miniforge"
    exit 1
fi
echo "✓ Python $PYTHON_VERSION"
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p logs
mkdir -p data
mkdir -p models/fast_big
mkdir -p config
echo "✓ Directories created"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment and install dependencies
echo "Installing dependencies..."
source venv/bin/activate

if [ -f "requirements.txt" ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "WARNING: requirements.txt not found"
fi
echo ""

# Create example environment file if it doesn't exist
if [ ! -f "config/mac_mini.env.example" ]; then
    echo "Creating example environment file..."
    cat > config/mac_mini.env.example << 'EOF'
# Pythia Mac Mini Environment Configuration
# Copy this to config/mac_mini.env and fill in your values
# DO NOT commit mac_mini.env to git!

# Coinbase API credentials
COINBASE_API_KEY=your_api_key_here
COINBASE_API_SECRET=your_api_secret_here

# Environment
PYTHIA_ENV=production

# Paths (adjust for your setup)
PYTHIA_DB_PATH=market_data.duckdb
PYTHIA_LOG_DIR=logs

# Optional: Log level
LOG_LEVEL=INFO
EOF
    echo "✓ Created config/mac_mini.env.example"
    echo ""
    echo "IMPORTANT: Copy and configure your environment file:"
    echo "  cp config/mac_mini.env.example config/mac_mini.env"
    echo "  vim config/mac_mini.env  # Add your API credentials"
    echo ""
fi

# Make deployment scripts executable
echo "Setting script permissions..."
chmod +x scripts/deploy/*.sh
echo "✓ Scripts are executable"
echo ""

# Check if models exist
echo "Checking for model files..."
MODEL_FILES=(
    "models/xgboost_slow_large_v1.pkl"
    "models/xgboost_slow_large_v3_model.pkl"
    "models/fast_big/xgboost_fast_vA.pkl"
    "models/fast_big/xgboost_fast_vB.pkl"
)

MISSING_MODELS=0
for model in "${MODEL_FILES[@]}"; do
    if [ -f "$model" ]; then
        echo "  ✓ $model"
    else
        echo "  ✗ $model (MISSING)"
        MISSING_MODELS=$((MISSING_MODELS + 1))
    fi
done

if [ $MISSING_MODELS -gt 0 ]; then
    echo ""
    echo "WARNING: $MISSING_MODELS model file(s) missing"
    echo "Train models or copy from another machine before running collector"
fi
echo ""

# Summary
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Configure environment: vim config/mac_mini.env"
echo "  2. Start collector: ./scripts/deploy/start_collector.sh"
echo "  3. Check status: ./scripts/deploy/status.sh"
echo ""
echo "Other commands:"
echo "  Stop collector: ./scripts/deploy/stop_collector.sh"
echo "  Update code: ./scripts/deploy/sync_from_git.sh"
echo ""
