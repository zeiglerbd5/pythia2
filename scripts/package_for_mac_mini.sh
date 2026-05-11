
#!/bin/bash
# Package Pythia for deployment to Mac Mini
# Creates a single tarball with everything needed

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
PACKAGE_NAME="pythia_mac_mini_$(date +%Y%m%d_%H%M%S).tar.gz"

cd "$PROJECT_DIR"

echo "========================================"
echo "Packaging Pythia for Mac Mini Deployment"
echo "========================================"
echo ""

# Files/directories to include
echo "Creating package with:"
echo "  ✓ Source code (src/)"
echo "  ✓ Configuration files (config/)"
echo "  ✓ Deployment scripts (scripts/deploy/)"
echo "  ✓ Model files (models/)"
echo "  ✓ Requirements (requirements.txt)"
echo ""

# Create tarball excluding unnecessary files
tar -czf "$PACKAGE_NAME" \
    --exclude='*.duckdb*' \
    --exclude='*.log' \
    --exclude='logs/*' \
    --exclude='data/*' \
    --exclude='venv/*' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='*.egg-info' \
    --exclude='.DS_Store' \
    --exclude='nohup.out' \
    --exclude='collector.log' \
    --exclude='.collector.pid' \
    src/ \
    config/ \
    scripts/ \
    models/ \
    requirements.txt \
    README.md \
    REMOTE_DEPLOYMENT.md \
    2>/dev/null || true

# Get package size
PACKAGE_SIZE=$(du -h "$PACKAGE_NAME" | awk '{print $1}')

echo "✓ Package created: $PACKAGE_NAME ($PACKAGE_SIZE)"
echo ""
echo "========================================"
echo "Deployment Instructions"
echo "========================================"
echo ""
echo "1. Copy package to Mac Mini:"
echo "   scp $PACKAGE_NAME your_user@mac-mini.local:~/"
echo ""
echo "2. SSH into Mac Mini:"
echo "   ssh your_user@mac-mini.local"
echo ""
echo "3. Extract and setup:"
echo "   mkdir -p ~/projects/Pythia"
echo "   cd ~/projects/Pythia"
echo "   tar -xzf ~/$PACKAGE_NAME"
echo "   ./scripts/deploy/setup_mac_mini.sh"
echo ""
echo "4. Configure API keys:"
echo "   cp config/mac_mini.env.example config/mac_mini.env"
echo "   nano config/mac_mini.env  # Add your Coinbase API keys"
echo ""
echo "5. Start collector:"
echo "   ./scripts/deploy/start_collector.sh"
echo ""
echo "6. Check status:"
echo "   ./scripts/deploy/status.sh"
echo ""
