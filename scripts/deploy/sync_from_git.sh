#!/bin/bash
# Pull latest code from GitHub and update dependencies if needed

set -e

# Find project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$PROJECT_DIR"

echo "========================================"
echo "Syncing Pythia from GitHub"
echo "========================================"
echo ""

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "WARNING: You have uncommitted changes"
    echo "Stashing local changes..."
    git stash
    STASHED=1
else
    STASHED=0
fi

# Get current commit
OLD_COMMIT=$(git rev-parse HEAD)

# Pull latest changes
echo "Pulling latest changes from origin/main..."
git pull origin main

# Get new commit
NEW_COMMIT=$(git rev-parse HEAD)

if [ "$OLD_COMMIT" = "$NEW_COMMIT" ]; then
    echo "✓ Already up to date"
else
    echo "✓ Updated from $OLD_COMMIT to $NEW_COMMIT"
    echo ""
    
    # Show what changed
    echo "Changes:"
    git log --oneline --no-decorate "$OLD_COMMIT..$NEW_COMMIT" | head -10
    echo ""
    
    # Check if requirements.txt changed
    if git diff --name-only "$OLD_COMMIT" "$NEW_COMMIT" | grep -q "requirements.txt"; then
        echo "requirements.txt changed, updating dependencies..."
        
        if [ -d "venv" ]; then
            source venv/bin/activate
            pip install --upgrade pip
            pip install -r requirements.txt
            echo "✓ Dependencies updated"
        else
            echo "WARNING: Virtual environment not found, skipping dependency update"
        fi
        echo ""
    fi
fi

if [ $STASHED -eq 1 ]; then
    echo "Your local changes were stashed. To restore:"
    echo "  git stash pop"
    echo ""
fi

echo "========================================"
echo "Sync Complete!"
echo "========================================"
echo ""
echo "To apply changes:"
echo "  1. Stop collector: ./scripts/deploy/stop_collector.sh"
echo "  2. Start collector: ./scripts/deploy/start_collector.sh"
echo ""
