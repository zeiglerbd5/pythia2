#!/bin/bash
# Start the Pythia integrated collector in background

set -e

# Find project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
PID_FILE="$PROJECT_DIR/.collector.pid"
LOG_DIR="$PROJECT_DIR/logs"

cd "$PROJECT_DIR"

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "ERROR: Collector is already running (PID: $OLD_PID)"
        echo "Stop it first: ./scripts/deploy/stop_collector.sh"
        exit 1
    else
        echo "Removing stale PID file..."
        rm "$PID_FILE"
    fi
fi

# Double-check with pgrep
if pgrep -f "integrated_collector" > /dev/null; then
    echo "ERROR: Collector process found but no PID file!"
    echo "Stop manually: pkill -f integrated_collector"
    exit 1
fi

# Load environment if exists
if [ -f "config/mac_mini.env" ]; then
    echo "Loading environment from config/mac_mini.env..."
    set -a  # Export all variables
    source config/mac_mini.env
    set +a
else
    echo "WARNING: config/mac_mini.env not found"
    echo "Create it from: cp config/mac_mini.env.example config/mac_mini.env"
fi

# Activate virtual environment
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Run setup first: ./scripts/deploy/setup_mac_mini.sh"
    exit 1
fi

source venv/bin/activate

# Create log directory
mkdir -p "$LOG_DIR"

# Generate timestamped log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/collector_${TIMESTAMP}.log"

echo "Starting Pythia integrated collector..."
echo "  Log file: $LOG_FILE"
echo "  PID file: $PID_FILE"
echo ""

# Start collector in background with nohup
nohup python3 -u -m src.data_ingestion.integrated_collector \
    > "$LOG_FILE" 2>&1 &

# Save PID
COLLECTOR_PID=$!
echo "$COLLECTOR_PID" > "$PID_FILE"

# Wait a moment to see if it crashes immediately
sleep 2

if ps -p "$COLLECTOR_PID" > /dev/null; then
    echo "✓ Collector started successfully (PID: $COLLECTOR_PID)"
    echo ""
    echo "Monitor commands:"
    echo "  Status: ./scripts/deploy/status.sh"
    echo "  Live log: tail -f $LOG_FILE"
    echo "  Predictions: tail -f $LOG_FILE | grep 'PRED'"
    echo "  Alerts: tail -f $LOG_FILE | grep 'HOME RUN'"
    echo ""
else
    echo "ERROR: Collector failed to start!"
    echo "Check log file: $LOG_FILE"
    rm "$PID_FILE"
    exit 1
fi
