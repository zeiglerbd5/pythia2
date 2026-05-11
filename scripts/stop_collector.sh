#!/bin/bash
# Stop the Pythia integrated collector gracefully

# Find project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
PID_FILE="$PROJECT_DIR/.collector.pid"

cd "$PROJECT_DIR"

echo "Stopping Pythia collector..."

# Try to get PID from file
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "Found PID file: $PID"
else
    # Fallback to pgrep
    PID=$(pgrep -f "integrated_collector" | head -1)
    if [ -z "$PID" ]; then
        echo "Collector is not running (no PID file or process found)"
        exit 0
    fi
    echo "Found process via pgrep: $PID"
fi

# Check if process exists
if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "Process $PID not found, cleaning up..."
    rm -f "$PID_FILE"
    exit 0
fi

# Send SIGTERM for graceful shutdown
echo "Sending SIGTERM to PID $PID..."
kill "$PID"

# Wait up to 10 seconds for graceful shutdown
TIMEOUT=10
echo "Waiting for graceful shutdown (up to ${TIMEOUT}s)..."

for i in $(seq 1 $TIMEOUT); do
    if ! ps -p "$PID" > /dev/null 2>&1; then
        echo "✓ Collector stopped gracefully"
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
    echo -n "."
done
echo ""

# Still running, force kill
echo "Process still running, sending SIGKILL..."
kill -9 "$PID" 2>/dev/null

sleep 1

if ps -p "$PID" > /dev/null 2>&1; then
    echo "ERROR: Failed to kill process $PID"
    exit 1
else
    echo "✓ Collector force-stopped"
    rm -f "$PID_FILE"
    exit 0
fi
