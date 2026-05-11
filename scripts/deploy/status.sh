#!/bin/bash
# Check Pythia collector status and show recent activity

# Find project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
PID_FILE="$PROJECT_DIR/.collector.pid"
LOG_DIR="$PROJECT_DIR/logs"

cd "$PROJECT_DIR"

echo "========================================"
echo "Pythia Collector Status"
echo "========================================"
echo ""

# Check if running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "✅ RUNNING (PID: $PID)"
        echo ""
        
        # Show process info
        echo "Process Info:"
        ps aux | grep "$PID" | grep -v grep | awk '{printf "  CPU: %s%%  RAM: %s%%  Started: %s %s\n", $3, $4, $9, $10}'
        echo ""
        
        # Find most recent log file
        LATEST_LOG=$(ls -t "$LOG_DIR"/collector_*.log 2>/dev/null | head -1)
        
        if [ -n "$LATEST_LOG" ]; then
            echo "Log File: $(basename "$LATEST_LOG")"
            echo ""
            
            # Recent predictions
            echo "Recent Predictions (last 10):"
            grep "PRED" "$LATEST_LOG" | tail -10 | while read line; do
                echo "  $line"
            done || echo "  (No predictions yet)"
            echo ""
            
            # HOME RUN alerts
            echo "Recent HOME RUN Alerts:"
            grep "HOME RUN" "$LATEST_LOG" | tail -5 | while read line; do
                echo "  $line"
            done || echo "  (No alerts)"
            echo ""
            
            # Queue stats
            echo "Trade Queue Stats:"
            grep "QUEUE" "$LATEST_LOG" | tail -3 | while read line; do
                echo "  $line"
            done || echo "  (No queue stats)"
            echo ""
            
            # Candle completions
            echo "Recent Candle Completions:"
            grep "1m candle complete" "$LATEST_LOG" | tail -5 | while read line; do
                echo "  $line"
            done || echo "  (No candles yet)"
            echo ""
        fi
        
        # Database size
        if [ -f "market_data.duckdb" ]; then
            DB_SIZE=$(du -h market_data.duckdb | awk '{print $1}')
            echo "Database Size: $DB_SIZE"
        fi
        
        # Disk space
        echo "Disk Space:"
        df -h . | tail -1 | awk '{printf "  Used: %s / %s (%s full)\n", $3, $2, $5}'
        
    else
        echo "❌ NOT RUNNING (stale PID file)"
        echo "  Clean up: rm $PID_FILE"
    fi
else
    # Check with pgrep
    if pgrep -f "integrated_collector" > /dev/null; then
        PID=$(pgrep -f "integrated_collector" | head -1)
        echo "⚠️  RUNNING but no PID file (PID: $PID)"
        echo "  Create PID file: echo $PID > $PID_FILE"
    else
        echo "❌ NOT RUNNING"
        echo ""
        echo "Start collector: ./scripts/deploy/start_collector.sh"
    fi
fi

echo ""
echo "========================================"
