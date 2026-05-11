#!/usr/bin/env python3
"""
Wrapper script to run spike documentation for 120 symbols × 27 days.
Reads symbols from database to avoid command-line length issues.
"""

import sqlite3
import subprocess
import sys

# Get top 120 symbols from database
conn = sqlite3.connect('market_data copy_86.db')
cursor = conn.cursor()
cursor.execute('SELECT symbol FROM order_book_features GROUP BY symbol ORDER BY COUNT(*) DESC LIMIT 120')
symbols = [row[0] for row in cursor.fetchall()]
conn.close()

symbols_str = ','.join(symbols)

# Generate date range
dates = [f'2025-09-{d:02d}' for d in range(24, 31)] + \
        [f'2025-10-{d:02d}' for d in range(1, 21)]
dates_str = ','.join(dates)

print(f"Running spike documentation for {len(symbols)} symbols × {len(dates)} days...")
print()

# Run the documentation script
cmd = [
    'python', 'scripts/document_spikes.py',
    '--db', 'market_data copy_86.db',
    '--symbols', symbols_str,
    '--dates', dates_str,
    '--spike-threshold', '0.05',
    '--lookahead', '10',
    '--output-md', 'SPIKE_EVENTS.md',
    '--output-json', 'data/spike_events.json'
]

sys.exit(subprocess.call(cmd))
