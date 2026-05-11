#!/usr/bin/env python3
"""
Extract private key from Coinbase API JSON file and save as PEM.
Usage: python extract_coinbase_key.py /path/to/downloaded_key.json
"""
import sys
import json
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python extract_coinbase_key.py /path/to/coinbase_key.json")
    sys.exit(1)

json_path = Path(sys.argv[1])
if not json_path.exists():
    print(f"Error: File not found: {json_path}")
    sys.exit(1)

# Read JSON
with open(json_path) as f:
    data = json.load(f)

# Extract private key
private_key = data.get('privateKey')
if not private_key:
    print("Error: 'privateKey' not found in JSON file")
    sys.exit(1)

# Save as PEM file
pem_path = json_path.parent / "coinbase_private_key.pem"
with open(pem_path, 'w') as f:
    f.write(private_key)

print(f"✓ Private key extracted to: {pem_path}")
print(f"✓ Set this in .env: COINBASE_PRIVATE_KEY_PATH={pem_path}")
