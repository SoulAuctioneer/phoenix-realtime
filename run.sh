#!/bin/bash

# Exit on error
set -e

# Activate the virtual environment
source venv/bin/activate

# Load environment variables from .env file
set -a
source local.env 2>/dev/null || true  # Don't fail if file doesn't exist
set +a

# Run the main.py file
echo "Starting application..."
python3 src/main.py
