# Load environment variables from .env file
set -a
source local.env
set +a

# Activate the virtual environment only if not already active
if [ -z "$VIRTUAL_ENV" ]; then
    source venv/bin/activate
fi

# Run the main.py file
python3 src/main.py
