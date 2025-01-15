# Activate the virtual environment
source venv/bin/activate

# Load environment variables from .env file
set -a
source local.env
set +a

# Run the main.py file
python3 src/main.py
