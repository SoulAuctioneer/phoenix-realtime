# Create a virtual environment
python3 -m venv venv

# Check and install portaudio if needed
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if ! brew list portaudio >/dev/null 2>&1; then
        echo "Installing portaudio..."
        brew install portaudio
    else
        echo "portaudio is already installed"
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if ! dpkg -l | grep -q libportaudio2; then
        echo "Installing portaudio..."
        sudo apt-get install -y libportaudio-dev
    else
        echo "portaudio is already installed"
    fi
elif [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "cygwin"* ]]; then
    # Windows
    echo "On Windows, PortAudio is bundled with PyAudio. No separate installation needed."
fi

# Install the required packages
./venv/bin/pip install -r requirements.txt
