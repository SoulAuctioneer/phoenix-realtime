# Remove existing virtual environment if it exists
if [ -d "venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf venv
fi

echo "Creating virtual environment..."
python3 -m venv venv

# Update package list
sudo apt-get update

# Check and install portaudio19-dev if needed
if ! dpkg -l | grep -q portaudio19-dev; then
    echo "Installing portaudio19-dev..."
    sudo apt-get install -y portaudio19-dev
else
    echo "portaudio19-dev is already installed"
fi

# Check and install libatlas-base-dev if needed
if ! dpkg -l | grep -q libatlas-base-dev; then
    echo "Installing libatlas-base-dev..."
    sudo apt-get install -y libatlas-base-dev
else
    echo "libatlas-base-dev is already installed"
fi

# Check and install libopenblas0 if needed
if ! dpkg -l | grep -q libopenblas0; then
    echo "Installing libopenblas0..."
    sudo apt-get install -y libopenblas0
else
    echo "libopenblas0 is already installed"
fi

# Install the required packages
./venv/bin/pip install -r requirements.txt
