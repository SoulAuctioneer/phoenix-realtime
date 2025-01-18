#!/bin/bash

# Exit on error
set -e

# Remove existing virtual environment if it exists
if [ -d "venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf venv
fi

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install or upgrade pip
python3 -m pip install --upgrade pip

# Install system dependencies based on OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS specific installations
    if ! brew list portaudio &>/dev/null; then
        echo "Installing portaudio via Homebrew..."
        brew install portaudio
    fi
elif [[ "$OSTYPE" == "linux"* ]]; then
    # Linux specific installations
    sudo apt-get update
    
    # Install required system packages
    packages=(ffmpeg portaudio19-dev libatlas-base-dev libopenblas0)
    for package in "${packages[@]}"; do
        if ! dpkg -l | grep -q $package; then
            echo "Installing $package..."
            sudo apt-get install -y $package
        else
            echo "$package is already installed"
        fi
    done
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt
echo "Installation complete!"

