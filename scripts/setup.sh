#!/usr/bin/env bash
# set -e

VENV=".venv"

if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV

    echo "Installing development dependencies..."
    $VENV/bin/pip install --upgrade pip
    $VENV/bin/pip install -e ".[dev]"
else
    echo "Virtual environment already exists."
fi

echo "Activating environment..."
source $VENV/bin/activate
