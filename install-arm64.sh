#!/usr/bin/env bash
# Setup script for Debian 11 (Bullseye) ARM64 — Tinker Board 2S
# Requires: glibc 2.31+, Python 3.9 (provided by Debian 11 apt)
set -euo pipefail

VENV_DIR="$(dirname "$0")/.venv-arm64"

echo "==> Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-venv \
    python3.9-dev \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6

echo "==> Creating virtual environment at $VENV_DIR..."
python3.9 -m venv "$VENV_DIR"

echo "==> Upgrading pip..."
"$VENV_DIR/bin/pip" install --upgrade pip

echo "==> Installing Python dependencies (ARM64 wheels)..."
"$VENV_DIR/bin/pip" install \
    "numpy>=1.23,<2.0" \
    "scipy>=1.9" \
    "opencv-python==4.9.0.80" \
    "tflite-runtime==2.13.0"

echo ""
echo "Setup complete."
echo "Run the app with:"
echo "  $VENV_DIR/bin/python main.py"
