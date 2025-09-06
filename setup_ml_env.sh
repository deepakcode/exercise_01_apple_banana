#!/usr/bin/env bash
set -e

# --- Config ---
VENV_NAME="my_ml_env"
PY_VERSION="3.11"

echo "=== Setting up ML environment on Intel macOS ==="

# 1. Ensure Homebrew is up to date and Python 3.11 is installed
echo "[Step 1/5] Installing Python ${PY_VERSION} via Homebrew..."
brew update
brew install python@${PY_VERSION} || true

# 2. Create virtual environment
echo "[Step 2/5] Creating virtual environment: ${VENV_NAME}"
python3.11 -m venv ${VENV_NAME}

# 3. Activate environment
echo "[Step 3/5] Activating virtual environment"
source ${VENV_NAME}/bin/activate

# 4. Upgrade pip and install dependencies
echo "[Step 4/5] Upgrading pip and installing requirements..."
python -m ensurepip --upgrade
pip install --upgrade pip setuptools wheel

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "ERROR: requirements.txt not found in current directory!"
    exit 1
fi

# 5. Verify installation
echo "[Step 5/5] Verifying TensorFlow installation..."
python -c "import sys, tensorflow as tf; print('Python', sys.version.split()[0]); print('TensorFlow', tf.__version__)"

echo "=== Setup complete! ==="
echo "Activate your environment anytime with: source ${VENV_NAME}/bin/activate"
