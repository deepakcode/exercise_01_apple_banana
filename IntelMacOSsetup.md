# README — Machine learning on an Intel macOS (MacBook Pro 16-inch, 2019)

This README contains step-by-step instructions for setting up and using a TensorFlow-based machine learning environment on your Intel Mac. It also includes an automated setup script (`setup_ml_env.sh`) that will recreate your environment exactly from the provided `requirements.txt`.

- **Machine:** MacBook Pro (16-inch, 2019) — Intel Core i7/i9 (6-core), 16 GB RAM
- **macOS:** Sequoia 15.5
- **Python:** Use **Python 3.11** (TensorFlow is not supported on Python 3.12+)
- **TensorFlow:** CPU build for Intel macs (confirmed working: `2.16.2`)

---

## 1. Quick Start (Recommended)
If you want a one-shot setup with no manual steps, use the provided **setup script**. It installs Python 3.11 (via Homebrew), creates a virtual environment, installs everything from `requirements.txt`, and verifies TensorFlow.

### Run setup script
```bash
chmod +x setup_ml_env.sh
./setup_ml_env.sh
```

After it finishes, activate your environment:
```bash
source my_ml_env/bin/activate
```

That’s it — you now have TensorFlow, NumPy, Matplotlib, and Scikit-learn installed.

---

## 2. Manual Setup (if you prefer step-by-step)

### Install Homebrew
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Install Python 3.11
```bash
brew install python@3.11
```

### Create a Virtual Environment
```bash
python3.11 -m venv my_ml_env
source my_ml_env/bin/activate
```

### Upgrade pip and install dependencies
```bash
python -m ensurepip --upgrade
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Verify TensorFlow installation
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

You should see `2.16.2` (or similar).

---

## 3. Training a Simple CNN
Once the environment is ready, you can use the provided sample script (`train_model.py`) to train a CNN on images of apples and bananas. It demonstrates:
- Loading images using `image_dataset_from_directory`
- Defining a CNN with convolution, pooling, and dense layers
- Training for 10 epochs
- Plotting training/validation accuracy and loss with Matplotlib
- Running inference on new images

---

## 4. Troubleshooting

### `zsh: command not found: pip`
Make sure you activated the virtual environment. If still an issue, run:
```bash
python -m pip install --upgrade pip
```

### Python 3.13 or 3.12 installed
TensorFlow currently does not support 3.12+ on macOS. Always use Python 3.11.

### AVX2/FMA warning
This is an informational log only. You can ignore it — TensorFlow works fine without CPU-specific optimizations.

### Apple Silicon notes
Do NOT install `tensorflow-macos` or `tensorflow-metal` on Intel Macs. Those are only for M1/M2/M3.

---

## 5. Reproducibility
- Your exact working environment is captured in `requirements.txt`. 
- To recreate the same setup elsewhere:
```bash
python3.11 -m venv my_ml_env
source my_ml_env/bin/activate
pip install -r requirements.txt
```

---

## 6. Next Steps
- Use the `setup_ml_env.sh` script whenever you need a clean environment.
- Train your own dataset using the CNN example in `train_model.py`.
- Add new libraries to `requirements.txt` as your project grows.

---

This guide ensures that your Intel Mac is correctly set up for TensorFlow-based ML projects, with both an automated and manual path available.

