# README — Machine learning on an Intel macOS (MacBook Pro 16-inch, 2019)

This README replaces the previous generic macOS instructions with tested, copy-paste commands and troubleshooting tips for your exact setup:

- **Machine:** MacBook Pro (16-inch, 2019) — Intel Core i7/i9 (6-core), 16 GB RAM
- **macOS:** Sequoia 15.5
- **Python:** Use **Python 3.11** (TensorFlow is not supported on Python 3.12+)
- **TensorFlow:** CPU build for Intel macs (example here uses `tensorflow` — confirmed working: `2.16.2` on your machine)

---

## 1. Overview / Goal
Get a reproducible development environment for training simple TensorFlow models on an Intel mac. We will create a virtual environment, install Python 3.11 (via Homebrew), ensure `pip` is available, install TensorFlow and the usual ML libraries, and verify everything works with a tiny test script.

---

## 2. Prerequisites
- Homebrew installed (recommended). If you don't have it:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

- You should not try to install TensorFlow with **system Python** or an unsupported Python (3.12+). Use Python 3.11.

---

## 3. Step-by-step commands (copy-paste)
Run these commands in Terminal. They install Python 3.11, create a venv, and install the required packages.

```bash
# 1) Install Python 3.11 via Homebrew
brew update
brew install python@3.11

# 2) Confirm Python 3.11 is available
python3.11 --version

# 3) Create a virtual environment (in project folder or home)
# from your project directory (recommended):
python3.11 -m venv my_ml_env

# 4) Activate the virtual environment
source my_ml_env/bin/activate

# 5) Ensure pip is present and up-to-date
python -m ensurepip --upgrade
pip install --upgrade pip setuptools wheel

# 6) Install TensorFlow and other libraries
# Pin to a compatible range; this works for Intel mac + Python 3.11
pip install "tensorflow<2.19" numpy matplotlib scikit-learn

# 7) Quick verification (should print TF version, e.g. 2.16.2)
python -c "import sys, tensorflow as tf; print('Python', sys.version.split()[0]); print('TensorFlow', tf.__version__)"
```

If you already have a working venv (like `my_ml_env`) and `python -c "import tensorflow as tf; print(tf.__version__)"` printed `2.16.2` as you showed, you're good to go.

---

## 4. One-shot setup script (optional)
Save the following as `setup_ml_env.sh` and run it. It automates the steps above.

```bash
#!/usr/bin/env bash
set -e

# Install Python 3.11 (Homebrew must be installed first)
brew update
brew install python@3.11

# Create venv in current directory
python3.11 -m venv my_ml_env
source my_ml_env/bin/activate
python -m ensurepip --upgrade
pip install --upgrade pip setuptools wheel

# Install ML libraries (cpu TensorFlow for Intel Macs)
pip install "tensorflow<2.19" numpy matplotlib scikit-learn

# Final check
python -c "import sys, tensorflow as tf; print('Python', sys.version.split()[0]); print('TensorFlow', tf.__version__)"

echo "\nSetup complete. Activate with: source my_ml_env/bin/activate"
```

To run:

```bash
chmod +x setup_ml_env.sh
./setup_ml_env.sh
```

---

## 5. Quick test script (to confirm training works)
Save as `quick_test_tf.py` then run `python quick_test_tf.py` inside the activated venv.

```python
# quick_test_tf.py
import sys
import numpy as np
import tensorflow as tf

print('Python', sys.version.split()[0])
print('TensorFlow', tf.__version__)

# tiny synthetic dataset
x = np.random.random((128, 10)).astype('float32')
y = np.random.randint(0, 2, (128, 1)).astype('float32')

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x, y, epochs=2, batch_size=16)
print('Quick test finished')
```

This does a short 2-epoch run on random data and confirms end-to-end functionality.

---

## 6. Common problems & fixes

### `zsh: command not found: pip`
- You probably activated a Python installation that doesn't expose `pip` on PATH. Use the venv activation above, then run `python -m pip` if `pip` is still not found:

```bash
python -m pip install --upgrade pip
```

### You have Python 3.13 (or 3.12+)
- TensorFlow currently does not support Python versions newer than 3.11 on macOS. Install Python 3.11 via Homebrew and create a venv from it (`python3.11 -m venv ...`).

### `This TensorFlow binary is optimized to use available CPU instructions... AVX2 FMA` (INFO)
- This is an informational message. The prebuilt TensorFlow you installed is functional on Intel CPUs but was not compiled with some CPU-specific optimizations. You can ignore this unless you want to compile TensorFlow from source to squeeze out a small additional CPU speed improvement (this is advanced and time-consuming).

### GPU / Apple Silicon notes
- **Intel Mac (your machine)**: `tensorflow-metal` and `tensorflow-macos` are *for Apple Silicon* (M1/M2/M3). Do NOT install `tensorflow-macos` on Intel mac.
- If you *ever* move to Apple Silicon, the recommended packages differ.

---

## 7. Tips & next steps
- Use `python -m pip` inside the venv for absolute reliability.
- Keep your venv per-project to avoid dependency conflicts.
- If you want a reproducible environment for sharing, create a `requirements.txt` after you install packages:

```bash
pip freeze > requirements.txt
```

- To recreate elsewhere:

```bash
python3.11 -m venv my_ml_env
source my_ml_env/bin/activate
pip install -r requirements.txt
```

---

If you'd like, I can:
- Add this README as an actual `README.md` file in your project folder (provide path),
- Produce a `requirements.txt` pinned to the versions on your machine (I can detect versions if you run `pip freeze` and paste the output), or
- Expand the test script to load your apples/bananas dataset and run a small training job using `image_dataset_from_directory` (the example in your original README).

