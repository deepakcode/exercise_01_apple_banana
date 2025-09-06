# Helloworld AI/ML — Image Classification (Apples vs Bananas)

Building a machine learning model from scratch on your Mac is an excellent way to learn the entire process. Here is a detailed, step-by-step guide covering everything from setting up your environment to training and testing your model.

This is a beginner-friendly machine learning project for image classification, built to run on **Intel macOS** (tested on MacBook Pro 16-inch, 2019). The project uses **TensorFlow (CPU build)** along with NumPy, Matplotlib, and Scikit-learn.

The goal: train a simple **Convolutional Neural Network (CNN)** to classify images of apples and bananas.

---

## 1. Requirements

* **Machine:** Intel MacBook Pro (16-inch, 2019), 16 GB RAM
* **macOS:** Sequoia 15.5 (Intel, not Apple Silicon)
* **Python:** 3.11 (TensorFlow does not support 3.12+ yet)
* **Libraries:** See `requirements.txt` for full list

  * TensorFlow `2.16.2`
  * NumPy `1.26.4`
  * Matplotlib `3.10.6`
  * Scikit-learn `1.7.1`

> ⚠️ **Important:** Do NOT install `tensorflow-macos` or `tensorflow-metal` on Intel Macs (those are for M1/M2/M3). Use the CPU build only.

---

## 2. Setup Instructions

### Option A — Quick Setup (Recommended)

Run the automated setup script:

```bash
chmod +x setup_ml_env.sh
./setup_ml_env.sh
```

This will:

* Install Python 3.11 via Homebrew
* Create a virtual environment `my_ml_env`
* Install dependencies from `requirements.txt`
* Verify TensorFlow installation

Activate the environment:

```bash
source my_ml_env/bin/activate
```

---

### Option B — Manual Setup

1. Install [Homebrew](https://brew.sh):

   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. Install Python 3.11:

   ```bash
   brew install python@3.11
   ```

3. Create and activate a virtual environment:

   ```bash
   python3.11 -m venv my_ml_env
   source my_ml_env/bin/activate
   ```

4. Upgrade pip and install requirements:

   ```bash
   python -m ensurepip --upgrade
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

5. Verify TensorFlow:

   ```bash
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```

   You should see `2.16.2`.

* **Install Necessary Libraries:** Now, install the core libraries you'll need.

  * **TensorFlow**: The main machine learning framework for building and training your model.
  * **NumPy**: A fundamental library for numerical operations, used for handling data arrays.
  * **Matplotlib**: A plotting library to visualize your data and model performance.
  * **Scikit-learn**: A library with useful tools for data preprocessing and evaluation.

---

## 3. The Data: Images of Apples and Bananas

To train a model, you need a dataset. You have two main options:

1. **Find a Public Dataset:**
   This is the easiest method. Websites like Kaggle and TensorFlow Datasets offer pre-collected and cleaned image datasets. You can search for "apple and banana dataset" or "fruit classification dataset" to find a suitable one.

2. **Scrape Your Own Data:**
   This is a bit more involved but gives you control. You can use Python scripts to download images from the web using keywords like "ripe apple" or "green banana." Just be mindful of image licensing.

---

### Dataset Options

1. **Public dataset (recommended):**

   * Kaggle dataset: [Fruit Images for Object Detection](https://www.kaggle.com/datasets/mbkinaci/fruit-images-for-object-detection?resource=download)
   * Or download via KaggleHub:

     ```python
     import kagglehub
     path = kagglehub.dataset_download("mbkinaci/fruit-images-for-object-detection")
     print("Path to dataset files:", path)
     ```

2. **Custom dataset:**
   Create a folder structure like this:

   ```
   data/
     apples/
       apple1.jpg
       apple2.jpg
     bananas/
       banana1.jpg
       banana2.jpg
   ```

---

## 4. The Model's Internals and Training Process

A machine learning model for image classification, like the one we'll build, is typically a **Convolutional Neural Network (CNN)**.

A CNN works by passing an image through several layers:

1. **Input Layer:** Raw image data (pixels).
2. **Convolutional Layer:** Detects features (edges, shapes, etc.).
3. **Pooling Layer:** Downsamples feature maps for efficiency.
4. **Flatten Layer:** Converts 2D maps into a 1D vector.
5. **Dense (Fully-Connected) Layer:** Combines features for classification.
6. **Output Layer:** Final prediction (apple vs banana).

---

## 5. Building, Training, and Testing the Model

Save the following code as `train_model.py`:

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Load and preprocess the data
img_height, img_width = 128, 128
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  'data',
  validation_split=0.2,
  subset='training',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  'data',
  validation_split=0.2,
  subset='validation',
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

class_names = train_ds.class_names
print(f"Class names: {class_names}")

# 2. Build the model
model = models.Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(len(class_names), activation='softmax')
])

# 3. Compile
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

# 4. Train
epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# 5. Evaluate
acc, val_acc = history.history['accuracy'], history.history['val_accuracy']
loss, val_loss = history.history['loss'], history.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```

---

## 6. Training the Model

Run:

```bash
python train_model.py
```

This will:

* Load images from `data/`
* Train for 10 epochs
* Plot accuracy/loss
* Save the model

At the end, add:

```python
model.save("saved_model")
print("✅ Model saved to 'saved_model/' folder")
```

---

## 7. Testing the Model

Run:

```bash
python test_model.py
```

Steps:

1. Place test images in `test_images/`
   Example: `test_images/apple1.jpg`, `test_images/banana1.jpg`
2. Run script → Example output:

   ```
   apple1.jpg → apples (97.35%)
   banana1.jpg → bananas (99.12%)
   ```

---

## 8. Troubleshooting

**Common Issues:**

* **`zsh: command not found: pip`** → Activate venv or run `python -m pip install --upgrade pip`
* **Installed Python 3.12 or 3.13** → Not supported, use 3.11
* **AVX2/FMA warnings** → Ignore
* **Git push rejected (large files)** → Remove venv, add `.gitignore`

> ✅ Only commit **source code + requirements.txt**, never the entire venv.

---

## 9. Next Steps

* Add more fruit classes
* Use data augmentation (`ImageDataGenerator`)
* Try transfer learning (MobileNet, ResNet)
* Visualize predictions with matplotlib
* Share results without committing large files

---

