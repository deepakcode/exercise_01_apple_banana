Perfect 👍 It’s a good practice to keep **training** and **testing/inference** separate.
Here’s a clean `test_model.py` script you can drop into your project.

---

## 🔹 `test_model.py`

```python
import tensorflow as tf
import numpy as np
import os

# Same image size as training
img_height, img_width = 128, 128

# Load the trained model
# (Make sure you saved the model at the end of training as model.save("saved_model"))
model = tf.keras.models.load_model("saved_model")

# Get class names (must match training order)
# Either hardcode or load from training
class_names = ['apples', 'bananas']

# Path to test images folder
test_dir = "test_images"

# Loop through test images
for fname in os.listdir(test_dir):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(test_dir, fname)

        # Load image
        img = tf.keras.preprocessing.image.load_img(path, target_size=(img_height, img_width))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, 0) / 255.0  # normalize

        # Predict
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(f"{fname} → {class_names[np.argmax(score)]} ({100*np.max(score):.2f}%)")
```

---

## 🔹 Update your training script (`train_model.py`)

At the very end of training, add:

```python
# Save the trained model
model.save("saved_model")
print("✅ Model saved to 'saved_model/' folder")
```

This ensures you can reload the model in `test_model.py`.

---

## 🔹 Usage

1. Train your model (once):

   ```bash
   python train_model.py
   ```

   It will save the model to `saved_model/`.

2. Put your test images in `test_images/` (e.g. `apple1.jpg`, `banana1.jpg`).

3. Run testing:

   ```bash
   python test_model.py
   ```

Example output:

```
apple1.jpg → apples (97.35%)
banana1.jpg → bananas (99.12%)
```

---

👉 Do you want me to also extend this `test_model.py` so it **plots the test image along with the predicted label & confidence** (like a mini visual demo)?
