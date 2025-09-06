import tensorflow as tf
import numpy as np
import os

# Same image size as training
img_height, img_width = 128, 128

# Load the trained model
# (Make sure you saved the model at the end of training as model.save("saved_model"))
model = tf.keras.models.load_model("saved_model.keras")


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
