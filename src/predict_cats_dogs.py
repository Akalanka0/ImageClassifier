import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from pathlib import Path

# -----------------------------
# Load model
# -----------------------------
model_path = Path("models/cats_dogs_model.keras")
model = tf.keras.models.load_model(model_path)
class_names = ['cats', 'dogs']
print(f"Model loaded from {model_path}")

# -----------------------------
# Load image
# -----------------------------
img_path = r"C:\Users\USER\Downloads\download (4).jpg"
img = image.load_img(img_path, target_size=(160, 160))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# -----------------------------
# Predict
# -----------------------------
pred_logits = model.predict(img_array)
score = tf.nn.softmax(pred_logits[0])
confidence = np.max(score)
predicted_class = class_names[np.argmax(score)]

# -----------------------------
# Unknown threshold
# -----------------------------
threshold = 0.7
if confidence < threshold:
    print("Prediction: Unknown object")
else:
    print(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")
