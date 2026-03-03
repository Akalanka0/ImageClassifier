import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

# -----------------------------
# Load model
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "models" / "cats_dogs_model.keras"

if not model_path.exists():
    raise FileNotFoundError(f"Model not found at {model_path}")

model = tf.keras.models.load_model(model_path)
# Use same class naming as the GUI for consistency
class_names = ['Cat', 'Dog']

def predict_local_image(img_path):
    img_path = Path(img_path)
    if not img_path.exists():
        print(f"Image not found at {img_path}. Please provide a valid image path.")
        return

    img = image.load_img(img_path, target_size=(160, 160))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    pred_logits = model.predict(img_array)
    score = tf.nn.softmax(pred_logits[0])
    confidence = float(np.max(score))
    predicted_class = class_names[int(np.argmax(score))]

    threshold = 0.7
    if confidence < threshold:
        print("Prediction: Unknown object")
    else:
        print(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict cat vs dog image")
    parser.add_argument("-i", "--image", type=str,
                        help="Path to image file. If omitted uses data/test_image.jpg",
                        default=str(BASE_DIR / "data" / "test_image.jpg"))
    args = parser.parse_args()
    predict_local_image(args.image)
