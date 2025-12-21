# src/gui_cats_dogs_modern.py
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os

# =========================
# Load trained model
# =========================
model_path = r"D:\Test\ImageClassifier\models\cats_dogs_model.keras"
try:
    model = load_model(model_path)
    print(f"Model loaded from '{model_path}'")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Image size expected by the model
img_width, img_height = 160, 160

# =========================
# Prediction function
# =========================
def predict_image(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize((img_width, img_height))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  

        pred = model.predict(img_array)
        probs = tf.nn.softmax(pred, axis=1).numpy()
        class_idx = probs.argmax()
        confidence = probs[0][class_idx]

        class_names = ['Cat', 'Dog']
        threshold = 0.55  

        if confidence < threshold:
            return "Unknown object", confidence
        else:
            return class_names[class_idx], confidence
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed: {e}")
        return None, None

# =========================
# GUI functions
# =========================
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        try:
            img = Image.open(file_path)
            img.thumbnail((300, 300))
            img_tk = ImageTk.PhotoImage(img)
            img_label.configure(image=img_tk)
            img_label.image = img_tk

            pred_class, confidence = predict_image(file_path)
            if pred_class:
                result_label.configure(text=f"{pred_class} (Confidence: {confidence:.2f})")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {e}")

# =========================
# Modern GUI Layout
# =========================
ctk.set_appearance_mode("Dark")  
ctk.set_default_color_theme("blue") 

app = ctk.CTk()
app.geometry("450x550")
app.title("Cat vs Dog Classifier")
app.resizable(False, False)

# Title
title_label = ctk.CTkLabel(app, text="Cat vs Dog Classifier", font=ctk.CTkFont(size=20, weight="bold"))
title_label.pack(pady=20)

# Upload Button
upload_btn = ctk.CTkButton(app, text="Upload Image", command=upload_image, width=200, height=40, corner_radius=10)
upload_btn.pack(pady=10)

# Image display
img_label = ctk.CTkLabel(app, text="Image Preview", width=300, height=300, corner_radius=10)
img_label.pack(pady=15)

# Prediction result
result_label = ctk.CTkLabel(app, text="Prediction will appear here", font=ctk.CTkFont(size=16))
result_label.pack(pady=20)

app.mainloop()
