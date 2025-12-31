import customtkinter as ctk
from tkinter import filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import ctypes

# =========================
# Load trained model
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "models" / "cats_dogs_model.keras"

model = None

def load_model_safely():
    global model
    try:
        model = load_model(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        messagebox.showerror("Model Error", f"Could not load model from:\n{model_path}\n\nPlease train the model or check the path.\n\nError: {e}")



IMG_WIDTH, IMG_HEIGHT = 160, 160

# =========================
# Prediction function
# =========================
def predict_image(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if model is None:
             messagebox.showerror("Error", "Model is not loaded.")
             return None, None

        preds = model.predict(img_array)
        probs = tf.nn.softmax(preds, axis=1).numpy()
        idx = np.argmax(probs)
        confidence = probs[0][idx]

        class_names = ["Cat", "Dog"]
        threshold = 0.55

        if confidence < threshold:
            return "Unknown", confidence
        return class_names[idx], confidence

    except Exception as e:
        messagebox.showerror("Error", str(e))
        return None, None

# =========================
# Upload image function
# =========================
def upload_image():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not path:
        return
    try:
        img = Image.open(path)
        img.thumbnail((340, 340))
        img_tk = ImageTk.PhotoImage(img)

        img_label.configure(image=img_tk, text="")
        img_label.image = img_tk

        result, conf = predict_image(path)
        if result:
            result_label.configure(text=f"{result}\nConfidence: {conf:.2f}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# =========================
# Fullscreen toggle
# =========================
is_fullscreen = False

def toggle_fullscreen(event=None):
    global is_fullscreen
    is_fullscreen = not is_fullscreen
    app.attributes("-fullscreen", is_fullscreen)

def exit_fullscreen(event=None):
    global is_fullscreen
    is_fullscreen = False
    app.attributes("-fullscreen", False)

# ==============
# Center window 
# ==============
def center_window():
    hwnd = ctypes.windll.user32.GetForegroundWindow()
    screen_width = ctypes.windll.user32.GetSystemMetrics(0)
    screen_height = ctypes.windll.user32.GetSystemMetrics(1)

    x = (screen_width - WINDOW_WIDTH) // 2
    y = (screen_height - WINDOW_HEIGHT) // 2 

    ctypes.windll.user32.MoveWindow(hwnd, x, y, WINDOW_WIDTH, WINDOW_HEIGHT, True)

# =========================
# Modern GUI setup
# =========================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Cat vs Dog Classifier")

WINDOW_WIDTH = 520
WINDOW_HEIGHT = 950
app.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
app.resizable(True, True)

# Keyboard shortcuts
app.bind("<F11>", toggle_fullscreen)
app.bind("<Escape>", exit_fullscreen)

# =========================
# UI Elements
# =========================
main_frame = ctk.CTkFrame(app)
main_frame.pack(fill="both", expand=True, padx=20, pady=20)

title_label = ctk.CTkLabel(
    main_frame,
    text="🐱 Cat vs Dog Classifier 🐶",
    font=ctk.CTkFont(size=22, weight="bold")
)
title_label.pack(pady=20)

subtitle_label = ctk.CTkLabel(
    main_frame,
    text="Upload an image to classify",
    font=ctk.CTkFont(size=14),
    text_color="gray"
)
subtitle_label.pack(pady=(0,15))

upload_btn = ctk.CTkButton(
    main_frame,
    text="Upload Image",
    width=220,
    height=45,
    corner_radius=12,
    command=upload_image
)
upload_btn.pack(pady=10)

img_label = ctk.CTkLabel(
    main_frame,
    text="Image Preview",
    width=340,
    height=340,
    corner_radius=15,
    fg_color=("gray20","gray15")
)
img_label.pack(pady=15)

result_label = ctk.CTkLabel(
    main_frame,
    text="Prediction will appear here",
    font=ctk.CTkFont(size=18)
)
result_label.pack(pady=20)

hint_label = ctk.CTkLabel(
    main_frame,
    text="F11 = Fullscreen | ESC = Exit",
    font=ctk.CTkFont(size=12),
    text_color="gray"
)
hint_label.pack(pady=10)


app.after(100, center_window)
# Load model after UI is ready to show error if needed
app.after(500, load_model_safely)

if __name__ == "__main__":
    app.mainloop()