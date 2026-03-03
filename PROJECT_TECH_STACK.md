# Cat & Dog Image Classifier — Technology Stack & Architecture

This document explains the technologies used in this project and where each one appears in the codebase.

---

## 1) Python (Runtime Language)

### What it is
Python is the primary language used to implement model training, command-line prediction, and the desktop GUI.

### Where it is used
- Main source scripts:
  - `src/train_cats_dogs.py`
  - `src/predict_cats_dogs.py`
  - `src/gui_cats_dogs.py`
- Project setup requires Python 3.8+.



---

## 2) TensorFlow (Core ML Framework)

### What it is
TensorFlow is the deep learning framework used to train the model and run inference.

### Where it is used
- Dependency: `requirements.txt`
- Imported in train/predict/gui modules.
- Used for model loading and prediction operations (for example, `tf.keras.models.load_model`, `tf.nn.softmax`).



---

## 3) Keras (via TensorFlow Keras APIs)

### What it is
Keras (inside TensorFlow) provides high-level APIs to define layers, compile models, and run training loops.

### Where it is used
- Mentioned in technical details (`TensorFlow / Keras`).
- Training script uses:
  - `ImageDataGenerator` for data loading and augmentation
  - `Dense`, `GlobalAveragePooling2D` for classification head
  - `Model`, `Adam` for model creation and optimization



---

## 4) MobileNetV2 (Transfer Learning Backbone)

### What it is
MobileNetV2 is a pretrained convolutional neural network used as the base model for transfer learning.

### Where it is used
- Mentioned in README as the base model.
- Training code initializes `MobileNetV2(..., weights='imagenet', include_top=False)` and freezes it initially.



---

## 5) NumPy (Numerical Operations)

### What it is
NumPy is used for array operations during image preprocessing and result interpretation.

### Where it is used
- Dependency: `requirements.txt`
- GUI preprocessing: convert images to arrays and expand batch dimension.
- CLI prediction: `argmax` / max confidence computation.



---

## 6) Pillow / PIL (Image Handling)

### What it is
Pillow is the image-processing library used to open, convert, resize, and preview images.

### Where it is used
- Dependency: `requirements.txt`
- GUI imports `Image` and `ImageTk` for:
  - opening user-selected image files
  - RGB conversion and resizing
  - preview rendering in the app



---

## 7) CustomTkinter (Desktop GUI Framework)

### What it is
CustomTkinter is a modern UI layer built on top of Tkinter for better-looking desktop applications.

### Where it is used
- Dependency: `requirements.txt`
- README identifies it as GUI tech.
- GUI code builds the interface with `CTk`, `CTkFrame`, `CTkLabel`, and `CTkButton`.



---

## 8) Tkinter (Dialogs and Message Boxes)

### What it is
Tkinter is Python’s standard GUI toolkit, used here for utility dialogs.

### Where it is used
- GUI imports `filedialog` and `messagebox`.
- Used for:
  - selecting an image from disk
  - displaying user-facing error messages



---

## Architecture Slide (Ready to Copy)

1. User uploads an image in the GUI (CustomTkinter + Tkinter dialogs).  
2. The image is processed (Pillow + NumPy).  
3. Model inference runs (TensorFlow/Keras).  
4. Prediction + confidence are displayed with threshold logic (Cat/Dog/Unknown).  
5. The model is trained separately via transfer learning with MobileNetV2.

---

## X

> “This is a Python desktop AI application for cat-vs-dog image classification. The model pipeline is built with TensorFlow and Keras, using MobileNetV2 transfer learning for efficient training. In the interface layer, CustomTkinter provides a modern desktop GUI, while Tkinter utilities handle file pickers and error dialogs. Pillow and NumPy manage image preprocessing before inference. The output is a predicted class with confidence, and low-confidence results are labeled as Unknown for safer behavior.”

---

## Source Pointers

- `README.md`
- `requirements.txt`
- `src/train_cats_dogs.py`
- `src/predict_cats_dogs.py`
- `src/gui_cats_dogs.py`
