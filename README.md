# Cat & Dog Image Classifier

A deep learning desktop application built with **TensorFlow** and **CustomTkinter** that classifies images as **Cat**, **Dog**, or **Unknown** based on prediction confidence.

This project is designed mainly for **learning, experimentation, and academic use**.

---

## 📋 Features

- **Transfer Learning**: Uses MobileNetV2 pre-trained on ImageNet  
- **Modern GUI**: Built with CustomTkinter for a sleek dark-mode interface  
- **Confidence Threshold**: Identifies "unknown" objects when confidence is low  
- **Easy to Use**: Simple image selection via file dialog  
- **GUI and CLI Support**: Run with a graphical interface or command line  

---

## 🚀 Setup

### 0️⃣ Create & Activate Virtual Environment (Recommended)

**Windows PowerShell:**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**Command Prompt (cmd.exe):**
```cmd
python -m venv venv (one time)
venv\Scripts\activate
```

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

**Note:** Python 3.8 or higher is required (tested on Python 3.9+).

### 2️⃣ Prepare Training Data (Optional)

If you want to train your own model, create the following structure:
```
data/
├── cats/
│   ├── cat1.jpg
│   ├── cat2.jpg
│   └── ...
└── dogs/
    ├── dog1.jpg
    ├── dog2.jpg
    └── ...
```

**Tip:** At least 100 images per class is recommended for reasonable accuracy.

### 3️⃣ Train the Model (Optional)
```bash
python src/train_cats_dogs.py
```

The trained model will be saved to:
```
models/cats_dogs_model.keras
```

---

## 🎯 Usage

### ▶ Run the GUI Application
```bash
python src/gui_cats_dogs.py
```

**Keyboard Shortcuts:**
- `F11` – Toggle fullscreen
- `ESC` – Exit fullscreen

### ▶ Run Prediction Script (CLI)
```bash
python src/predict_cats_dogs.py -i path\to\image.jpg
```

**Note:** The CLI accepts `-i/--image`. If omitted, it will try `data/test_image.jpg`.

**Confidence Thresholds:** GUI = 0.55, CLI = 0.70 (adjust in source if needed)

---

## 📁 Project Structure
```
ImageClassifier/
├── src/
│   ├── gui_cats_dogs.py       # GUI application
│   ├── predict_cats_dogs.py   # CLI prediction script
│   └── train_cats_dogs.py     # Model training script
├── models/
│   └── cats_dogs_model.keras  # Trained model
├── data/                       # Training data (not included)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🛠️ Technical Details

- **Framework**: TensorFlow / Keras
- **Base Model**: MobileNetV2 (ImageNet weights)
- **Input Size**: 160×160 pixels
- **Classes**: Cat, Dog, Unknown
- **Confidence Thresholds:**
  - GUI: 0.55 (balanced for user interaction)
  - CLI: 0.70 (stricter for testing accuracy)

---

## 🤖 Development & Learning

**Disclaimer:** This project was created as a learning exercise to explore deep learning and Python development.

### 🛠️ AI Assistants & Tools Used
I utilized a modern, AI-assisted workflow to accelerate learning and development:

* **ChatGPT (OpenAI)** – Used for code generation, debugging, and conceptual understanding.
* **Google Antigravity** – Used as a  development resource for planning and logic.
* **GitHub Copilot** – Used for code generation.
* **Claude (Anthropic)** – Used for code refinement and documentation assistance.
---

**Important:** A significant portion of this codebase was generated or heavily assisted by AI tools.


---

## 📝 Notes

- The model file is included at `models/cats_dogs_model.keras`
- If the model is missing, the GUI shows an error dialog instead of crashing
- All paths are relative for portability across systems

---

## 🐛 Troubleshooting

**Model not found:**
- Ensure `models/cats_dogs_model.keras` exists
- Train a new model if necessary

**Training fails:**
- Verify `data` directory exists with `cats/` and `dogs/` subdirectories
- Ensure enough images (≥100 per class)

**GUI doesn't start:**
- Verify dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+)

**Low accuracy:**
- Increase dataset size
- Improve image quality
- Retrain the model

---

## 👤 Author

**Akalanka Senanayake**  
Undergraduate  
University of Kelaniya

---

## 📄 License

This project is open-source and intended for educational purposes.
