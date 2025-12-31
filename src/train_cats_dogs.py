import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
train_dir = BASE_DIR / "data"
model_save_path = BASE_DIR / "models" / "cats_dogs_model.keras"

# -----------------------------
# Parameters
# -----------------------------
img_size = (160, 160)
batch_size = 32
epochs = 5  # increase for better accuracy

def train_model():
    """Train the cat vs dog classifier model."""
    # -----------------------------
    # Data generators
    # -----------------------------
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.3,
        horizontal_flip=True,
        rotation_range=20,
        zoom_range=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # -----------------------------
    # Load base model
    # -----------------------------
    base_model = MobileNetV2(input_shape=img_size + (3,), include_top=False, weights='imagenet')
    base_model.trainable = False  

    # -----------------------------
    # Add custom layers
    # -----------------------------
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    # -----------------------------
    # Train
    # -----------------------------
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs
    )

    # -----------------------------
    # Save model
    # -----------------------------
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved at {model_save_path}")
    return history

if __name__ == "__main__":
    if not train_dir.exists():
        print(f"Error: Training data directory not found at {train_dir}")
        print("Please create the 'data' folder with subdirectories for each class (e.g., data/cats/, data/dogs/)")
        exit(1)
    
    print(f"Starting training with data from: {train_dir}")
    train_model()
    print("Training complete!")
