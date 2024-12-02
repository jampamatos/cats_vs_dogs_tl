"""
src/load_data.py

Handles dataset preparation for the cats vs. dogs classification task.
- Cleans invalid image files.
- Configures data generators for training and validation.
- Visualizes a sample of the dataset.
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Path to the dataset
DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"Directory {DATASET_DIR} not found.")

# --- Clean invalid images ---
invalid_files = []  # List to track removed invalid files

def clean_dataset(directory):
    """Removes invalid or corrupted image files from the dataset directory."""
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Validate the image
            except (IOError, SyntaxError):
                invalid_files.append(file_path)
                os.remove(file_path)
    print(f"{len(invalid_files)} invalid files removed.")

clean_dataset(DATASET_DIR)

# --- Hyperparameters ---
IMG_SIZE = (224, 224)  # Image dimensions
BATCH_SIZE = 32        # Batch size for training and validation

# --- Data Generators ---
datagen = ImageDataGenerator(
    rescale=1.0 / 255,      # Normalize pixel values to [0, 1]
    validation_split=0.2,   # Split 80% for training, 20% for validation
    rotation_range=20,      # Random rotation for augmentation
    horizontal_flip=True,   # Flip images horizontally
    zoom_range=0.2          # Random zoom for augmentation
)

# Training data generator
train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

# Validation data generator
val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=True
)

# --- Visualize Dataset ---
def plot_images(data_gen, num_images=9):
    """Displays a grid of sample images and their corresponding labels."""
    images, labels = next(data_gen)
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(f"Class: {int(labels[i])}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("output_images.png")
    print("Visualization saved as 'output_images.png'")

plot_images(train_data)
