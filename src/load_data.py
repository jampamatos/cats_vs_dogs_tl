# Testing image load and visualization

import os
# Reduce TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image

# Path to dataset
DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

# Check if directory exist
if not os.path.exists(DATASET_DIR): raise FileNotFoundError(f"Directory {DATASET_DIR} not found.")

# Function to clean invalid images
invalid_files = []
def clean_dataset(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except (IOError, SyntaxError):
                invalid_files.append(file_path)
                os.remove(file_path)
    print(f"{len(invalid_files)} invalid files removed.")

clean_dataset(DATASET_DIR)

# Hyperparameters configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Basic augmentation data generator
datagen = ImageDataGenerator(
    rescale=1.0 / 255,      # Normalize pixels to [0, 1] interval
    validation_split=0.2,   # Split 80% training 20% validation
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2
)

# Train data
train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

# Validation data
val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=True
)

# Visualize some images to check if working
def plot_images(data_gen, num_images=9):
    images, labels = next(data_gen)
    plt.figure(figsize=(10,10))
    for i in range (num_images):
        plt.subplot(3,3, i+1)
        plt.imshow(images[i])
        plt.title(f"Class: {int(labels[i])}")
        plt.axis("off")
    plt.tight_layout()
    # Save the plot as an image instead of displaying
    plt.savefig("output_images.png")
    print("Visualization saved as 'output_images.png'")

plot_images(train_data)