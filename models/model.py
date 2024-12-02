import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Reduce log messages from TensorFlow to critic errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # Force TensorFlow to use CPU rather than GPUs 

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D

# Base model congiguration
base_model = MobileNetV2(
    input_shape=(224,224,3),     # Image size (IMG_SIZE + 3 channels)
    include_top=False,           # Remove fully-connected layers
    weights='imagenet'           # Use pre-trained weights in ImageNet
)

# Freeze base model so to not alter pre-trained weights 
base_model.trainable = False

# Complete model adding final layers
model = Sequential([
    base_model,                     # Base Model
    GlobalAveragePooling2D(),       # Return data to a 2D vector
    Dense(128, activation='relu'),  # Fully-connected layer with 128 neurons
    Dropout(0.5),                   # Dropout random 50% neurons to avoid overfitting
    Dense(1, activation='sigmoid')  # Binary output (0=cat, 1=dog)
])

# Compile model
model.compile(
    optimizer='adam',               # Adam Optimizer
    loss='binary_crossentropy',     # Binary Loss
    metrics=['accuracy']            # Precision metric
)

# Summarizes the model
model.summary()