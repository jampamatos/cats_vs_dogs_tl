"""
models/model.py

Defines a binary classification model for cats vs. dogs using transfer learning with MobileNetV2.
Custom layers are added for the specific classification task, leveraging pre-trained ImageNet weights.
"""

import os
# Suppress TensorFlow logs and force CPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

# Initialize pre-trained MobileNetV2 as the base model
base_model = MobileNetV2(
    input_shape=(224, 224, 3),  # Input: 224x224 RGB images
    include_top=False,          # Exclude original fully connected layers
    weights='imagenet'          # Use ImageNet weights
)
base_model.trainable = False  # Freeze base model weights

# Build the full model
model = Sequential([
    base_model,                     # Pre-trained base
    GlobalAveragePooling2D(),       # Converts feature maps to 2D vectors
    Dense(128, activation='relu'),  # Fully connected layer
    Dropout(0.5),                   # Dropout for regularization
    Dense(1, activation='sigmoid')  # Output: binary classification (0=cats, 1=dogs)
])

# Compile the model
model.compile(
    optimizer='adam',               # Optimizer for training
    loss='binary_crossentropy',     # Loss for binary classification
    metrics=['accuracy']            # Evaluation metric
)

# Print the model summary
model.summary()
