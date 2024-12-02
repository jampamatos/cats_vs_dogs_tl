"""
src/train_model.py

Trains the binary classification model for cats vs. dogs.
- Loads the preprocessed dataset and model.
- Configures callbacks for saving the best model.
- Plots and saves accuracy and loss graphs.
"""

import os
import sys

# Add the project root directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import tensorflow as tf
from models.model import model           # Import the defined model
from src.load_data import train_data, val_data  # Import training and validation data generators

# --- Hyperparameters ---
EPOCHS = 10  # Number of training epochs
STEPS_PER_EPOCH = train_data.samples // train_data.batch_size  # Steps per epoch for training
VALIDATION_STEPS = val_data.samples // val_data.batch_size     # Steps per epoch for validation

# --- Callbacks ---
checkpoint_path = 'models/saved_model.keras'  # Path to save the best model during training
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',  # Monitor validation accuracy to save the best model
        save_best_only=True,     # Save only when validation accuracy improves
        verbose=1
    )
]

# --- Model Training ---
history = model.fit(
    train_data,                # Training data generator
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_data,  # Validation data generator
    validation_steps=VALIDATION_STEPS,
    epochs=EPOCHS,
    callbacks=callbacks,       # Save the best model during training
    verbose=1                  # Print detailed training logs
)

# --- Save Final Model ---
save_file_path = 'models/final_model.keras'
model.save(save_file_path)
print(f"Final model saved in {save_file_path}")

# --- Generate and Save Performance Graphs ---
# Create the directory for saving plots
os.makedirs('img', exist_ok=True)

# Accuracy Plot
accuracy_plot_path = 'img/accuracy_plot.png'
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid()
plt.savefig(accuracy_plot_path)
plt.show()

# Loss Plot
loss_plot_path = 'img/loss_plot.png'
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss over Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.savefig(loss_plot_path)
plt.show()

print(f"Plots saved as {accuracy_plot_path} and {loss_plot_path}.")
