import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import tensorflow as tf
from models.model import model
from src.load_data import train_data, val_data

# Hyperparameters configurations
EPOCHS = 10
STEPS_PER_EPOCH = train_data.samples // train_data.batch_size
VALIDATION_STEPS = val_data.samples // val_data.batch_size

# Callback for saving the model during training
checkpoint_path = 'models/saved_model.keras'
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Model training
history = model.fit(
    train_data,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_data,
    validation_steps=VALIDATION_STEPS,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Saves the final model
save_file_path = 'models/final_model.keras'
model.save(save_file_path)
print(f"Final model saved in {save_file_path}")

accuracy_plot_path = 'img/accuracy_plot.png'
loss_plot_path = 'img/loss_plot.png'
os.makedirs('img', exist_ok=True)

# Generates accuracy graphs
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid()
plt.savefig(accuracy_plot_path)
plt.show()

# Generating loss graphs
plt.figure(figsize=(10,5))
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