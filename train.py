"""
train.py - Training pipeline for MNIST digit recognition CNN.
Features:
  - Loads and normalizes MNIST dataset
  - Image data augmentation (rotation, shifting, zoom)
  - CNN training with Adam optimizer, ReduceLROnPlateau, and ModelCheckpoint
  - Target test accuracy: >= 99%
  - Evaluates test set and outputs accuracy & classification report
  - Saves training curves (training_curves.png) and confusion matrix (confusion_matrix.png)
  - Saves the trained model as model.keras and model.h5
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import keras
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.datasets import mnist
from keras.utils import to_categorical

from model import build_mnist_cnn, compile_model


def load_and_preprocess_data():
    """
    Loads MNIST dataset, reshapes, and normalizes pixel values to [0, 1].
    Encodes labels as one-hot vectors.
    """
    print("[1/5] Loading and preprocessing MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape to (num_samples, 28, 28, 1) and normalize to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # Ground-truth integers for metrics
    y_test_labels = y_test.copy()

    # One-hot encode targets
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    print(f"      Train samples: {x_train.shape[0]} | Shape: {x_train.shape}")
    print(f"      Test samples:  {x_test.shape[0]} | Shape: {x_test.shape}")
    return x_train, y_train, x_test, y_test, y_test_labels


def build_data_augmenter():
    """
    Constructs a Keras Sequential layer for data augmentation (rotation, translation, zoom).
    """
    augmenter = keras.Sequential(
        [
            layers.RandomRotation(0.08, fill_mode="constant", fill_value=0.0),
            layers.RandomTranslation(0.08, 0.08, fill_mode="constant", fill_value=0.0),
            layers.RandomZoom(0.08, fill_mode="constant", fill_value=0.0),
        ],
        name="data_augmentation",
    )
    return augmenter


def plot_and_save_curves(history, output_path: str = "training_curves.png"):
    """
    Plots training and validation loss and accuracy curves over epochs.
    """
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Accuracy Plot
    ax1.plot(epochs_range, acc, "o-", label="Train Accuracy", color="#3b82f6", linewidth=2)
    ax1.plot(epochs_range, val_acc, "s--", label="Val Accuracy", color="#10b981", linewidth=2)
    ax1.set_title("Training & Validation Accuracy", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.set_ylim([0.95, 1.0])

    # Loss Plot
    ax2.plot(epochs_range, loss, "o-", label="Train Loss", color="#ef4444", linewidth=2)
    ax2.plot(epochs_range, val_loss, "s--", label="Val Loss", color="#f59e0b", linewidth=2)
    ax2.set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[+] Saved training curves to: {output_path}")


def plot_and_save_confusion_matrix(y_true, y_pred, output_path: str = "confusion_matrix.png"):
    """
    Calculates and plots the confusion matrix heatmap with Seaborn.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7), dpi=150)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=[str(i) for i in range(10)],
        yticklabels=[str(i) for i in range(10)],
        square=True,
    )
    plt.title("MNIST Test Set Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Digit", fontsize=12, labelpad=10)
    plt.ylabel("True Digit", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[+] Saved confusion matrix to: {output_path}")


def main():
    # 1. Dataset
    x_train, y_train, x_test, y_test, y_test_labels = load_and_preprocess_data()

    # Split small portion of train for validation (e.g. 5,000 samples)
    val_samples = 5000
    x_val = x_train[:val_samples]
    y_val = y_train[:val_samples]
    x_train_split = x_train[val_samples:]
    y_train_split = y_train[val_samples:]

    batch_size = 128
    epochs = 12

    # 2. Data Augmentation Dataset
    augmenter = build_data_augmenter()
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train_split, y_train_split))
        .shuffle(buffer_size=10000)
        .batch(batch_size)
        .map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # 3. Model
    print("[2/5] Building and compiling CNN architecture...")
    model = build_mnist_cnn(input_shape=(28, 28, 1), num_classes=10)
    compile_model(model, learning_rate=1e-3)
    model.summary()

    # Callbacks
    checkpoint_filepath = "model.keras"
    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # 4. Training
    print(f"[3/5] Starting training for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # 5. Evaluation
    print("[4/5] Evaluating model on the 10,000-sample test set...")
    # Load best checkpoint
    if os.path.exists(checkpoint_filepath):
        print(f"      Loading best model checkpoint from {checkpoint_filepath}...")
        model = keras.models.load_model(checkpoint_filepath)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print("\n" + "=" * 50)
    print(f"  >>> FINAL TEST ACCURACY: {test_acc * 100:.2f}% <<<")
    print(f"  >>> FINAL TEST LOSS:     {test_loss:.4f} <<<")
    print("=" * 50 + "\n")

    # Predictions for confusion matrix and report
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    print("Classification Report:")
    print(classification_report(y_test_labels, y_pred_classes, digits=4))

    # Also save as legacy model.h5 if supported
    try:
        model.save("model.h5")
        print("[+] Also saved model as model.h5")
    except Exception as e:
        print(f"Note: model.h5 export skipped ({e}). model.keras is primary format.")

    # 6. Plots
    print("[5/5] Generating and saving evaluation plots...")
    plot_and_save_curves(history, output_path="training_curves.png")
    plot_and_save_confusion_matrix(y_test_labels, y_pred_classes, output_path="confusion_matrix.png")

    print("\n[SUCCESS] Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
