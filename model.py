"""
model.py - Convolutional Neural Network architecture for MNIST digit recognition.
"""

from typing import Tuple
import keras
from keras import layers, models, optimizers


def build_mnist_cnn(input_shape: Tuple[int, int, int] = (28, 28, 1), num_classes: int = 10) -> keras.Model:
    """
    Builds a high-accuracy Convolutional Neural Network (CNN) for MNIST classification.

    Architecture:
      - Block 1: 2x Conv2D (32 filters, 3x3), BatchNormalization, MaxPooling2D (2x2), Dropout (0.25)
      - Block 2: 2x Conv2D (64 filters, 3x3), BatchNormalization, MaxPooling2D (2x2), Dropout (0.25)
      - Classifier: Flatten, Dense (256, relu), BatchNormalization, Dropout (0.4), Dense (10, softmax)

    Args:
        input_shape: Shape of the input images, default (28, 28, 1).
        num_classes: Number of output classes (0-9 digits), default 10.

    Returns:
        Compiled or uncompiled tf.keras.Model instance.
    """
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),

            # Block 1
            layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            # Block 2
            layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),

            # Dense Classifier
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="mnist_cnn_classifier",
    )

    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Compiles the Keras model using Adam optimizer and categorical crossentropy loss.

    Args:
        model: tf.keras.Model instance.
        learning_rate: Learning rate for Adam optimizer.

    Returns:
        The compiled model.
    """
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    cnn = build_mnist_cnn()
    compile_model(cnn)
    cnn.summary()
