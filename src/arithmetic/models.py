"""
Models for arithmetic expression recognition.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from .constants import CANONICAL_SYMBOLS

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    _TF_AVAILABLE = True
    _TF_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    tf = None  # type: ignore
    keras = None  # type: ignore
    layers = None  # type: ignore
    _TF_AVAILABLE = False
    _TF_IMPORT_ERROR = exc


def _ensure_tf() -> None:
    if not _TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is required for the arithmetic CNN but is not available. "
            f"Original import error: {_TF_IMPORT_ERROR}"
        )


@dataclass
class PredictionOutput:
    labels: np.ndarray
    probabilities: np.ndarray


class ArithmeticCNNModel:
    """High-accuracy CNN tailored to arithmetic symbol dataset."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (28, 28, 1),
        num_classes: int = len(CANONICAL_SYMBOLS),
        base_filters: int = 48,
        dropout_rate: float = 0.35,
    ) -> None:
        _ensure_tf()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.base_filters = base_filters
        self.dropout_rate = dropout_rate
        self.model: keras.Model | None = None
        self._build()

    def _build(self) -> None:
        data_augmentation = keras.Sequential(
            [
                layers.Input(shape=self.input_shape),
                layers.RandomRotation(0.20),
                layers.RandomTranslation(0.15, 0.15),
                layers.RandomZoom(0.20),
                layers.RandomContrast(0.20),
            ],
            name="augmentation",
        )

        L2 = keras.regularizers.l2(1e-4)
        inputs = keras.Input(shape=self.input_shape)
        x = data_augmentation(inputs)

        filters = self.base_filters
        for block in range(3):
            x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False, kernel_regularizer=L2)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False, kernel_regularizer=L2)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(self.dropout_rate)(x)
            filters *= 2

        x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False, kernel_regularizer=L2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)
        self.model = keras.Model(inputs, outputs, name="arithmetic_cnn")
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 40,
        batch_size: int = 128,
    ) -> "keras.callbacks.History":
        callbacks: list[keras.callbacks.Callback] = [
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_accuracy",
                factor=0.5,
                patience=4,
                min_delta=1e-3,
                verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=8,
                restore_best_weights=True,
                verbose=1,
            ),
        ]

        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        return self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=2,
            callbacks=callbacks,
        )

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        results = self.model.evaluate(X, y, verbose=0)
        metrics = dict(zip(self.model.metrics_names, results))
        return metrics

    def predict(self, X: np.ndarray) -> PredictionOutput:
        probs = self.model.predict(X, verbose=0)
        labels = probs.argmax(axis=1)
        return PredictionOutput(labels=labels, probabilities=probs)

    def predict_single(self, sample: np.ndarray) -> Tuple[int, np.ndarray]:
        batch = sample.reshape(1, *self.input_shape)
        output = self.predict(batch)
        return int(output.labels[0]), output.probabilities[0]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)

    def load(self, path: str) -> None:
        _ensure_tf()
        self.model = keras.models.load_model(path)


class ArithmeticEnsemble:
    """
    Lightweight ensemble that averages multiple stochastic forward passes to
    boost confidence estimates.
    """

    def __init__(self, backbone: ArithmeticCNNModel, passes: int = 5) -> None:
        self.backbone = backbone
        self.passes = passes

    def predict(self, X: np.ndarray) -> PredictionOutput:
        probs = []
        for _ in range(self.passes):
            probs.append(self.backbone.model(X, training=True).numpy())  # type: ignore
        stacked = np.stack(probs, axis=0)
        mean = stacked.mean(axis=0)
        labels = mean.argmax(axis=1)
        return PredictionOutput(labels=labels, probabilities=mean)

    def predict_single(self, sample: np.ndarray) -> Tuple[int, np.ndarray]:
        batch = sample.reshape(1, *self.backbone.input_shape)
        output = self.predict(batch)
        return int(output.labels[0]), output.probabilities[0]
