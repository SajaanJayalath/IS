import json
import os
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_cnn(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    """Build a simple CNN for symbol classification."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train(data_dir: str = None,
          model_out: str = None,
          labels_out: str = None,
          img_size: Tuple[int, int] = (28, 28),
          batch_size: int = 64,
          epochs: int = 10,
          seed: int = 42) -> None:
    """
    Train the CNN on directory-structured dataset and save model + labels mapping.

    Directory layout (class folders):
      Data/0 .. Data/9, Data/+, Data/-, Data/times, Data/div, Data/(, Data/)
    """
    # Resolve default paths relative to this file for robustness
    here = os.path.dirname(os.path.abspath(__file__))
    if data_dir is None:
        data_dir = os.path.join(here, "Data")
    if model_out is None:
        model_out = os.path.join(here, "models", "math_symbol_model.h5")
    if labels_out is None:
        labels_out = os.path.join(here, "models", "labels.json")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=batch_size,
        subset="training",
        shuffle=True,
        seed=seed,
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=batch_size,
        subset="validation",
        shuffle=False,
        seed=seed,
    )

    num_classes = train_gen.num_classes
    input_shape = (img_size[0], img_size[1], 1)

    model = build_cnn(input_shape, num_classes)
    model.summary()

    # Ensure output directory exists before callbacks attempt to write
    os.makedirs(os.path.dirname(model_out), exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_out, save_best_only=True, monitor="val_accuracy", mode="max"),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy", mode="max"),
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)

    model.save(model_out)

    # Save labels mapping (index -> class label)
    # Keras assigns indices alphabetically by class name
    class_indices = train_gen.class_indices  # dict[label] = index
    idx_to_label = [None] * len(class_indices)
    for label, idx in class_indices.items():
        idx_to_label[idx] = label

    with open(labels_out, "w", encoding="utf-8") as f:
        json.dump({"idx_to_label": idx_to_label}, f, ensure_ascii=False, indent=2)

    print(f"Model saved to: {model_out}")
    print(f"Labels saved to: {labels_out}")


if __name__ == "__main__":
    # Basic CLI entry for quick training
    import argparse

    parser = argparse.ArgumentParser(description="Train math symbol CNN")
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--data", default=os.path.join(here, "Data"), help="Dataset root directory")
    parser.add_argument("--out", default=os.path.join(here, "models", "math_symbol_model.h5"), help="Output model path")
    parser.add_argument("--labels", default=os.path.join(here, "models", "labels.json"), help="Output labels JSON path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    train(data_dir=args.data, model_out=args.out, labels_out=args.labels, epochs=args.epochs, batch_size=args.batch)
