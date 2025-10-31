# pyright: reportMissingImports=false
import os
import sys
from typing import Tuple
import numpy as np

# Prefer TensorFlow-Keras, but fall back to standalone Keras if available
try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
    from tensorflow.keras.models import load_model  # type: ignore
except Exception:
    from keras.preprocessing.image import ImageDataGenerator  # type: ignore
    from keras.models import load_model  # type: ignore


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "Data")
    model_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "models", "math_symbol_model.h5")

    if not os.path.isfile(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(2)

    datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(28, 28),
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=512,
        subset="validation",
        shuffle=False,
    )

    model = load_model(model_path)
    # Single pass over validation set for predictions
    preds = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes

    overall_acc = float(np.mean(y_pred == y_true))
    print(f"VALIDATION_ACCURACY={overall_acc:.4f}")

    # Map indices to class labels
    class_indices = val_gen.class_indices  # dict[label] = idx
    idx_to_label = [None] * len(class_indices)
    for label, idx in class_indices.items():
        idx_to_label[idx] = label

    # Per-class accuracy
    totals = np.bincount(y_true, minlength=len(idx_to_label))
    correct = np.bincount(y_true[y_true == y_pred], minlength=len(idx_to_label))

    print("PER_CLASS_ACCURACY:")
    for idx, label in enumerate(idx_to_label):
        total = int(totals[idx])
        corr = int(correct[idx])
        acc = (corr / total) if total > 0 else 0.0
        print(f"- {label}: {acc:.4f} ({corr}/{total})")


if __name__ == "__main__":
    main()
