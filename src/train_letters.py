"""
Dedicated training pipeline for letter recognition in the Handwritten Number Recognition System (HNRS).

This script focuses on uppercase and lowercase letters from the NIST Special Database 19
``by_class`` archive and produces models and metadata compatible with the GUI letter mode.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from data_loader import NISTByClassLoader
from models import CNNModel, RandomForestModel


def _project_paths() -> Tuple[str, str]:
    """Return (project_root, models_dir)."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    models_dir = os.path.join(project_root, "models")
    return project_root, models_dir


def _ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
def _summarize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for key, value in metrics.items():
        if key in {"predictions", "probabilities"}:
            continue
        if isinstance(value, np.ndarray):
            summary[key] = value.tolist()
        else:
            summary[key] = value
    return summary


def train_cnn_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    epochs: int,
    batch_size: int,
    use_augmentation: bool,
    label_smoothing: float,
    output_path: str,
) -> Dict[str, Any]:
    print("\n=== Training Letter CNN ===")
    cnn = CNNModel(
        num_classes=len(class_names),
        class_names=class_names,
        use_augmentation=use_augmentation,
        label_smoothing=label_smoothing,
    )
    cnn.train(X_train, y_train, X_val=X_val, y_val=y_val, epochs=epochs, batch_size=batch_size)
    evaluation = cnn.evaluate(X_test, y_test)
    summary = _summarize_metrics(evaluation)
    summary["model_path"] = output_path
    cnn.save_model(output_path)
    print(f"Saved CNN model to {output_path}")
    return summary


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    estimators: int,
    max_depth: int | None,
    output_path: str,
) -> Dict[str, Any]:
    print("\n=== Training Letter Random Forest ===")
    rf = RandomForestModel(n_estimators=estimators, max_depth=max_depth)
    rf.train(X_train, y_train)
    evaluation = rf.evaluate(X_test, y_test)
    summary = _summarize_metrics(evaluation)
    summary["model_path"] = output_path
    rf.save_model(output_path)
    print(f"Saved Random Forest model to {output_path}")
    return summary


def build_metadata(
    dataset_name: str,
    loader: NISTByClassLoader,
    results: Dict[str, Dict[str, Any]],
    label_mapping: Dict[int, str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    timestamp = datetime.utcnow().isoformat() + "Z"
    return {
        "dataset": dataset_name,
        "generated_at": timestamp,
        "options": {
            "include_digits": args.include_digits,
            "include_uppercase": args.include_uppercase,
            "include_lowercase": args.include_lowercase,
            "max_per_class": args.max_per_class,
            "test_size": args.test_size,
            "cnn_epochs": args.epochs,
            "cnn_batch_size": args.batch_size,
            "cnn_use_augmentation": not args.no_augmentation,
            "cnn_label_smoothing": args.label_smoothing,
            "rf_estimators": args.rf_estimators,
            "rf_max_depth": args.rf_max_depth,
            "rf_max_samples": args.rf_max_samples,
        },
        "class_count": len(label_mapping),
        "class_names": [label_mapping[idx] for idx in sorted(label_mapping.keys())],
        "results": results,
        "dataset_summary": {
            "train_samples": int(loader.X_train.shape[0]) if loader.X_train is not None else None,
            "test_samples": int(loader.X_test.shape[0]) if loader.X_test is not None else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train letter recognition models for HNRS")
    parser.add_argument("--data-dir", type=str, default=os.path.join("data", "by_class", "by_class"),
                        help="Root directory of the NIST by_class dataset")
    parser.add_argument("--include-digits", action="store_true", default=False,
                        help="Include digit classes alongside letters")
    parser.set_defaults(include_uppercase=True, include_lowercase=True)
    parser.add_argument("--no-uppercase", dest="include_uppercase", action="store_false",
                        help="Exclude uppercase letters")
    parser.add_argument("--no-lowercase", dest="include_lowercase", action="store_false",
                        help="Exclude lowercase letters")
    parser.add_argument("--max-per-class", type=int, default=2000,
                        help="Optional limit on samples per class (None to disable)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Holdout fraction for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    parser.add_argument("--epochs", type=int, default=12, help="CNN training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="CNN batch size")
    parser.add_argument("--no-augmentation", action="store_true", help="Disable CNN data augmentation")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing factor for CNN")
    parser.add_argument("--skip-cnn", action="store_true", help="Skip CNN training")

    parser.add_argument("--train-rf", action="store_true", help="Train Random Forest model as well")
    parser.add_argument("--rf-estimators", type=int, default=200, help="Number of trees for Random Forest")
    parser.add_argument("--rf-max-depth", type=int, default=24, help="Maximum depth for Random Forest (0 = None)")
    parser.add_argument("--rf-max-samples", type=int, default=60000,
                        help="Cap Random Forest training samples for speed (0 = all)")

    args = parser.parse_args()

    if args.rf_max_depth == 0:
        args.rf_max_depth = None
    if args.rf_max_samples == 0:
        args.rf_max_samples = None

    project_root, models_dir = _project_paths()
    _ensure_dir(models_dir)

    loader = NISTByClassLoader(
        root_dir=args.data_dir,
        include_digits=args.include_digits,
        include_uppercase=args.include_uppercase,
        include_lowercase=args.include_lowercase,
        max_per_class=args.max_per_class,
        test_size=args.test_size,
        random_state=args.seed,
    )

    X_train_raw, y_train, X_test_raw, y_test = loader.load_data()
    label_mapping = loader.get_label_mapping()
    class_names = [label_mapping[idx] for idx in sorted(label_mapping.keys())]

    X_train_proc, _, X_test_proc, _ = loader.preprocess_data(normalize=True, reshape_for_cnn=True)

    print(f"Prepared dataset with {len(class_names)} classes")

    rng = np.random.default_rng(args.seed)

    results: Dict[str, Dict[str, Any]] = {}

    # CNN training
    if not args.skip_cnn:
        X_train_cnn, X_val_cnn, y_train_cnn, y_val_cnn = train_test_split(
            X_train_proc,
            y_train,
            test_size=0.1,
            random_state=args.seed,
            stratify=y_train,
        )
        cnn_path = os.path.join(models_dir, "cnn_model_nist_by_class.h5")
        results["cnn"] = train_cnn_model(
            X_train_cnn,
            y_train_cnn,
            X_val_cnn,
            y_val_cnn,
            X_test_proc,
            y_test,
            class_names,
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_augmentation=not args.no_augmentation,
            label_smoothing=args.label_smoothing,
            output_path=cnn_path,
        )
    else:
        print("Skipping CNN training")

    # Random Forest training (optional)
    if args.train_rf:
        rf_train = X_train_proc
        rf_labels = y_train
        if args.rf_max_samples and rf_train.shape[0] > args.rf_max_samples:
            indices = rng.choice(rf_train.shape[0], size=args.rf_max_samples, replace=False)
            rf_train = rf_train[indices]
            rf_labels = rf_labels[indices]
            print(f"Random Forest: sampled {rf_train.shape[0]} training examples")
        rf_path = os.path.join(models_dir, "rf_model_nist_by_class.pkl")
        results["random_forest"] = train_random_forest(
            rf_train,
            rf_labels,
            X_test_proc,
            y_test,
            estimators=args.rf_estimators,
            max_depth=args.rf_max_depth,
            output_path=rf_path,
        )
    else:
        print("Random Forest training disabled")

    # Persist metadata and label mapping
    metadata = build_metadata(
        dataset_name="nist_by_class_letters",
        loader=loader,
        results=results,
        label_mapping=label_mapping,
        args=args,
    )

    label_mapping_path = os.path.join(models_dir, "label_mapping_nist_by_class.json")
    metadata_path = os.path.join(models_dir, "metadata_nist_by_class_letters.json")

    _save_json(label_mapping_path, label_mapping)
    _save_json(metadata_path, metadata)

    print(f"Saved label mapping to {label_mapping_path}")
    print(f"Saved metadata to {metadata_path}")

    print("\nTraining complete. Available results:")
    for model_name, info in results.items():
        acc = info.get("accuracy")
        print(f"  {model_name}: accuracy={acc:.4f}" if acc is not None else f"  {model_name}: see metadata")


if __name__ == "__main__":
    main()