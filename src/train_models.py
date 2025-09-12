"""
Training script for all ML models in the HNRS system
Trains CNN, SVM, and Random Forest models and compares their performance
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse
import json
from datetime import datetime
from data_loader import MNISTDataLoader, get_data_loader
from models import CNNModel, SVMModel, RandomForestModel, ModelComparison

def _project_paths():
    """Compute project-rooted paths independent of the current working dir."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    models_dir = os.path.join(project_root, 'models')
    data_dir = os.path.join(project_root, 'MNIST_CSV')
    return project_root, models_dir, data_dir

def create_models_directory(models_dir: str):
    """Create directory to save trained models"""
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print("Created 'models' directory for saving trained models")

def train_all_models(dataset: str = 'mnist_csv', data_dir: str | None = None, epochs: int = 10, subset_size: int = 10000):
    """Train and evaluate all models

    Args:
        dataset: 'mnist_csv' or 'image_folder'
        data_dir: path to dataset root (required for image_folder)
        epochs: CNN training epochs
        subset_size: number of samples for SVM/RF speed-up
    """
    print("="*60)
    print("HANDWRITTEN NUMBER RECOGNITION SYSTEM - MODEL TRAINING")
    print("="*60)
    
    # Resolve paths and create models directory
    project_root, models_dir, data_dir = _project_paths()
    create_models_directory(models_dir)
    
    # Load data
    print("\n1. Loading data...")
    if dataset == 'mnist_csv':
        # project-rooted MNIST_CSV directory by default
        _project_root, _models_dir, default_data_dir = _project_paths()
        use_dir = data_dir or default_data_dir
    else:
        use_dir = data_dir  # image_folder requires explicit path

    data_loader = get_data_loader(dataset, use_dir)
    
    # Load training and test data
    X_train, y_train, X_test, y_test = data_loader.load_data()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    X_train_processed, y_train, X_test_processed, y_test = data_loader.preprocess_data(
        normalize=True, reshape_for_cnn=True
    )
    
    # Create validation split for CNN from the processed data
    from sklearn.model_selection import train_test_split
    X_train_cnn, X_val_cnn, y_train_cnn, y_val_cnn = train_test_split(
        X_train_processed, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    print(f"CNN Training data: {X_train_cnn.shape}")
    print(f"CNN Validation data: {X_val_cnn.shape}")
    
    # For SVM and Random Forest, use flattened data (no validation split needed)
    X_train_flat = X_train_processed.reshape(X_train_processed.shape[0], -1)
    X_test_flat = X_test_processed.reshape(X_test_processed.shape[0], -1)
    
    # Initialize model comparison
    comparison = ModelComparison()
    
    # Train CNN Model
    print("\n" + "="*50)
    print("3. TRAINING CNN MODEL")
    print("="*50)
    
    start_time = time.time()
    cnn_model = CNNModel()
    
    # Train CNN
    cnn_history = cnn_model.train(
        X_train_cnn, y_train_cnn, 
        X_val_cnn, y_val_cnn, 
        epochs=epochs,
        batch_size=128
    )
    
    cnn_train_time = time.time() - start_time
    print(f"CNN training completed in {cnn_train_time:.2f} seconds")
    
    # Evaluate CNN
    print("Evaluating CNN...")
    cnn_results = cnn_model.evaluate(X_test_processed, y_test)
    comparison.add_result("CNN", cnn_results)
    
    # Save CNN model (generic + dataset-specific filenames)
    cnn_generic_path = os.path.join(models_dir, 'cnn_model.h5')
    cnn_dataset_path = os.path.join(models_dir, f"cnn_model_{dataset}.h5")
    cnn_model.save_model(cnn_generic_path)
    try:
        cnn_model.save_model(cnn_dataset_path)
    except Exception:
        pass
    print(f"CNN model saved to '{cnn_generic_path}'")
    print(f"CNN model (dataset) saved to '{cnn_dataset_path}'")
    
    # Train SVM Model
    print("\n" + "="*50)
    print("4. TRAINING SVM MODEL")
    print("="*50)
    
    start_time = time.time()
    svm_model = SVMModel(kernel='rbf', C=1.0, gamma='scale')
    
    # Use subset of data for SVM to speed up training
    subset_size = min(subset_size, len(X_train_flat))
    indices = np.random.choice(len(X_train_flat), subset_size, replace=False)
    X_train_svm = X_train_flat[indices]
    y_train_svm = y_train[indices]
    
    print(f"Training SVM with {subset_size} samples for faster training...")
    svm_model.train(X_train_svm, y_train_svm)
    
    svm_train_time = time.time() - start_time
    print(f"SVM training completed in {svm_train_time:.2f} seconds")
    
    # Evaluate SVM
    print("Evaluating SVM...")
    svm_results = svm_model.evaluate(X_test_processed, y_test)
    comparison.add_result("SVM", svm_results)
    
    # Save SVM model (generic + dataset-specific filenames)
    svm_generic_path = os.path.join(models_dir, 'svm_model.pkl')
    svm_dataset_path = os.path.join(models_dir, f"svm_model_{dataset}.pkl")
    svm_model.save_model(svm_generic_path)
    try:
        svm_model.save_model(svm_dataset_path)
    except Exception:
        pass
    print(f"SVM model saved to '{svm_generic_path}'")
    print(f"SVM model (dataset) saved to '{svm_dataset_path}'")
    
    # Train Random Forest Model
    print("\n" + "="*50)
    print("5. TRAINING RANDOM FOREST MODEL")
    print("="*50)
    
    start_time = time.time()
    rf_model = RandomForestModel(n_estimators=100, max_depth=20, random_state=42)
    
    # Use subset of data for Random Forest to speed up training
    X_train_rf = X_train_flat[indices]  # Use same subset as SVM
    y_train_rf = y_train[indices]
    
    print(f"Training Random Forest with {subset_size} samples for faster training...")
    rf_model.train(X_train_rf, y_train_rf)
    
    rf_train_time = time.time() - start_time
    print(f"Random Forest training completed in {rf_train_time:.2f} seconds")
    
    # Evaluate Random Forest
    print("Evaluating Random Forest...")
    rf_results = rf_model.evaluate(X_test_processed, y_test)
    comparison.add_result("Random Forest", rf_results)
    
    # Save Random Forest model (generic + dataset-specific filenames)
    rf_generic_path = os.path.join(models_dir, 'rf_model.pkl')
    rf_dataset_path = os.path.join(models_dir, f"rf_model_{dataset}.pkl")
    rf_model.save_model(rf_generic_path)
    try:
        rf_model.save_model(rf_dataset_path)
    except Exception:
        pass
    print(f"Random Forest model saved to '{rf_generic_path}'")
    print(f"Random Forest model (dataset) saved to '{rf_dataset_path}'")
    
    # Compare all models
    print("\n" + "="*60)
    print("6. MODEL COMPARISON RESULTS")
    print("="*60)
    
    # Print training times
    print("\nTraining Times:")
    print(f"CNN:           {cnn_train_time:.2f} seconds")
    print(f"SVM:           {svm_train_time:.2f} seconds")
    print(f"Random Forest: {rf_train_time:.2f} seconds")
    
    # Compare accuracies
    comparison.compare_accuracies()
    
    # Print detailed classification reports
    print("\n" + "="*50)
    print("DETAILED CLASSIFICATION REPORTS")
    print("="*50)
    
    for model_name, results in comparison.results.items():
        print(f"\n{model_name} Classification Report:")
        print("-" * 40)
        print(results['classification_report'])
    
    # Plot confusion matrices
    print("\nGenerating confusion matrix plots...")
    comparison.plot_confusion_matrices()
    
    # Plot CNN training history
    if hasattr(cnn_model, 'history') and cnn_model.history is not None:
        print("\nGenerating CNN training history plot...")
        cnn_model.plot_training_history()

    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nTrained models saved in directory:")
    print(models_dir)
    print("- cnn_model.h5 (CNN)")
    print("- svm_model.pkl (SVM)")
    print("- rf_model.pkl (Random Forest)")
    
    # Save training metadata for traceability
    try:
        def _to_serializable(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            return x

        metadata = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": dataset,
            "data_dir": os.path.abspath(use_dir) if use_dir else None,
            "epochs": int(epochs),
            "subset_size": int(subset_size),
            "train_shapes": {
                "X_train": list(X_train.shape),
                "X_test": list(X_test.shape),
            },
            "times_seconds": {
                "cnn_train": float(cnn_train_time),
                "svm_train": float(svm_train_time),
                "rf_train": float(rf_train_time),
            },
            "metrics": {
                name: {
                    "accuracy": float(res.get("accuracy", 0.0)),
                    "classification_report": res.get("classification_report", ""),
                    "confusion_matrix": _to_serializable(res.get("confusion_matrix")),
                }
                for name, res in comparison.results.items()
            }
        }
        # Write generic metadata and dataset-specific metadata files
        meta = os.path.join(models_dir, "metadata.json")
        meta_ds = os.path.join(models_dir, f"metadata_{dataset}.json")
        with open(meta, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        try:
            with open(meta_ds, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        print(f"\nSaved training metadata to: {meta}")
        print(f"Saved dataset-specific metadata to: {meta_ds}")
    except Exception as e:
        print(f"\nWarning: failed to write training metadata: {e}")

    return comparison

def quick_test():
    """Quick test with small dataset"""
    print("Running quick test with small dataset...")
    
    # Create small dummy dataset
    X_train = np.random.random((1000, 28, 28, 1))
    y_train = np.random.randint(0, 10, 1000)
    X_test = np.random.random((200, 28, 28, 1))
    y_test = np.random.randint(0, 10, 200)
    
    # Test CNN
    print("\nTesting CNN...")
    cnn = CNNModel()
    cnn.train(X_train, y_train, epochs=2, batch_size=32)
    cnn_results = cnn.evaluate(X_test, y_test)
    print(f"CNN Test Accuracy: {cnn_results['accuracy']:.4f}")
    
    # Test SVM
    print("\nTesting SVM...")
    svm = SVMModel()
    svm.train(X_train, y_train)
    svm_results = svm.evaluate(X_test, y_test)
    print(f"SVM Test Accuracy: {svm_results['accuracy']:.4f}")
    
    # Test Random Forest
    print("\nTesting Random Forest...")
    rf = RandomForestModel(n_estimators=10)  # Reduced for speed
    rf.train(X_train, y_train)
    rf_results = rf.evaluate(X_test, y_test)
    print(f"Random Forest Test Accuracy: {rf_results['accuracy']:.4f}")
    
    print("\nQuick test completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HNRS models")
    parser.add_argument("--quick-test", action="store_true", help="Run a quick dummy-data test")
    parser.add_argument("--dataset", choices=["mnist_csv", "image_folder", "emnist_digits", "svhn", "combined"], default="mnist_csv",
                        help="Dataset source")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (required for image_folder)")
    parser.add_argument("--epochs", type=int, default=10, help="CNN training epochs")
    parser.add_argument("--subset-size", type=int, default=10000,
                        help="Samples for SVM/RF speed-up")

    args = parser.parse_args()

    if args.quick_test:
        quick_test()
    else:
        try:
            comparison = train_all_models(dataset=args.dataset,
                                          data_dir=args.data_dir,
                                          epochs=args.epochs,
                                          subset_size=args.subset_size)
        except FileNotFoundError as e:
            _, _, default_csv_dir = _project_paths()
            print(f"\nError: {e}")
            if args.dataset == 'mnist_csv':
                print(f"Expected MNIST CSV files in: {default_csv_dir}")
            else:
                print("For image_folder, ensure structure: <root>/train/0..9[/images], optional <root>/test/0..9")
            print("Run with '--quick-test' flag to test with dummy data")
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            print("Run with '--quick-test' flag to test with dummy data")
