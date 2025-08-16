"""
Training script for all ML models in the HNRS system
Trains CNN, SVM, and Random Forest models and compares their performance
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from data_loader import MNISTDataLoader
from models import CNNModel, SVMModel, RandomForestModel, ModelComparison

def create_models_directory():
    """Create directory to save trained models"""
    if not os.path.exists('models'):
        os.makedirs('models')
        print("Created 'models' directory for saving trained models")

def train_all_models():
    """Train and evaluate all models"""
    print("="*60)
    print("HANDWRITTEN NUMBER RECOGNITION SYSTEM - MODEL TRAINING")
    print("="*60)
    
    # Create models directory
    create_models_directory()
    
    # Load data
    print("\n1. Loading MNIST data...")
    data_loader = MNISTDataLoader(data_dir='../MNIST_CSV')
    
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
    
    # Train CNN with reduced epochs for faster training
    cnn_history = cnn_model.train(
        X_train_cnn, y_train_cnn, 
        X_val_cnn, y_val_cnn, 
        epochs=10,  # Reduced for faster training
        batch_size=128
    )
    
    cnn_train_time = time.time() - start_time
    print(f"CNN training completed in {cnn_train_time:.2f} seconds")
    
    # Evaluate CNN
    print("Evaluating CNN...")
    cnn_results = cnn_model.evaluate(X_test_processed, y_test)
    comparison.add_result("CNN", cnn_results)
    
    # Save CNN model
    cnn_model.save_model('models/cnn_model.h5')
    print("CNN model saved to 'models/cnn_model.h5'")
    
    # Train SVM Model
    print("\n" + "="*50)
    print("4. TRAINING SVM MODEL")
    print("="*50)
    
    start_time = time.time()
    svm_model = SVMModel(kernel='rbf', C=1.0, gamma='scale')
    
    # Use subset of data for SVM to speed up training
    subset_size = 10000  # Use 10k samples for faster training
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
    
    # Save SVM model
    svm_model.save_model('models/svm_model.pkl')
    print("SVM model saved to 'models/svm_model.pkl'")
    
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
    
    # Save Random Forest model
    rf_model.save_model('models/rf_model.pkl')
    print("Random Forest model saved to 'models/rf_model.pkl'")
    
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
    print("\nTrained models saved in 'models/' directory:")
    print("- cnn_model.h5 (CNN)")
    print("- svm_model.pkl (SVM)")
    print("- rf_model.pkl (Random Forest)")
    
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
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick-test":
        quick_test()
    else:
        try:
            comparison = train_all_models()
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print("Make sure MNIST CSV files are in the 'MNIST_CSV' directory")
            print("Run with '--quick-test' flag to test with dummy data")
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            print("Run with '--quick-test' flag to test with dummy data")
