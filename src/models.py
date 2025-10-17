"""
Machine Learning Models for Handwritten Number Recognition System (HNRS)
Implements CNN, SVM, and Random Forest models for digit classification
"""

import numpy as np
# TensorFlow is optional for SVM/RF; guard its import to support environments
# where TensorFlow is unavailable (e.g., Python 3.13). The CNN model will raise
# a clear error when used without TensorFlow.
try:
    import tensorflow as tf  # type: ignore
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers  # type: ignore
    _TF_AVAILABLE = True
    _TF_IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover - environment dependent
    tf = None  # type: ignore
    keras = None  # type: ignore
    layers = None  # type: ignore
    _TF_AVAILABLE = False
    _TF_IMPORT_ERROR = _e
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os
from typing import Tuple, Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class CNNModel:
    """Convolutional Neural Network for digit recognition"""
    
    def __init__(self, input_shape=(28, 28, 1), num_classes=10,
                 use_augmentation: bool = True,
                 label_smoothing: float = 0.0,
                 class_names: Optional[List[str]] = None):
        self.input_shape = input_shape
        self.num_classes = int(num_classes)
        self.model = None
        self.history = None
        self.use_augmentation = use_augmentation
        self.label_smoothing = float(label_smoothing)
        self.class_names = class_names if class_names is not None else [str(i) for i in range(self.num_classes)]
        
    def _ensure_tf(self):
        """Ensure TensorFlow is available before using CNN features."""
        if not _TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is not available. The CNN model requires TensorFlow. "
                "Install a compatible version (e.g., tensorflow==2.15.* on Python 3.10/3.11) "
                f"or switch to SVM/Random Forest. Original import error: {_TF_IMPORT_ERROR}"
            )
        
    def build_model(self):
        """Build CNN architecture optimized for MNIST digit recognition"""
        self._ensure_tf()
        aug_layers = []
        if self.use_augmentation:
            # Lightweight on-the-fly augmentation applied only during training
            aug_layers = [
                layers.Input(shape=self.input_shape),
                layers.RandomRotation(0.08),
                layers.RandomTranslation(0.05, 0.05),
                layers.RandomZoom(0.10),
                layers.RandomContrast(0.10),
            ]

        L2 = keras.regularizers.l2(1e-4)

        self.model = keras.Sequential([
            *(aug_layers or [layers.Input(shape=self.input_shape)]),
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L2),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L2),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=L2),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=L2),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=L2),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu', kernel_regularizer=L2),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu', kernel_regularizer=L2),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        # Loss: prefer SparseCategoricalCrossentropy; add label_smoothing only if supported
        loss_obj: any
        try:
            import inspect  # type: ignore
            supports_ls = 'label_smoothing' in inspect.signature(
                keras.losses.SparseCategoricalCrossentropy.__init__  # type: ignore
            ).parameters
        except Exception:
            supports_ls = False

        if self.label_smoothing > 0 and supports_ls:
            loss_obj = keras.losses.SparseCategoricalCrossentropy(label_smoothing=self.label_smoothing)
        else:
            if self.label_smoothing > 0 and not supports_ls:
                print("Warning: label_smoothing not supported for SparseCategoricalCrossentropy in this Keras version; using standard loss.")
            loss_obj = 'sparse_categorical_crossentropy'

        self.model.compile(
            optimizer='adam',
            loss=loss_obj,
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=20, batch_size=128):
        """Train the CNN model"""
        self._ensure_tf()
        if self.model is None:
            self.build_model()
        
        # Callbacks for better training
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]
        
        # Train model
        validation_data = (X_val, y_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        self._ensure_tf()
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def save_model(self, filepath):
        """Save trained model"""
        self._ensure_tf()
        if self.model is None:
            raise ValueError("No model to save!")
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """Load trained model"""
        self._ensure_tf()
        self.model = keras.models.load_model(filepath)
    
    def predict(self, X):
        """Make predictions on input data"""
        self._ensure_tf()
        if self.model is None:
            raise ValueError("Model not trained or loaded yet!")

        X = np.asarray(X)
        if X.ndim == 2:
            X = X.reshape(1, X.shape[0], X.shape[1], 1)
        elif X.ndim == 3:
            if X.shape[-1] == 1:
                X = X.reshape(1, X.shape[0], X.shape[1], 1)
            else:
                X = X.reshape(1, X.shape[0], X.shape[1], X.shape[2])
        elif X.ndim == 4:
            if X.shape[0] != 1:
                # keep as is for batch predictions
                pass
        else:
            raise ValueError(f"Unexpected input shape for CNN predict: {X.shape}")

        X = X.astype(np.float32)
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1), predictions
    
    def plot_training_history(self):
        """Plot training history"""
        self._ensure_tf()
        if self.history is None:
            print("No training history available!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in self.history.history:
            ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

class SVMModel:
    """Support Vector Machine for digit recognition"""
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = None
        
    def train(self, X_train, y_train):
        """Train SVM model"""
        # Flatten images for SVM
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        # Initialize and train SVM
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=True,  # Enable probability estimates
            random_state=42
        )
        
        print(f"Training SVM with {X_train_flat.shape[0]} samples...")
        self.model.fit(X_train_flat, y_train)
        print("SVM training completed!")
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Evaluate SVM performance"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Flatten test images
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Get predictions
        y_pred = self.model.predict(X_test_flat)
        y_pred_proba = self.model.predict_proba(X_test_flat)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def save_model(self, filepath):
        """Save trained SVM model"""
        if self.model is None:
            raise ValueError("No model to save!")
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, filepath):
        """Load trained SVM model"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
    
    def predict(self, X):
        """Make predictions on input data"""
        if self.model is None:
            raise ValueError("Model not trained or loaded yet!")

        X = np.array(X)
        if X.ndim == 4:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X.reshape(1, -1)

        predictions = self.model.predict(X_flat)
        probabilities = self.model.predict_proba(X_flat)
        return predictions, probabilities

class RandomForestModel:
    """Random Forest for digit recognition"""
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        
    def train(self, X_train, y_train):
        """Train Random Forest model"""
        # Flatten images for Random Forest
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        
        # Initialize and train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1  # Use all available cores
        )
        
        print(f"Training Random Forest with {X_train_flat.shape[0]} samples...")
        self.model.fit(X_train_flat, y_train)
        print("Random Forest training completed!")
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Evaluate Random Forest performance"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Flatten test images
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Get predictions
        y_pred = self.model.predict(X_test_flat)
        y_pred_proba = self.model.predict_proba(X_test_flat)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def save_model(self, filepath):
        """Save trained Random Forest model"""
        if self.model is None:
            raise ValueError("No model to save!")
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, filepath):
        """Load trained Random Forest model"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
    
    def predict(self, X):
        """Make predictions on input data"""
        if self.model is None:
            raise ValueError("Model not trained or loaded yet!")

        X = np.array(X)
        if X.ndim == 4:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X.reshape(1, -1)

        predictions = self.model.predict(X_flat)
        probabilities = self.model.predict_proba(X_flat)
        return predictions, probabilities

class ModelComparison:
    """Compare performance of different models"""
    
    def __init__(self):
        self.results = {}
    
    def add_result(self, model_name, evaluation_result):
        """Add evaluation result for a model"""
        self.results[model_name] = evaluation_result
    
    def compare_accuracies(self):
        """Compare accuracies of all models"""
        if not self.results:
            print("No results to compare!")
            return
        
        print("\n" + "="*50)
        print("MODEL ACCURACY COMPARISON")
        print("="*50)
        
        for model_name, result in self.results.items():
            print(f"{model_name:15}: {result['accuracy']:.4f}")
        
        # Find best model
        best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest Model: {best_model[0]} ({best_model[1]['accuracy']:.4f})")
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        if not self.results:
            print("No results to plot!")
            return
        
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, result) in enumerate(self.results.items()):
            sns.heatmap(
                result['confusion_matrix'], 
                annot=True, 
                fmt='d', 
                ax=axes[idx],
                cmap='Blues'
            )
            axes[idx].set_title(f'{model_name}\nAccuracy: {result["accuracy"]:.4f}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()

# Test function (skips CNN if TF unavailable)
if __name__ == "__main__":
    print("Testing model implementations...")

    # Create dummy data for testing
    X_dummy = np.random.random((100, 28, 28, 1))
    y_dummy = np.random.randint(0, 10, 100)

    # Test CNN (only if TF is available)
    print("\nTesting CNN...")
    try:
        cnn = CNNModel()
        cnn.build_model()
        print("CNN model built successfully!")
    except Exception as e:
        print(f"Skipping CNN test: {e}")

    # Test SVM
    print("\nTesting SVM...")
    svm = SVMModel()
    print("SVM model initialized successfully!")

    # Test Random Forest
    print("\nTesting Random Forest...")
    rf = RandomForestModel()
    print("Random Forest model initialized successfully!")

    print("\nAll model classes imported successfully!")
