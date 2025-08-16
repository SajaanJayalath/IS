"""
Data loading and preprocessing module for HNRS
Handles MNIST CSV data loading, preprocessing, and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

class MNISTDataLoader:
    """
    Handles loading and preprocessing of MNIST data from CSV files
    """
    
    def __init__(self, data_dir="MNIST_CSV"):
        """
        Initialize the data loader
        
        Args:
            data_dir (str): Directory containing MNIST CSV files
        """
        self.data_dir = data_dir
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """
        Load MNIST data from CSV files
        
        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        print("Loading MNIST data from CSV files...")
        
        # Load training data
        train_path = os.path.join(self.data_dir, "mnist_train.csv")
        test_path = os.path.join(self.data_dir, "mnist_test.csv")
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"MNIST CSV files not found in {self.data_dir}")
        
        # Load data
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)
        
        print(f"Training data shape: {self.train_data.shape}")
        print(f"Test data shape: {self.test_data.shape}")
        
        # Separate features and labels
        self.X_train = self.train_data.iloc[:, 1:].values  # All columns except first (label)
        self.y_train = self.train_data.iloc[:, 0].values   # First column (label)
        
        self.X_test = self.test_data.iloc[:, 1:].values
        self.y_test = self.test_data.iloc[:, 0].values
        
        print(f"X_train shape: {self.X_train.shape}")
        print(f"y_train shape: {self.y_train.shape}")
        print(f"X_test shape: {self.X_test.shape}")
        print(f"y_test shape: {self.y_test.shape}")
        
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def preprocess_data(self, normalize=True, reshape_for_cnn=False):
        """
        Preprocess the loaded data
        
        Args:
            normalize (bool): Whether to normalize pixel values to [0,1]
            reshape_for_cnn (bool): Whether to reshape for CNN (28x28x1)
            
        Returns:
            tuple: Preprocessed (X_train, y_train, X_test, y_test)
        """
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        X_train_processed = self.X_train.copy()
        X_test_processed = self.X_test.copy()
        
        # Normalize pixel values to [0, 1]
        if normalize:
            X_train_processed = X_train_processed.astype('float32') / 255.0
            X_test_processed = X_test_processed.astype('float32') / 255.0
            print("Data normalized to [0, 1] range")
        
        # Reshape for CNN if requested
        if reshape_for_cnn:
            X_train_processed = X_train_processed.reshape(-1, 28, 28, 1)
            X_test_processed = X_test_processed.reshape(-1, 28, 28, 1)
            print("Data reshaped for CNN: (samples, 28, 28, 1)")
        
        return X_train_processed, self.y_train, X_test_processed, self.y_test
    
    def get_class_distribution(self):
        """
        Get the distribution of classes in the dataset
        
        Returns:
            dict: Class distribution for train and test sets
        """
        if self.y_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        train_dist = pd.Series(self.y_train).value_counts().sort_index()
        test_dist = pd.Series(self.y_test).value_counts().sort_index()
        
        return {
            'train': train_dist.to_dict(),
            'test': test_dist.to_dict()
        }
    
    def visualize_samples(self, num_samples=10, save_path=None):
        """
        Visualize random samples from the dataset
        
        Args:
            num_samples (int): Number of samples to visualize
            save_path (str): Path to save the visualization
        """
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Select random samples
        indices = np.random.choice(len(self.X_train), num_samples, replace=False)
        
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            # Reshape pixel data to 28x28
            image = self.X_train[idx].reshape(28, 28)
            label = self.y_train[idx]
            
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f'Label: {label}')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def create_validation_split(self, validation_size=0.2, random_state=42):
        """
        Create a validation split from training data
        
        Args:
            validation_size (float): Proportion of training data for validation
            random_state (int): Random state for reproducibility
            
        Returns:
            tuple: (X_train_split, X_val, y_train_split, y_val)
        """
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            self.X_train, self.y_train, 
            test_size=validation_size, 
            random_state=random_state,
            stratify=self.y_train
        )
        
        print(f"Training split: {X_train_split.shape}")
        print(f"Validation split: {X_val.shape}")
        
        return X_train_split, X_val, y_train_split, y_val

def main():
    """
    Test the data loader functionality
    """
    # Initialize data loader
    loader = MNISTDataLoader()
    
    # Load data
    X_train, y_train, X_test, y_test = loader.load_data()
    
    # Get class distribution
    distribution = loader.get_class_distribution()
    print("\nClass distribution:")
    print("Training set:", distribution['train'])
    print("Test set:", distribution['test'])
    
    # Visualize samples
    print("\nVisualizing sample images...")
    loader.visualize_samples(num_samples=10)
    
    # Test preprocessing
    print("\nTesting preprocessing...")
    X_train_norm, y_train, X_test_norm, y_test = loader.preprocess_data(normalize=True)
    print(f"Normalized data range: [{X_train_norm.min():.3f}, {X_train_norm.max():.3f}]")
    
    # Test CNN reshape
    X_train_cnn, y_train, X_test_cnn, y_test = loader.preprocess_data(
        normalize=True, reshape_for_cnn=True
    )
    print(f"CNN data shape: {X_train_cnn.shape}")

if __name__ == "__main__":
    main()
