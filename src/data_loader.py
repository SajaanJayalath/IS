"""
Data loading and preprocessing module for HNRS

Adds flexible dataset options:
- MNIST CSV (existing)
- Image Folder dataset with class subfolders (0..9)

Both loaders expose a similar API:
- load_data() -> (X_train, y_train, X_test, y_test)
- preprocess_data(normalize=True, reshape_for_cnn=False)
"""

import os
import glob
from typing import Tuple, Optional, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2

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


class ImageFolderDataLoader:
    """
    Loads digit images from a folder structure:

    root/
      train/
        0/ ... image files
        1/
        ...
        9/
      test/
        0/ ...
        ...

    Optional: if "test/" is missing, creates a train/test split from train/.
    Supports common image extensions (png, jpg, jpeg, bmp, tif, tiff).
    """

    IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")

    def __init__(self, root_dir: str, image_size: int = 28):
        self.root_dir = root_dir
        self.image_size = image_size
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

    def _collect_files(self, split: str) -> List[Tuple[str, int]]:
        split_dir = os.path.join(self.root_dir, split)
        if not os.path.isdir(split_dir):
            return []

        files_with_labels: List[Tuple[str, int]] = []
        for label_str in map(str, range(10)):
            class_dir = os.path.join(split_dir, label_str)
            if not os.path.isdir(class_dir):
                continue
            for pat in self.IMG_EXTS:
                for path in glob.glob(os.path.join(class_dir, pat)):
                    files_with_labels.append((path, int(label_str)))
        return files_with_labels

    def _read_and_preprocess(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")

        # Normalize orientation: make foreground bright. Heuristic using border vs center mean.
        h, w = img.shape[:2]
        border = np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]])
        if border.mean() < img[h//4:3*h//4, w//4:3*w//4].mean():
            img = 255 - img

        # Resize with aspect-ratio padding into square
        target = self.image_size
        scale = min(target / w, target / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((target, target), dtype=np.uint8)
        y0 = (target - new_h) // 2
        x0 = (target - new_w) // 2
        canvas[y0:y0+new_h, x0:x0+new_w] = resized
        return canvas

    def _load_split(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        files = self._collect_files(split)
        if not files:
            return np.empty((0, self.image_size * self.image_size), dtype=np.float32), np.empty((0,), dtype=np.int64)

        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        for path, label in files:
            img28 = self._read_and_preprocess(path)
            X_list.append(img28.reshape(-1))
            y_list.append(label)
        X = np.stack(X_list, axis=0).astype(np.float32)
        y = np.array(y_list, dtype=np.int64)
        return X, y

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        print(f"Loading ImageFolder dataset from: {self.root_dir}")
        X_train, y_train = self._load_split("train")
        X_test, y_test = self._load_split("test")

        if X_train.size == 0:
            raise FileNotFoundError(
                f"No training images found. Expected structure root/train/0..9 with image files in {self.root_dir}"
            )

        # If no explicit test set, split from train
        if X_test.size == 0:
            print("No explicit test set found. Creating 80/20 split from train...")
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )

        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

        print(f"Training data shape: {self.X_train.shape}")
        print(f"Test data shape: {self.X_test.shape}")

        return self.X_train, self.y_train, self.X_test, self.y_test

    def preprocess_data(self, normalize: bool = True, reshape_for_cnn: bool = False):
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        X_train = self.X_train.copy()
        X_test = self.X_test.copy()

        if normalize:
            X_train = X_train.astype("float32") / 255.0
            X_test = X_test.astype("float32") / 255.0

        if reshape_for_cnn:
            X_train = X_train.reshape(-1, self.image_size, self.image_size, 1)
            X_test = X_test.reshape(-1, self.image_size, self.image_size, 1)

        return X_train, self.y_train, X_test, self.y_test

    def get_class_distribution(self):
        if self.y_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return {
            'train': pd.Series(self.y_train).value_counts().sort_index().to_dict(),
            'test': pd.Series(self.y_test).value_counts().sort_index().to_dict(),
        }

    def visualize_samples(self, num_samples: int = 10, save_path: Optional[str] = None):
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        indices = np.random.choice(len(self.X_train), min(num_samples, len(self.X_train)), replace=False)
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.ravel()
        for i, idx in enumerate(indices):
            image = self.X_train[idx].reshape(self.image_size, self.image_size)
            label = self.y_train[idx]
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f'Label: {label}')
            axes[i].axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        plt.show()


class _TorchvisionBase:
    """Common helpers for torchvision-backed datasets."""

    def __init__(self, root_dir: str, image_size: int = 28):
        self.root_dir = root_dir
        self.image_size = image_size
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

    def _ensure_tv(self):
        try:
            import torchvision  # type: ignore
        except Exception as e:
            raise ImportError(
                "torchvision is required for this dataset. Install with 'pip install torchvision'."
            ) from e

    def preprocess_data(self, normalize: bool = True, reshape_for_cnn: bool = False):
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        if normalize:
            X_train = X_train.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0
        if reshape_for_cnn:
            X_train = X_train.reshape(-1, self.image_size, self.image_size, 1)
            X_test = X_test.reshape(-1, self.image_size, self.image_size, 1)
        return X_train, self.y_train, X_test, self.y_test

    def get_class_distribution(self):
        if self.y_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return {
            'train': pd.Series(self.y_train).value_counts().sort_index().to_dict(),
            'test': pd.Series(self.y_test).value_counts().sort_index().to_dict(),
        }


class EMNISTDigitsLoader(_TorchvisionBase):
    """
    EMNIST 'digits' split via torchvision. Applies rotate+flip to match MNIST orientation.
    """

    def load_data(self):
        self._ensure_tv()
        from torchvision import datasets  # type: ignore

        os.makedirs(self.root_dir, exist_ok=True)
        train_ds = datasets.EMNIST(self.root_dir, split='digits', train=True, download=True)
        test_ds = datasets.EMNIST(self.root_dir, split='digits', train=False, download=True)

        def to_np(img_pil) -> np.ndarray:
            img = np.array(img_pil)
            img = np.rot90(img, k=1)
            img = np.fliplr(img)
            return img

        X_train = np.stack([to_np(img) for img, _ in train_ds], axis=0).reshape(len(train_ds), -1).astype(np.float32)
        y_train = np.array([int(lbl) for _, lbl in train_ds], dtype=np.int64)

        X_test = np.stack([to_np(img) for img, _ in test_ds], axis=0).reshape(len(test_ds), -1).astype(np.float32)
        y_test = np.array([int(lbl) for _, lbl in test_ds], dtype=np.int64)

        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test
        print(f"Loaded EMNIST digits: train {self.X_train.shape}, test {self.X_test.shape}")
        return self.X_train, self.y_train, self.X_test, self.y_test


class SVHNLoader(_TorchvisionBase):
    """
    SVHN via torchvision. Converts RGB 32x32 to grayscale 28x28 and remaps labels (10 -> 0).
    """

    def load_data(self):
        self._ensure_tv()
        from torchvision import datasets  # type: ignore

        os.makedirs(self.root_dir, exist_ok=True)
        train_ds = datasets.SVHN(self.root_dir, split='train', download=True)
        test_ds = datasets.SVHN(self.root_dir, split='test', download=True)

        def to_np(img_pil) -> np.ndarray:
            img = np.array(img_pil)
            if img.ndim == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img.astype(np.uint8)
            gray = cv2.resize(gray, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            h, w = gray.shape
            border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
            if border.mean() < gray[h//4:3*h//4, w//4:3*w//4].mean():
                gray = 255 - gray
            return gray

        def map_label(y):
            return 0 if int(y) == 10 else int(y)

        X_train = np.stack([to_np(img) for img, _ in train_ds], axis=0).reshape(len(train_ds), -1).astype(np.float32)
        y_train = np.array([map_label(lbl) for _, lbl in train_ds], dtype=np.int64)

        X_test = np.stack([to_np(img) for img, _ in test_ds], axis=0).reshape(len(test_ds), -1).astype(np.float32)
        y_test = np.array([map_label(lbl) for _, lbl in test_ds], dtype=np.int64)

        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test
        print(f"Loaded SVHN: train {self.X_train.shape}, test {self.X_test.shape}")
        return self.X_train, self.y_train, self.X_test, self.y_test


class CombinedDataLoader:
    """
    Combined dataset loader that merges MNIST (CSV), EMNIST digits, and SVHN
    into a single training/testing set with unified preprocessing.

    Balancing strategy: samples an equal number from each dataset up to a cap
    to avoid dominance and excessive memory use.
    """

    def __init__(self, image_size: int = 28, train_cap_per_ds: int = 60000, test_cap_per_ds: int = 10000):
        self.image_size = image_size
        self.train_cap_per_ds = int(train_cap_per_ds)
        self.test_cap_per_ds = int(test_cap_per_ds)
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

    def _stack_and_sample(self, X: np.ndarray, y: np.ndarray, k: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        n = len(X)
        if n <= k:
            return X, y
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, k, replace=False)
        return X[idx], y[idx]

    def load_data(self):
        # Load component datasets
        mnist = MNISTDataLoader(data_dir='MNIST_CSV')
        Xtr_m, ytr_m, Xte_m, yte_m = mnist.load_data()

        emnist = EMNISTDigitsLoader(root_dir=os.path.join('data', 'emnist'))
        Xtr_e, ytr_e, Xte_e, yte_e = emnist.load_data()

        svhn = SVHNLoader(root_dir=os.path.join('data', 'svhn'))
        Xtr_s, ytr_s, Xte_s, yte_s = svhn.load_data()

        # Determine balanced sample sizes
        k_tr = min(self.train_cap_per_ds, len(Xtr_m), len(Xtr_e), len(Xtr_s))
        k_te = min(self.test_cap_per_ds, len(Xte_m), len(Xte_e), len(Xte_s))

        # Sample equally from each dataset for balance
        Xtr_m_s, ytr_m_s = self._stack_and_sample(Xtr_m, ytr_m, k_tr, seed=1)
        Xtr_e_s, ytr_e_s = self._stack_and_sample(Xtr_e, ytr_e, k_tr, seed=2)
        Xtr_s_s, ytr_s_s = self._stack_and_sample(Xtr_s, ytr_s, k_tr, seed=3)

        Xte_m_s, yte_m_s = self._stack_and_sample(Xte_m, yte_m, k_te, seed=4)
        Xte_e_s, yte_e_s = self._stack_and_sample(Xte_e, yte_e, k_te, seed=5)
        Xte_s_s, yte_s_s = self._stack_and_sample(Xte_s, yte_s, k_te, seed=6)

        # Concatenate
        X_train = np.vstack([Xtr_m_s, Xtr_e_s, Xtr_s_s]).astype(np.float32)
        y_train = np.concatenate([ytr_m_s, ytr_e_s, ytr_s_s]).astype(np.int64)
        X_test = np.vstack([Xte_m_s, Xte_e_s, Xte_s_s]).astype(np.float32)
        y_test = np.concatenate([yte_m_s, yte_e_s, yte_s_s]).astype(np.int64)

        # Shuffle
        rng = np.random.default_rng(123)
        perm_tr = rng.permutation(len(X_train))
        perm_te = rng.permutation(len(X_test))
        self.X_train, self.y_train = X_train[perm_tr], y_train[perm_tr]
        self.X_test, self.y_test = X_test[perm_te], y_test[perm_te]

        print(f"Loaded Combined dataset: train {self.X_train.shape}, test {self.X_test.shape} (per-dataset cap {k_tr}/{k_te})")
        return self.X_train, self.y_train, self.X_test, self.y_test

    def preprocess_data(self, normalize: bool = True, reshape_for_cnn: bool = False):
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        X_train = self.X_train.copy()
        X_test = self.X_test.copy()
        if normalize:
            X_train = X_train.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0
        if reshape_for_cnn:
            X_train = X_train.reshape(-1, self.image_size, self.image_size, 1)
            X_test = X_test.reshape(-1, self.image_size, self.image_size, 1)
        return X_train, self.y_train, X_test, self.y_test

    def get_class_distribution(self):
        if self.y_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return {
            'train': pd.Series(self.y_train).value_counts().sort_index().to_dict(),
            'test': pd.Series(self.y_test).value_counts().sort_index().to_dict(),
        }


def get_data_loader(dataset: str, data_dir: Optional[str] = None):
    """
    Factory for dataset loaders.

    - dataset == 'mnist_csv': expects CSVs in <data_dir> with mnist_train.csv and mnist_test.csv
    - dataset == 'image_folder': expects class subfolders 0..9 under train/ and optional test/
    - dataset == 'emnist_digits': downloads EMNIST digits via torchvision
    - dataset == 'svhn': downloads SVHN via torchvision
    - dataset == 'combined': concatenates MNIST CSV + EMNIST(digits) + SVHN (balanced sample)
    """
    dataset = (dataset or 'mnist_csv').lower()
    if dataset == 'mnist_csv':
        return MNISTDataLoader(data_dir=data_dir or 'MNIST_CSV')
    elif dataset == 'image_folder':
        if not data_dir:
            raise ValueError("For dataset 'image_folder', please provide --data-dir pointing to the root folder.")
        return ImageFolderDataLoader(root_dir=data_dir)
    elif dataset == 'emnist_digits':
        return EMNISTDigitsLoader(root_dir=data_dir or os.path.join('data', 'emnist'))
    elif dataset == 'svhn':
        return SVHNLoader(root_dir=data_dir or os.path.join('data', 'svhn'))
    elif dataset == 'combined':
        return CombinedDataLoader()
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'mnist_csv', 'image_folder', 'emnist_digits', or 'svhn'.")

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
