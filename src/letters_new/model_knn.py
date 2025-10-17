from __future__ import annotations

import os
import struct
from typing import Dict, Tuple

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle


EMNIST_DIR = os.path.join('MNIST_CSV', 'EMNIST', 'raw')


def _read_idx_images(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, rows * cols)


def _read_idx_labels(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def load_emnist_letters(max_samples: int = 20000) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    images_path = os.path.join(EMNIST_DIR, 'emnist-letters-train-images-idx3-ubyte')
    labels_path = os.path.join(EMNIST_DIR, 'emnist-letters-train-labels-idx1-ubyte')
    X = _read_idx_images(images_path)
    y = _read_idx_labels(labels_path)
    # EMNIST letters labels are 1..26 mapping to 'a'..'z'
    y = y.astype(np.int64)
    mapping = {i: chr(ord('a') + i - 1) for i in range(1, 27)}
    # Subsample for speed
    if max_samples and X.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X, y = X[idx], y[idx]
    # Normalise
    X = (X.astype(np.float32) / 255.0)
    return X, y, mapping


def train_knn(max_samples: int = 20000, n_neighbors: int = 5) -> Tuple[KNeighborsClassifier, Dict[int, str]]:
    X, y, mapping = load_emnist_letters(max_samples)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    knn.fit(X, y)
    return knn, mapping


def save_model(model: KNeighborsClassifier, mapping: Dict[int, str], out_dir: str = 'models_new') -> str:
    os.makedirs(out_dir, exist_ok=True)
    payload = {'model': model, 'mapping': mapping}
    path = os.path.join(out_dir, 'letters_knn.pkl')
    with open(path, 'wb') as f:
        pickle.dump(payload, f)
    return path


def load_model(path: str = os.path.join('models_new', 'letters_knn.pkl')) -> Tuple[KNeighborsClassifier, Dict[int, str]]:
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    return payload['model'], payload['mapping']

