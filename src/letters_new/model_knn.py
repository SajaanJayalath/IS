from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle

from data_loader import NISTByClassLoader


DEFAULT_BYCLASS_ROOT = os.path.join('data', 'by_class', 'by_class')


def load_byclass_letters(
    root_dir: str = DEFAULT_BYCLASS_ROOT,
    include_digits: bool = False,
    include_uppercase: bool = True,
    include_lowercase: bool = True,
    max_per_class: int | None = 2000,
    max_total: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    loader = NISTByClassLoader(
        root_dir=root_dir,
        include_digits=include_digits,
        include_uppercase=include_uppercase,
        include_lowercase=include_lowercase,
        max_per_class=max_per_class,
        test_size=0.2,
        random_state=42,
    )
    X_train_raw, y_train, X_test_raw, y_test = loader.load_data()
    X_train, y_train, _, _ = loader.preprocess_data(normalize=True, reshape_for_cnn=False)
    mapping = loader.get_label_mapping()
    X = X_train.astype(np.float32) / 255.0 if X_train.max() > 1 else X_train.astype(np.float32)
    y = y_train.astype(np.int64)

    if max_total and X.shape[0] > max_total:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], size=max_total, replace=False)
        X, y = X[idx], y[idx]
    return X, y, mapping


def train_knn_byclass(
    root_dir: str = DEFAULT_BYCLASS_ROOT,
    max_per_class: int | None = 2000,
    max_total: int | None = None,
    n_neighbors: int = 5,
) -> Tuple[KNeighborsClassifier, Dict[int, str]]:
    X, y, mapping = load_byclass_letters(
        root_dir=root_dir,
        include_digits=False,
        include_uppercase=True,
        include_lowercase=True,
        max_per_class=max_per_class,
        max_total=max_total,
    )
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    knn.fit(X, y)
    return knn, mapping


def save_model(model: KNeighborsClassifier, mapping: Dict[int, str], out_dir: str = 'models_new') -> str:
    os.makedirs(out_dir, exist_ok=True)
    payload = {'model': model, 'mapping': mapping}
    path = os.path.join(out_dir, 'letters_byclass_knn.pkl')
    with open(path, 'wb') as f:
        pickle.dump(payload, f)
    return path


def load_model(path: str = os.path.join('models_new', 'letters_byclass_knn.pkl')) -> Tuple[KNeighborsClassifier, Dict[int, str]]:
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    return payload['model'], payload['mapping']
