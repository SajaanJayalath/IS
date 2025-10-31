"""
Dataset preparation utilities for arithmetic symbol recognition.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable

from .constants import CANONICAL_SYMBOLS, RAW_SYMBOL_MAP
from .preprocess import ArithmeticPreprocessor


class ArithmeticDatasetBuilder:
    """
    Convert the raw Kaggle symbol dataset into MNIST-shaped numpy arrays ready for training.
    """

    def __init__(self, target_size: int = 28) -> None:
        self.preprocessor = ArithmeticPreprocessor(target_size=target_size)
        self.target_size = target_size

    def _iter_samples(self, root: Path) -> Iterable[Tuple[str, Path]]:
        for subdir in sorted(root.iterdir()):
            if not subdir.is_dir():
                continue
            label = RAW_SYMBOL_MAP.get(subdir.name)
            if label is None:
                continue
            for file in subdir.glob("*.*"):
                yield label, file

    def build_arrays(self, root: str | os.PathLike[str]) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
        """
        Process all samples under ``root`` and return images/labels arrays.
        """
        root_path = Path(root)
        label_tokens: List[str] = []
        images = []

        for label, file in tqdm(list(self._iter_samples(root_path)), desc="Preparing symbols"):
            raw = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
            if raw is None:
                continue
            processed = self.preprocessor.normalise(raw)
            images.append(processed.reshape(self.target_size, self.target_size, 1))
            label_tokens.append(label)

        if not images:
            raise RuntimeError(f"No valid samples found under {root_path}")

        # Build mapping using canonical order but only for encountered symbols.
        encountered = set(label_tokens)
        ordered_symbols = [symbol for symbol in CANONICAL_SYMBOLS if symbol in encountered]
        if len(ordered_symbols) != len(encountered):
            # Include any non-canonical leftovers (sorted for determinism)
            extras = sorted(encountered.difference(set(ordered_symbols)))
            ordered_symbols.extend(extras)
        label_mapping = {idx: symbol for idx, symbol in enumerate(ordered_symbols)}
        symbol_to_index = {symbol: idx for idx, symbol in label_mapping.items()}

        indices = [symbol_to_index[token] for token in label_tokens]

        X = np.stack(images, axis=0).astype(np.float32)
        y = np.array(indices, dtype=np.int64)
        return X, y, label_mapping

    def save_npz(self, root: str | os.PathLike[str], output: str) -> Dict[int, str]:
        X, y, mapping = self.build_arrays(root)
        np.savez_compressed(output, images=X, labels=y)
        return mapping

    def dump_label_mapping(self, mapping: Dict[int, str], path: str | os.PathLike[str]) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({int(k): v for k, v in mapping.items()}, handle, indent=2)


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    validation_fraction: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified train/validation split.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=random_state)
    train_idx, val_idx = next(splitter.split(X, y))
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def balance_classes(X: np.ndarray, y: np.ndarray, max_per_class: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample classes with excessive examples to keep training balanced.
    """
    rng = np.random.default_rng(17)
    unique_labels = np.unique(y)
    indices: List[int] = []
    for label in unique_labels:
        label_idx = np.where(y == label)[0]
        if label_idx.size > max_per_class:
            label_idx = rng.choice(label_idx, size=max_per_class, replace=False)
        indices.extend(label_idx.tolist())
    indices = rng.permutation(indices)
    return X[indices], y[indices]
