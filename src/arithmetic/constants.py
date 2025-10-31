"""
Label configuration and defaults for the arithmetic symbol recogniser.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

# Canonical vocabulary for arithmetic expressions.
CANONICAL_SYMBOLS: List[str] = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "+",
    "-",
    "*",
    "/",
    "÷",
    "=",
    "(",
    ")",
    ".",
]

# Mapping from raw folder names in the Kaggle dataset to canonical symbols.
RAW_SYMBOL_MAP: Dict[str, str] = {
    "0": "0",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "+": "+",
    "-": "-",
    "=": "=",
    "times": "*",
    "*": "*",
    "x": "*",
    "forward_slash": "/",
    "div": "÷",
    ")": ")",
    "(": "(",
    "dot": ".",
    ".": ".",
}

# Friendly display names for GUI dropdowns.
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "cnn": "CNN (Arithmetic)",
    "ensemble": "Ensemble",
    # External repo option (wired via GUI adapter)
    "mer": "MER (External)",
}


@dataclass(frozen=True)
class DatasetSplitConfig:
    """Paths for prepared dataset splits."""

    train_images: str
    train_labels: str
    val_images: str
    val_labels: str


def default_label_mapping() -> Dict[int, str]:
    """
    Deterministic index->symbol mapping used for saved models.
    """
    return {idx: symbol for idx, symbol in enumerate(CANONICAL_SYMBOLS)}
