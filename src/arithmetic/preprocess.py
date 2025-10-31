"""
Preprocessing utilities tuned for the handwritten arithmetic symbols dataset.
"""

from __future__ import annotations

import cv2
import numpy as np


class ArithmeticPreprocessor:
    """Pipeline for normalising arithmetic glyphs to 28x28 white-background images."""

    def __init__(self, target_size: int = 28) -> None:
        self.target_size = int(target_size)

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def normalise_background(self, image: np.ndarray) -> np.ndarray:
        """Ensure black ink on white background."""
        border = np.concatenate(
            [image[0, :], image[-1, :], image[:, 0], image[:, -1]]
        )
        ink_is_dark = border.mean() > image.mean()
        if ink_is_dark:
            return 255 - image
        return image

    def denoise(self, image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, (3, 3), sigmaX=0.5)

    def binarise(self, image: np.ndarray) -> np.ndarray:
        """Adaptive threshold that preserves faint strokes."""
        return cv2.adaptiveThreshold(
            image,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=2,
        )

    def morphological_refine(self, binary: np.ndarray) -> np.ndarray:
        """Remove noise and reinforce thin operators like '-'."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        reinforced = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_line, iterations=1)
        return reinforced

    def deskew(self, binary: np.ndarray) -> np.ndarray:
        moments = cv2.moments(binary)
        if abs(moments["mu02"]) < 1e-2:
            return binary
        skew = moments["mu11"] / moments["mu02"]
        rows, cols = binary.shape
        M = np.array([[1, skew, -0.5 * cols * skew], [0, 1, 0]], dtype=np.float32)
        deskewed = cv2.warpAffine(binary, M, (cols, rows), flags=cv2.INTER_LINEAR)
        return deskewed

    def extract_square(self, binary: np.ndarray) -> np.ndarray:
        ys, xs = np.where(binary > 0)
        if ys.size == 0 or xs.size == 0:
            return np.zeros((self.target_size, self.target_size), dtype=np.uint8)
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        crop = binary[y0 : y1 + 1, x0 : x1 + 1]
        h, w = crop.shape
        scale = (self.target_size - 4) / max(h, w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
        y_offset = (self.target_size - new_h) // 2
        x_offset = (self.target_size - new_w) // 2
        canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
        return canvas

    def normalise(self, image: np.ndarray) -> np.ndarray:
        """Full pipeline from raw image to 28x28 float32 array."""
        gray = self.to_grayscale(image)
        clean = self.normalise_background(gray)
        blurred = self.denoise(clean)
        binary = self.binarise(blurred)
        refined = self.morphological_refine(binary)
        deskewed = self.deskew(refined)
        square = self.extract_square(deskewed)
        return square.astype(np.float32) / 255.0


def preprocess_patch(patch: np.ndarray, target_size: int = 28) -> np.ndarray:
    """Utility for composing preprocessor steps when only isolated patches are available."""
    pre = ArithmeticPreprocessor(target_size=target_size)
    if patch.ndim == 2:
        base = patch
    else:
        base = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return pre.normalise(base)
