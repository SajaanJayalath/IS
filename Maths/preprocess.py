from __future__ import annotations

import io
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps
import cv2


def _resize_with_padding(gray: np.ndarray, size: Tuple[int, int] = (28, 28), pad: int = 4) -> np.ndarray:
    """
    Resize grayscale image to target size preserving aspect ratio with padding.
    Adds a small margin so symbols are not touching the border.
    Expects input as uint8 [0,255], returns float32 [0,1].
    """
    h, w = gray.shape[:2]
    # Add margin
    gray = cv2.copyMakeBorder(gray, pad, pad, pad, pad, borderType=cv2.BORDER_CONSTANT, value=255)
    h, w = gray.shape[:2]

    # Compute scale preserving aspect
    target_h, target_w = size
    scale = min((target_w - 2) / w, (target_h - 2) / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Place on white background
    canvas = np.full((target_h, target_w), 255, dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    # Normalize to [0,1]
    return (canvas.astype(np.float32) / 255.0)


def _segment_core(pil_image: Image.Image) -> Tuple[List[Tuple[int,int,int,int]], np.ndarray, np.ndarray]:
    """Return sorted bounding boxes and intermediate images using a
    connected-components + projection splitting approach.

    Output: (boxes_ltr, thresh_image, gray_image)
    """
    # Convert to grayscale NP array
    img = pil_image.convert("L")
    gray = np.array(img)

    # Ensure white background, black strokes
    if gray.mean() < 127:
        gray = 255 - gray

    # Adaptive threshold is more robust to stroke thickness and lighting
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    th = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=10,
    )

    # Morph close small gaps, open tiny noise
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Connected components yields more stable symbols than raw contours
    num, labels, stats, _centroids = cv2.connectedComponentsWithStats(th, connectivity=8)
    H, W = th.shape
    min_area = max(10, (H * W) // 2000)

    boxes: List[Tuple[int,int,int,int]] = []

    def split_wide(x, y, w, h, depth=0):
        # For very wide regions, try splitting via vertical projection
        MAX_DEPTH = 2
        if depth >= MAX_DEPTH:
            boxes.append((x, y, w, h))
            return
        if w <= int(1.35 * h):
            boxes.append((x, y, w, h))
            return
        roi = th[y:y+h, x:x+w]
        # Column sums; low values indicate gaps between symbols
        proj = roi.sum(axis=0)
        # Normalize
        if proj.size == 0:
            boxes.append((x, y, w, h))
            return
        m = proj.max() if proj.max() > 0 else 1
        ratio = proj / float(m)
        # Find long valley around 0 where to split
        thresh = 0.12
        gaps = np.where(ratio < thresh)[0]
        if gaps.size == 0:
            boxes.append((x, y, w, h))
            return
        # Choose split at the widest gap segment
        # Group consecutive indices
        starts = [int(gaps[0])]
        ends = []
        for i in range(1, len(gaps)):
            if gaps[i] != gaps[i-1] + 1:
                ends.append(int(gaps[i-1]))
                starts.append(int(gaps[i]))
        ends.append(int(gaps[-1]))
        lengths = [e - s + 1 for s, e in zip(starts, ends)]
        idx = int(np.argmax(lengths))
        split_col = (starts[idx] + ends[idx]) // 2
        # Avoid degenerate splits
        if split_col <= 2 or split_col >= w - 3:
            boxes.append((x, y, w, h))
            return
        # Recurse on left and right
        split_wide(x, y, split_col, h, depth+1)
        split_wide(x + split_col, y, w - split_col, h, depth+1)

    for i in range(1, num):  # skip background=0
        x, y, w, h, area = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3], stats[i, 4]
        if area < min_area or w <= 1 or h <= 1:
            continue
        split_wide(int(x), int(y), int(w), int(h), depth=0)

    if not boxes:
        return [], th, gray

    # Sort left-to-right
    boxes.sort(key=lambda b: b[0])
    return boxes, th, gray


def segment_symbols(pil_image: Image.Image) -> List[np.ndarray]:
    """
    Segment a PIL image of a drawn expression into individual symbol images.
    Returns a list of 28x28 grayscale arrays normalized to [0,1].
    """
    boxes, th, gray = _segment_core(pil_image)
    if not boxes:
        return []

    # Extract, invert to black-on-white, resize with padding
    symbol_images: List[np.ndarray] = []
    for (x, y, w, h) in boxes:
        roi = gray[y : y + h, x : x + w]
        # Ensure black strokes on white background
        # Decide by mean inside ROI after thresholding
        roi_bin = th[y : y + h, x : x + w]
        # roi_bin is 255 where strokes, since we used THRESH_BINARY_INV
        # Convert to black-on-white grayscale consistent with dataset (black strokes ~0)
        mask = roi_bin > 0
        roi_clean = np.full_like(roi, 255)
        roi_clean[mask] = 0
        symbol_images.append(_resize_with_padding(roi_clean))

    return symbol_images


def segment_symbols_with_debug(pil_image: Image.Image):
    """Return (symbol_images, boxes_ltr, thresh_image, gray_image)."""
    boxes, th, gray = _segment_core(pil_image)
    imgs = []
    for (x, y, w, h) in boxes:
        roi = gray[y : y + h, x : x + w]
        roi_bin = th[y : y + h, x : x + w]
        mask = roi_bin > 0
        roi_clean = np.full_like(roi, 255)
        roi_clean[mask] = 0
        imgs.append(_resize_with_padding(roi_clean))
    return imgs, boxes, th, gray


def image_from_bytes(data: bytes) -> Image.Image:
    """Create PIL Image from raw bytes (utility)."""
    return Image.open(io.BytesIO(data)).convert("RGBA")
