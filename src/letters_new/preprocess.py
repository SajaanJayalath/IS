import cv2
import numpy as np


def normalize_background(gray: np.ndarray) -> np.ndarray:
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    center = gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    if center.size > 0 and border.mean() < center.mean():
        gray = 255 - gray
    return gray


def binarize(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 25, 10)
    return thr


def enhance_strokes(binary: np.ndarray) -> np.ndarray:
    # Work with white foreground
    inv = 255 - binary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=1)
    dil = cv2.dilate(closed, kernel, iterations=1)
    out = 255 - dil
    return out


def crop_and_pad(binary: np.ndarray, target: int = 28) -> np.ndarray:
    ys, xs = np.where(binary < 128)
    if ys.size == 0:
        return np.full((target, target), 255, dtype=np.uint8)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = binary[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    scale = min((target - 2) / w, (target - 2) / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target, target), 255, dtype=np.uint8)
    yoff = (target - new_h) // 2
    xoff = (target - new_w) // 2
    canvas[yoff:yoff + new_h, xoff:xoff + new_w] = resized
    return canvas


def preprocess_single(image: np.ndarray) -> np.ndarray:
    gray = normalize_background(image)
    binary = binarize(gray)
    thick = enhance_strokes(binary)
    patch = crop_and_pad(thick, 28)
    # Convert to black text on white background, normalised [0,1]
    if patch.mean() < 127:
        patch = 255 - patch
    return (patch.astype(np.float32) / 255.0).reshape(28, 28, 1)

