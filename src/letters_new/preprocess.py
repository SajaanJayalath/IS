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
    """Robust binarization with Gaussian blur and Otsu fallback.

    Returns a binary image with black ink on white background.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    try:
        thr = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 25, 10
        )
    except Exception:
        _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thr


def enhance_strokes(binary: np.ndarray) -> np.ndarray:
    """Enhance strokes and remove small specks.

    Keeps black ink on white background.
    """
    inv = 255 - binary  # foreground white
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=1)
    # Remove tiny noise components
    num, labels, stats, _ = cv2.connectedComponentsWithStats((closed > 0).astype(np.uint8), 8)
    mask = np.zeros_like(closed)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 10:
            mask[labels == i] = 255
    dil = cv2.dilate(mask, kernel, iterations=1)
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


def _deskew(gray_or_binary: np.ndarray) -> np.ndarray:
    """Deskew image using min-area rectangle on foreground mask."""
    if gray_or_binary.max() <= 1:
        g = (gray_or_binary * 255).astype(np.uint8)
    else:
        g = gray_or_binary.astype(np.uint8)
    # Build binary with white strokes
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return gray_or_binary
    largest = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    angle = rect[2]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:
        return gray_or_binary
    (h, w) = g.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def _center_of_mass(gray_or_binary: np.ndarray) -> np.ndarray:
    """Center foreground mass to image center."""
    if gray_or_binary.max() <= 1:
        g = (gray_or_binary * 255).astype(np.uint8)
    else:
        g = gray_or_binary.astype(np.uint8)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() > 127:
        bw = 255 - bw
    M = cv2.moments(bw)
    if abs(M.get("m00", 0)) < 1e-3:
        return gray_or_binary
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    h, w = g.shape[:2]
    sx = int(round((w // 2) - cx))
    sy = int(round((h // 2) - cy))
    Mat = np.float32([[1, 0, sx], [0, 1, sy]])
    shifted = cv2.warpAffine(g, Mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=255)
    return shifted


def preprocess_single(image: np.ndarray, target: int = 28) -> np.ndarray:
    """Stronger preprocessing for letters: normalize, binarize, deskew, center, pad.

    Returns a (target, target, 1) float32 array in [0,1] with white background.
    """
    gray = normalize_background(image)
    # Light CLAHE to help low contrast
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    except Exception:
        pass
    # Binarize, enhance, deskew, center
    binary = binarize(gray)
    enhanced = enhance_strokes(binary)
    deskewed = _deskew(enhanced)
    centered = _center_of_mass(deskewed)
    patch = crop_and_pad(centered, target)
    # Ensure white background
    if patch.mean() < 127:
        patch = 255 - patch
    return (patch.astype(np.float32) / 255.0).reshape(target, target, 1)
